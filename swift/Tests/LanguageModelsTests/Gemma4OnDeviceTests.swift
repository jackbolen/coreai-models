// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Testing
import Tokenizers

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Generable
private struct Arrival {
    let arrivesBeforeDeadline: Bool
}

@Generable
private struct Verdict: Equatable {
    let isRisky: Bool
}

/// Runs a static-shape Gemma 4 bundle, from `COREAI_TEST_GEMMA4_BUNDLE`. The tests share the one
/// engine a bundle gets, so they run one at a time.
@Suite(
    "Gemma 4 on device",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_GEMMA4_BUNDLE"] != nil))
struct Gemma4OnDeviceTests {
    private static var bundleURL: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["COREAI_TEST_GEMMA4_BUNDLE"] ?? "")
    }

    @Test("A guided request thinks, then decodes, and the thought stays out of the next prompt")
    func guidedRequestThinks() async throws {
        let model = try await CoreAILanguageModel(resourcesAt: Self.bundleURL)
        #expect(model.capabilities.contains(.reasoning))

        let session = LanguageModelSession(model: model, instructions: "Answer the question.")
        _ = try await session.respond(
            to: "A train leaves at 3:15 pm and the trip takes 2 hours 50 minutes. Does it arrive before 6:30 pm?",
            generating: Arrival.self,
            options: GenerationOptions(samplingMode: .greedy),
            contextOptions: ContextOptions(includeSchemaInPrompt: false, reasoningLevel: .moderate))

        var thought = ""
        for case .reasoning(let reasoning) in session.transcript {
            thought += reasoning.segments.compactMap {
                if case .text(let text) = $0 { return text.content }
                return nil
            }.joined()
        }
        #expect(!thought.isEmpty)
        print("\(Self.bundleURL.lastPathComponent): \(thought.count) characters of thought")

        let tokenizer = try await LanguageModelBundle(at: Self.bundleURL).loadTokenizer()
        let nextTurn = tokenizer.decode(
            tokens: CoreAILanguageModel.CoreAIExecutor.makeTokens(
                from: Array(session.transcript), using: tokenizer))
        #expect(!nextTurn.contains("<|channel>"))
        #expect(!nextTurn.contains(thought.prefix(80)))
    }

    /// Instructions long enough that replaying them costs far more than a short question and a short
    /// answer.
    private static let rules = (1...300)
        .map { "Rule \($0): item \($0) is \($0 % 7 == 0 ? "risky" : "routine")." }
        .joined(separator: "\n")

    @Test("A guided request resumes from the instructions an earlier request left behind")
    func guidedRequestResumes() async throws {
        let model = try await CoreAILanguageModel(resourcesAt: Self.bundleURL)
        func ask(_ item: Int) async throws -> (Verdict, seconds: Double) {
            let session = LanguageModelSession(model: model, instructions: Self.rules)
            let start = ContinuousClock.now
            let response = try await session.respond(
                to: "Is item \(item) risky?", generating: Verdict.self,
                options: GenerationOptions(samplingMode: .greedy),
                contextOptions: ContextOptions(includeSchemaInPrompt: false))
            return (response.content, start.duration(to: .now) / .seconds(1))
        }

        let first = try await ask(14)
        let resumed = try await ask(15)
        model.unload()
        let cold = try await ask(15)
        print(
            "\(Self.bundleURL.lastPathComponent): first \(first.seconds) s, resumed \(resumed.seconds) s, "
                + "cold \(cold.seconds) s")

        #expect(resumed.0 == cold.0)
        #expect(resumed.seconds * 1.5 < cold.seconds)
    }

    @Test("The first token after a resume has the logits of a cold run")
    func resumedLogitsMatchCold() async throws {
        let engine = try await CoreAIRunner(contentsOf: Self.bundleURL).makeInferenceEngine()
        let tokenizer = try await LanguageModelBundle(at: Self.bundleURL).loadTokenizer()
        func prompt(_ item: Int) throws -> [Int32] {
            try tokenizer.applyChatTemplate(messages: [
                ["role": "system", "content": Self.rules], ["role": "user", "content": "Is item \(item) risky?"],
            ]).map(Int32.init)
        }
        let system = try tokenizer.applyChatTemplate(messages: [["role": "system", "content": Self.rules]])
        let shared = zip(system.map(Int32.init), try prompt(14)).prefix { $0 == $1 }.count

        func firstLogits(_ input: [Int32]) async throws -> [Float] {
            engine.retainPrefix(length: shared)
            let output = try await engine.generate(
                with: input, samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: 1, logits: .lastPosition))
            for try await item in output { return (item.logits ?? []).map(Float.init) }
            return []
        }

        _ = try await firstLogits(prompt(14))
        let resumed = try await firstLogits(prompt(15))
        #expect(engine.lastPrefixHitCount > 0)
        try await engine.reset()
        let cold = try await firstLogits(prompt(15))

        #expect(cold.count > 1_000)
        #expect(zip(resumed, cold).map { abs($0 - $1) }.max() ?? .infinity < 0.05)
    }
}

#endif
