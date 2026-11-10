// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Generable
private struct Arrival {
    let arrivesBeforeDeadline: Bool
}

@Suite(
    "Gemma 4 reasoning with guided generation",
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_GEMMA4_BUNDLE"] != nil))
struct Gemma4ReasoningTests {
    @Test("A guided request thinks, then decodes, and the thought stays out of the next prompt")
    func guidedRequestThinks() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_TEST_GEMMA4_BUNDLE"])
        let url = URL(fileURLWithPath: path)
        let model = try await CoreAILanguageModel(resourcesAt: url)
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
        print("\(url.lastPathComponent): \(thought.count) characters of thought")

        let tokenizer = try await LanguageModelBundle(at: url).loadTokenizer()
        let nextTurn = tokenizer.decode(
            tokens: CoreAILanguageModel.CoreAIExecutor.makeTokens(
                from: Array(session.transcript), using: tokenizer))
        #expect(!nextTurn.contains("<|channel>"))
        #expect(!nextTurn.contains(thought.prefix(80)))
    }
}

#endif
