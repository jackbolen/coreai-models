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

        func firstLogits(_ engine: any InferenceEngine, _ input: [Int32]) async throws -> [Float] {
            engine.cachePrefixes(endingAt: [shared])
            let output = try await engine.generate(
                with: input, samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: 1, logits: .lastPosition))
            for try await item in output { return (item.logits ?? []).map(Float.init) }
            return []
        }

        _ = try await firstLogits(engine, prompt(14))
        let resumed = try await firstLogits(engine, prompt(15))
        #expect(engine.lastPrefixHitCount > 0)
        // The reference keeps no prefix, so it replays every token.
        let reference = try await EngineFactory.createEngine(
            bundle: LanguageModelBundle(at: Self.bundleURL), options: EngineOptions(prefixCacheBudgetBytes: 0))
        let cold = try await firstLogits(reference, prompt(15))

        #expect(cold.count > 1_000)
        #expect(zip(resumed, cold).map { abs($0 - $1) }.max() ?? .infinity < 0.05)
    }

    /// A stretch of transcript, long enough that replaying it costs far more than a question.
    private static func transcript(_ name: String, lines: Int = 50) -> String {
        (1...lines).map { "\(name) \($0): the tool call touched file \($0 * 7) and finished without error.\n" }
            .joined()
    }

    /// An engine that keeps prefixes beside a reference that keeps none, over one bundle, and the
    /// requests both of them run.
    private struct BranchRig {
        let cached: any InferenceEngine
        /// Its budget leaves no block and no snapshot behind, so it replays every token.
        let reference: any InferenceEngine
        let tokenizer: any Tokenizer
        let instructions: String

        init(
            bundleURL: URL, instructions: String = String(Gemma4OnDeviceTests.rules.prefix(6_000)),
            budgetBytes: Int = EngineOptions.defaultPrefixCacheBudgetBytes
        ) async throws {
            let bundle = try LanguageModelBundle(at: bundleURL)
            cached = try await EngineFactory.createEngine(
                bundle: bundle, options: EngineOptions(prefixCacheBudgetBytes: budgetBytes))
            reference = try await EngineFactory.createEngine(
                bundle: bundle, options: EngineOptions(prefixCacheBudgetBytes: 0))
            tokenizer = try await bundle.loadTokenizer()
            self.instructions = instructions
        }

        struct Outcome {
            let seconds: Double
            let referenceSeconds: Double
            /// The tokens of the prompt that the engine matched against what it kept.
            let matchedTokens: Int

            func expectResumes(_ name: String) {
                #expect(seconds * 1.5 < referenceSeconds, "\(name)")
            }
        }

        func prompt(_ transcript: String) throws -> [Int32] {
            try tokenizer.applyChatTemplate(messages: [
                ["role": "system", "content": instructions], ["role": "user", "content": transcript],
            ]).map(Int32.init)
        }

        /// Tokens that two prompts continuing `transcript` differently share.
        func end(of transcript: String) throws -> Int {
            zip(try prompt(transcript + "Alpha"), try prompt(transcript + "Beta")).prefix { $0 == $1 }.count
        }

        func firstLogits(
            _ engine: any InferenceEngine, _ input: [Int32], declaring lengths: [Int] = []
        ) async throws -> (logits: [Float], seconds: Double) {
            engine.cachePrefixes(endingAt: lengths)
            let start = ContinuousClock.now
            let output = try await engine.generate(
                with: input, samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: 1, logits: .lastPosition))
            for try await item in output {
                return ((item.logits ?? []).map(Float.init), start.duration(to: .now) / .seconds(1))
            }
            return ([], 0)
        }

        /// Warms both engines' graphs, so that the timings after it compare prefill alone.
        func warmUp() async throws {
            _ = try await firstLogits(cached, prompt("warm up"))
            try await reference.reset()
            _ = try await firstLogits(reference, prompt("warm up"))
        }

        /// Sends `transcript` to both engines and expects the logits of the first token to agree.
        func run(_ name: String, _ transcript: String, declaring lengths: [Int] = []) async throws -> Outcome {
            let input = try prompt(transcript)
            let resumed = try await firstLogits(cached, input, declaring: lengths)
            let matched = cached.lastPrefixHitCount
            try await reference.reset()
            let cold = try await firstLogits(reference, input)
            print(
                "\(name): resumed \(resumed.seconds) s, cold \(cold.seconds) s, "
                    + "\(matched) of \(input.count) tokens matched")
            #expect(cold.logits.count > 1_000, "\(name)")
            #expect(zip(resumed.logits, cold.logits).map { abs($0 - $1) }.max() ?? .infinity < 0.05, "\(name)")
            return Outcome(seconds: resumed.seconds, referenceSeconds: cold.seconds, matchedTokens: matched)
        }
    }

    @Test("Requests that branch from a shared transcript resume from it and match cold runs")
    func branchesMatchCold() async throws {
        let rig = try await BranchRig(bundleURL: Self.bundleURL)
        let (base, mainOne, mainTwo, fork, sibling) = (
            Self.transcript("history"), Self.transcript("main one"), Self.transcript("main two"),
            Self.transcript("fork"), Self.transcript("sibling")
        )
        let baseEnd = try rig.end(of: base)
        let mainOneEnd = try rig.end(of: base + mainOne)
        let mainTwoEnd = try rig.end(of: base + mainOne + mainTwo)
        try await rig.warmUp()

        let requests: [(name: String, transcript: String, declared: [Int])] = [
            ("main", base + "Alpha", [baseEnd]),
            ("main continues", base + mainOne + "Alpha", [baseEnd, mainOneEnd]),
            ("fork", base + fork + "Alpha", [baseEnd]),
            ("main after the fork", base + mainOne + mainTwo + "Alpha", [baseEnd, mainOneEnd, mainTwoEnd]),
            ("sibling of the fork", base + sibling + "Alpha", [baseEnd]),
        ]
        for request in requests {
            let outcome = try await rig.run(request.name, request.transcript, declaring: request.declared)
            // The first request has nothing to resume from; every other one skips most of its prompt.
            if request.name != "main" { outcome.expectResumes(request.name) }
        }
    }

    @Test("A fork that leaves the main line mid-stretch gives its siblings a point to resume from")
    func junctionServesSiblings() async throws {
        let rig = try await BranchRig(bundleURL: Self.bundleURL)
        let (base, mainOne, mainTwo) = (
            Self.transcript("history"), Self.transcript("main one"), Self.transcript("main two")
        )
        // The first half of the main line's second stretch: all that a fork of it inherits.
        let inherited = Self.transcript("main two", lines: 25)
        let baseEnd = try rig.end(of: base)
        let mainOneEnd = try rig.end(of: base + mainOne)
        let mainTwoEnd = try rig.end(of: base + mainOne + mainTwo)
        try await rig.warmUp()

        _ = try await rig.run("main", base + "Alpha", declaring: [baseEnd])
        _ = try await rig.run("main continues", base + mainOne + "Alpha", declaring: [baseEnd, mainOneEnd])
        _ = try await rig.run(
            "main again", base + mainOne + mainTwo + "Alpha", declaring: [baseEnd, mainOneEnd, mainTwoEnd])

        // Neither fork knows where it leaves the main line: they declare only the Add ends they share.
        let fork = base + mainOne + inherited
        let first = try await rig.run(
            "first fork", fork + Self.transcript("fork a") + "Alpha", declaring: [baseEnd, mainOneEnd])
        let sibling = try await rig.run(
            "sibling", fork + Self.transcript("fork b") + "Alpha", declaring: [baseEnd, mainOneEnd])
        first.expectResumes("first fork")
        sibling.expectResumes("sibling")
        // The first fork took a snapshot where it left the main line; the sibling resumes from it.
        #expect(sibling.matchedTokens > first.matchedTokens)
    }

    @Test("A fork of a fork resumes from its parent, and the main line resumes after them")
    func forksOfForksInterleaveWithMain() async throws {
        let rig = try await BranchRig(bundleURL: Self.bundleURL)
        let (base, mainOne, mainTwo, forkA, forkB, forkC) = (
            Self.transcript("history"), Self.transcript("main one"), Self.transcript("main two"),
            Self.transcript("fork a"), Self.transcript("fork b"), Self.transcript("fork c")
        )
        let baseEnd = try rig.end(of: base)
        let mainOneEnd = try rig.end(of: base + mainOne)
        let mainTwoEnd = try rig.end(of: base + mainOne + mainTwo)
        let forkAEnd = try rig.end(of: base + mainOne + forkA)
        let forkBEnd = try rig.end(of: base + mainOne + forkA + forkB)
        try await rig.warmUp()

        let shared = [baseEnd, mainOneEnd]
        let requests: [(name: String, transcript: String, declared: [Int])] = [
            ("main", base + "Alpha", [baseEnd]),
            ("main continues", base + mainOne + "Alpha", shared),
            ("fork", base + mainOne + forkA + "Alpha", shared + [forkAEnd]),
            ("fork of the fork", base + mainOne + forkA + forkB + "Alpha", shared + [forkAEnd, forkBEnd]),
            ("main after the forks", base + mainOne + mainTwo + "Alpha", shared + [mainTwoEnd]),
            ("sibling of the fork of the fork", base + mainOne + forkA + forkC + "Alpha", shared + [forkAEnd]),
            ("fork of the fork again", base + mainOne + forkA + forkB + "Beta", shared + [forkAEnd, forkBEnd]),
        ]
        for request in requests {
            let outcome = try await rig.run(request.name, request.transcript, declaring: request.declared)
            if request.name != "main" { outcome.expectResumes(request.name) }
        }
    }

    @Test("A restore that needs a larger context bucket than the live caches hold grows it")
    func restoreGrowsTheContextBucket() async throws {
        let rig = try await BranchRig(bundleURL: Self.bundleURL, instructions: "Answer the question about the log.")
        let history = Self.transcript("history", lines: 200)
        let historyEnd = try rig.end(of: history)
        try await rig.warmUp()

        _ = try await rig.run("long", history + "Alpha", declaring: [historyEnd])
        // A prompt of a few dozen tokens shrinks the live caches to the smallest context bucket.
        _ = try await rig.run("short", "Is item 14 risky?")
        let grown = try await rig.run("long again", history + "Beta", declaring: [historyEnd])
        // The resume point lies past the smallest bucket, which holds 1024 tokens.
        #expect(grown.matchedTokens > 1_024)
        grown.expectResumes("long again")
    }

    @Test("Under a budget that holds almost nothing, requests still match cold runs and the first snapshot stays")
    func evictionKeepsThePinnedSnapshot() async throws {
        let rig = try await BranchRig(bundleURL: Self.bundleURL, budgetBytes: 4_000_000)
        let (base, mainOne, mainTwo, fork, sibling) = (
            Self.transcript("history"), Self.transcript("main one"), Self.transcript("main two"),
            Self.transcript("fork"), Self.transcript("sibling")
        )
        let baseEnd = try rig.end(of: base)
        let mainOneEnd = try rig.end(of: base + mainOne)
        let mainTwoEnd = try rig.end(of: base + mainOne + mainTwo)
        try await rig.warmUp()

        let requests: [(name: String, transcript: String, declared: [Int])] = [
            ("main", base + "Alpha", [baseEnd]),
            ("main continues", base + mainOne + "Alpha", [baseEnd, mainOneEnd]),
            ("fork", base + fork + "Alpha", [baseEnd]),
            ("main after the fork", base + mainOne + mainTwo + "Alpha", [baseEnd, mainOneEnd, mainTwoEnd]),
            ("sibling of the fork", base + sibling + "Alpha", [baseEnd]),
        ]
        for request in requests {
            _ = try await rig.run(request.name, request.transcript, declaring: request.declared)
        }

        // The Add end the main line left behind was evicted, so a request over it resumes from the
        // first snapshot instead.
        let evicted = try await rig.run("over an evicted snapshot", base + mainOne + "Gamma", declaring: [baseEnd, mainOneEnd])
        #expect(evicted.matchedTokens > baseEnd / 2 && evicted.matchedTokens <= baseEnd)
        let pinned = try await rig.run("over the pinned snapshot", base + Self.transcript("other") + "Alpha", declaring: [baseEnd])
        #expect(pinned.matchedTokens > baseEnd / 2)
        pinned.expectResumes("over the pinned snapshot")
    }

    @Test("A request resumes from the first segment of its prompt when only the last one changes")
    func segmentedPromptResumes() async throws {
        let model = try await CoreAILanguageModel(resourcesAt: Self.bundleURL)
        // Short instructions hold no block, so only the end of the first segment can be resumed from.
        func ask(_ item: Int) async throws -> (Verdict, seconds: Double) {
            let session = LanguageModelSession(model: model, instructions: "Answer the question about the log.")
            let start = ContinuousClock.now
            let response = try await session.respond(
                to: Prompt {
                    Self.transcript("history", lines: 200)
                    "Is item \(item) risky?"
                },
                generating: Verdict.self,
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
}

#endif
