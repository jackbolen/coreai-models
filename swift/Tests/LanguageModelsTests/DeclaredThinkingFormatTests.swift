// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import TestUtilities
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Declared thinking format")
struct DeclaredThinkingFormatTests {
    /// A tokenizer directory holding `config` as its `tokenizer_config.json`, or none when nil.
    private func tokenizerDirectory(config: String?) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let config {
            try config.write(
                to: directory.appending(path: "tokenizer_config.json"), atomically: true, encoding: .utf8)
        }
        return directory
    }

    @Test("Reads the markers the tokenizer config declares")
    func readsDeclaredMarkers() throws {
        let directory = try tokenizerDirectory(
            config: #"{"response_template":{"fields":{"thinking":{"open":"<|channel>thought\n","close":"<channel|>"}}}}"#
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        guard case .tagPair(let open, let close)? = declaredThinkingFormat(inTokenizerDirectory: directory)
        else {
            Issue.record("expected a tag pair")
            return
        }
        #expect(open == "<|channel>thought\n")
        #expect(close == "<channel|>")
    }

    @Test("Declares nothing without a thinking field, with an empty marker, or without a config")
    func declaresNothing() throws {
        for config in [
            nil, "{}", #"{"response_template":{"fields":{}}}"#,
            #"{"response_template":{"fields":{"thinking":{"open":"","close":"</think>"}}}}"#,
        ] {
            let directory = try tokenizerDirectory(config: config)
            defer { try? FileManager.default.removeItem(at: directory) }
            #expect(declaredThinkingFormat(inTokenizerDirectory: directory) == nil)
        }
    }

    // Gemma 4 was trained with the thinking of earlier turns stripped.
    @Test("A thought in the transcript is left out of the rendered prompt")
    func renderingOmitsThoughts() {
        let tokenizer = CapturingTokenizer()
        func text(_ content: String) -> [Transcript.Segment] { [.text(Transcript.TextSegment(content: content))] }
        _ = CoreAILanguageModel.CoreAIExecutor.makeTokens(
            from: [
                .prompt(Transcript.Prompt(segments: text("question"))),
                .reasoning(Transcript.Reasoning(segments: text("a private thought"))),
                .response(Transcript.Response(assetIDs: [], segments: text("answer"))),
            ],
            using: tokenizer)
        let rendered = tokenizer.capturedMessages.compactMap { $0["content"] as? String }
        #expect(rendered == ["question", "answer"])
    }
}

#endif
