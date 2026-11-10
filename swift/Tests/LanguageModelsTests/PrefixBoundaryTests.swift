// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Prefix boundaries")
struct PrefixBoundaryTests {
    private typealias Executor = CoreAILanguageModel.CoreAIExecutor

    private static func text(_ pieces: [String]) -> [Transcript.Segment] {
        pieces.map { .text(Transcript.TextSegment(content: $0)) }
    }

    private static func instructions(_ content: String) -> Transcript.Entry {
        .instructions(Transcript.Instructions(segments: text([content]), toolDefinitions: []))
    }

    private static func prompt(_ pieces: String...) -> Transcript.Entry {
        .prompt(Transcript.Prompt(segments: text(pieces)))
    }

    /// One token per byte of each entry's text, the way a template lays out turns, ending in the
    /// marker that closes the last one. `merge` joins the two bytes it names into a single token
    /// wherever they meet, as a vocabulary entry for the pair would.
    private static func render(merging merge: (UInt8, UInt8)? = nil) -> ([Transcript.Entry]) -> [Int] {
        { entries in
            var bytes = Array(
                entries.map { entry -> String in
                    switch entry {
                    case .instructions(let value): value.segments.map(\.text).joined()
                    case .prompt(let value): value.segments.map(\.text).joined()
                    default: ""
                    }
                }.joined(separator: "|").utf8)
            bytes.append(UInt8(ascii: "#"))
            var tokens: [Int] = []
            var index = 0
            while index < bytes.count {
                if let merge, index + 1 < bytes.count, bytes[index] == merge.0, bytes[index + 1] == merge.1 {
                    tokens.append(1000)
                    index += 2
                } else {
                    tokens.append(Int(bytes[index]))
                    index += 1
                }
            }
            return tokens
        }
    }

    private static func boundaries(
        _ transcript: [Transcript.Entry], merging merge: (UInt8, UInt8)? = nil
    ) -> [Int] {
        let render = render(merging: merge)
        return Executor.prefixBoundaries(of: render(transcript), transcript: transcript, render: render)
    }

    @Test("Instructions alone end at the instructions")
    func instructionsOnly() {
        #expect(Self.boundaries([Self.instructions("rules"), Self.prompt("question")]) == [5])
    }

    @Test("No instructions and one segment give no boundary")
    func nothingToCut() {
        #expect(Self.boundaries([Self.prompt("question")]).isEmpty)
        #expect(Self.boundaries([]).isEmpty)
    }

    @Test("Each segment but the last ends a boundary after the instructions")
    func segmentsAddBoundaries() {
        let transcript = [Self.instructions("rules"), Self.prompt("alpha", "beta", "gamma")]
        // "rules|" is 6 tokens; "alpha" 5 more, "beta" 4 more.
        #expect(Self.boundaries(transcript) == [5, 11, 15])
    }

    @Test("A cut that tokenizes differently shortens its boundary")
    func mergedTokensShortenBoundary() {
        // The first segment ends in "a" and the second starts with "b". Where the whole prompt merges
        // them into one token, the cut's tokens agree with it only up to the "a".
        let transcript = [Self.instructions("rules"), Self.prompt("xa", "by")]
        #expect(Self.boundaries(transcript) == [5, 8])
        #expect(Self.boundaries(transcript, merging: (UInt8(ascii: "a"), UInt8(ascii: "b"))) == [5, 7])
    }

    @Test("Earlier turns lead every cut of the last prompt")
    func earlierTurnsLead() {
        let transcript = [
            Self.instructions("rules"), Self.prompt("one"), Self.prompt("alpha", "beta"),
        ]
        // "rules|one|" is 10 tokens, then "alpha".
        #expect(Self.boundaries(transcript) == [5, 15])
    }
}

extension Transcript.Segment {
    fileprivate var text: String {
        if case .text(let value) = self { return value.content }
        return ""
    }
}

#endif
