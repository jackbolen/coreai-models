// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CryptoKit
import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("PrefixCache")
struct PrefixCacheTests {
    private static let blockSize = 2
    private static let blockBytes = 100
    private static let snapshotBytes = 1000

    /// A payload of `bytes` bytes.
    private func payload(_ bytes: Int) -> [String: NDArray] {
        ["state": NDArray(shape: [bytes / 2], scalarType: .float16)]
    }

    /// The tokens of blocks named by id: block `n` is `[n, n]`.
    private func tokens(_ ids: [Int], tail: Bool = false) -> [Int32] {
        ids.flatMap { [Int32($0), Int32($0)] } + (tail ? [99] : [])
    }

    private func makeCache(
        budget: Int = .max,
        keyOf: @escaping (PrefixCache.Key, ArraySlice<Int32>) -> PrefixCache.Key = PrefixCache.chainKey
    ) -> PrefixCache {
        PrefixCache(blockSize: Self.blockSize, budgetBytes: budget, keyOf: keyOf)
    }

    /// Records the blocks `ids` as one path, from `parent` (the root when nil), taking a snapshot
    /// after the blocks whose 1-based position along the path is in `snapshotAfter`.
    @discardableResult
    private func record(
        _ cache: PrefixCache, _ ids: [Int], under parent: PrefixCache.Block? = nil,
        snapshotAfter: [Int: Bool] = [:]
    ) -> [PrefixCache.Block] {
        var blocks: [PrefixCache.Block] = []
        var parent = parent
        for (index, id) in ids.enumerated() {
            let block = cache.record(
                block: tokens([id])[...], under: parent, kv: payload(Self.blockBytes))!
            if let pinned = snapshotAfter[index + 1] {
                cache.attachSnapshot(at: block, ring: payload(Self.snapshotBytes), pinned: pinned)
            }
            blocks.append(block)
            parent = block
        }
        return blocks
    }

    @Test("A prompt matches the blocks it begins with and nothing past the first difference")
    func chainLookup() {
        let cache = makeCache()
        record(cache, [1, 2, 3])

        #expect(cache.lookup(tokens([1, 2, 3], tail: true)).blocks.count == 3)
        #expect(cache.lookup(tokens([1, 2, 3], tail: true)).matchedTokens == 6)
        #expect(cache.lookup(tokens([1, 2, 9], tail: true)).blocks.count == 2)
        #expect(cache.lookup(tokens([1, 9, 3], tail: true)).blocks.count == 1)
        #expect(cache.lookup(tokens([9, 2, 3], tail: true)).blocks.isEmpty)
        #expect(cache.lookup([]).blocks.isEmpty)
    }

    @Test("The last token of a prompt is never matched")
    func lastTokenIsNeverMatched() {
        let cache = makeCache()
        record(cache, [1, 2, 3])

        #expect(cache.lookup(tokens([1, 2, 3])).blocks.count == 2)
        #expect(cache.lookup(Array(tokens([1, 2, 3]).dropLast())).blocks.count == 2)
        #expect(cache.lookup(Array(tokens([1, 2, 3]).dropLast(2))).blocks.count == 1)
        #expect(cache.lookup(Array(tokens([1, 2, 3]).dropLast(3))).blocks.count == 1)
    }

    @Test("Two blocks with one key are told apart by their tokens")
    func keyCollisionNeverReusesWrongState() {
        let key = SHA256.hash(data: [7])
        let cache = makeCache(keyOf: { _, _ in key })
        let first = cache.record(block: tokens([1])[...], under: nil, kv: payload(Self.blockBytes))

        #expect(first != nil)
        #expect(cache.record(block: tokens([2])[...], under: nil, kv: payload(Self.blockBytes)) == nil)
        #expect(cache.record(block: tokens([1])[...], under: nil, kv: payload(Self.blockBytes)) === first)
        #expect(cache.lookup(tokens([1], tail: true)).blocks.count == 1)
        #expect(cache.lookup(tokens([2], tail: true)).blocks.isEmpty)
        #expect(cache.blockCount == 1)
    }

    @Test("A key depends on the whole path, so equal blocks under different parents are different blocks")
    func keysChain() {
        let cache = makeCache()
        let left = record(cache, [1, 5])
        let right = record(cache, [2, 5])

        #expect(left[1] !== right[1])
        #expect(left[1].key != right[1].key)
        #expect(cache.blockCount == 4)
    }

    @Test("The resume point is the deepest snapshot on the path")
    func resumesFromDeepestSnapshot() {
        let cache = makeCache()
        record(cache, [1, 2, 3, 4, 5], snapshotAfter: [2: false, 4: false])

        let full = cache.lookup(tokens([1, 2, 3, 4, 5], tail: true))
        #expect(full.resumeBlockCount == 4)
        #expect(full.resumeTokens == 8)
        #expect(full.resume === full.blocks[3])
        #expect(cache.lookup(tokens([1, 2, 3], tail: true)).resumeBlockCount == 2)
        #expect(cache.lookup(tokens([1], tail: true)).resume == nil)
    }

    @Test("A snapshot is kept once written")
    func snapshotsAreImmutable() {
        let cache = makeCache()
        let blocks = record(cache, [1], snapshotAfter: [1: false])

        #expect(!cache.attachSnapshot(at: blocks[0], ring: payload(Self.snapshotBytes), pinned: true))
        #expect(blocks[0].snapshot?.pinned == false)
        #expect(cache.snapshotCount == 1)
    }

    @Test("A junction needs a snapshot below on another branch and enough gain")
    func junctionRule() {
        let cache = makeCache()
        record(cache, [1, 2, 3, 4, 5, 6])
        let leavesAtFive = tokens([1, 2, 3, 4, 5, 9], tail: true)
        #expect(cache.lookup(leavesAtFive).junctionBlockCount == nil)

        cache.attachSnapshot(
            at: cache.lookup(tokens([1, 2, 3, 4, 5, 6], tail: true)).blocks[5],
            ring: payload(Self.snapshotBytes), pinned: false)
        let junction = cache.lookup(leavesAtFive)
        #expect(junction.junctionBlockCount == 5)
        #expect(junction.junctionTokens == 10)
        #expect(junction.resume == nil)

        // The snapshot is below the path's end, so a prompt that follows it to the end has none to
        // plan.
        #expect(cache.lookup(tokens([1, 2, 3, 4, 5, 6], tail: true)).junctionBlockCount == nil)
        // A prompt that leaves at four blocks has the snapshot below on another branch too.
        #expect(cache.lookup(tokens([1, 2, 3, 4, 9], tail: true)).junctionBlockCount == 4)
        // Three blocks is not enough gain.
        #expect(cache.lookup(tokens([1, 2, 3, 9], tail: true)).junctionBlockCount == nil)
    }

    @Test("A snapshot on the path near the match end leaves too little gain for a junction")
    func nearbySnapshotSuppressesJunction() {
        let cache = makeCache()
        record(cache, [1, 2, 3, 4, 5, 6], snapshotAfter: [3: false, 6: false])

        let match = cache.lookup(tokens([1, 2, 3, 4, 5, 9], tail: true))
        #expect(match.resumeBlockCount == 3)
        #expect(match.junctionBlockCount == nil)
    }

    @Test("A fork of a fork resumes from its own branch point once snapshots exist there")
    func forkOfAFork() {
        let cache = makeCache()
        record(cache, [1, 2, 3, 4, 5, 6, 7, 8], snapshotAfter: [8: false])

        // The first fork leaves after six blocks: a junction there.
        let fork = tokens([1, 2, 3, 4, 5, 6, 20, 21], tail: true)
        let first = cache.lookup(fork)
        #expect(first.junctionBlockCount == 6)
        let shared = first.blocks
        cache.attachSnapshot(at: shared[5], ring: payload(Self.snapshotBytes), pinned: false)
        let branch = record(cache, [20, 21], under: shared[5], snapshotAfter: [2: false])

        // A second fork of the first resumes at the junction, with nothing more to plan.
        let sibling = cache.lookup(tokens([1, 2, 3, 4, 5, 6, 30, 31], tail: true))
        #expect(sibling.resumeBlockCount == 6)
        #expect(sibling.junctionBlockCount == nil)

        // A fork of the fork leaves it after one block of its own.
        let grandchild = cache.lookup(tokens([1, 2, 3, 4, 5, 6, 20, 40, 41], tail: true))
        #expect(grandchild.blocks.count == 7)
        #expect(grandchild.resumeBlockCount == 6)
        #expect(grandchild.junctionBlockCount == nil)

        // Resuming at the end of the first fork's own branch.
        let continued = cache.lookup(tokens([1, 2, 3, 4, 5, 6, 20, 21, 22], tail: true))
        #expect(continued.resume === branch[1])
        #expect(continued.resumeBlockCount == 8)
    }

    @Test("Blocks no snapshot sits at or below go first, least recently used first")
    func evictsUnrestorableBlocksFirst() {
        // Budget for four blocks.
        let cache = makeCache(budget: 4 * Self.blockBytes)
        let chain = record(cache, [1, 2, 3])
        _ = cache.lookup(tokens([1, 2, 3], tail: true))
        record(cache, [4])
        record(cache, [5])
        #expect(cache.byteCount == 5 * Self.blockBytes)

        cache.evict()

        // The leaf of the chain is the least recently used leaf; its parents are not leaves.
        #expect(cache.byteCount == 4 * Self.blockBytes)
        #expect(!chain[2].isAttached)
        #expect(chain[1].isAttached)
        #expect(cache.lookup(tokens([4], tail: true)).blocks.count == 1)
        #expect(cache.lookup(tokens([1, 2, 3], tail: true)).blocks.count == 2)
    }

    @Test("A leaf with a snapshot, and the blocks above it, stay while others go")
    func snapshotsKeepTheirBlocks() {
        let cache = makeCache(budget: 3 * Self.blockBytes + Self.snapshotBytes)
        record(cache, [1, 2, 3], snapshotAfter: [3: false])
        record(cache, [4, 5])

        cache.evict()

        #expect(cache.lookup(tokens([1, 2, 3], tail: true)).resumeBlockCount == 3)
        #expect(cache.lookup(tokens([4, 5], tail: true)).blocks.isEmpty)
    }

    @Test("Snapshots go oldest first, after the blocks they leave unrestorable")
    func evictsSnapshotsOldestFirst() {
        let cache = makeCache(budget: 2300)
        let pinned = record(cache, [1], snapshotAfter: [1: true])
        let older = record(cache, [2], snapshotAfter: [1: false])
        let newer = record(cache, [3], snapshotAfter: [1: false])
        #expect(cache.byteCount == 3300)

        cache.evict()

        #expect(pinned[0].snapshot != nil)
        #expect(older[0].snapshot == nil)
        #expect(newer[0].snapshot != nil)
        // Its block stays: the budget holds without it.
        #expect(cache.lookup(tokens([2], tail: true)).blocks.count == 1)
        #expect(cache.byteCount == 2300)
    }

    @Test("A pinned snapshot and its blocks stay however far over budget the cache is")
    func pinnedSnapshotIsNeverEvicted() {
        let cache = makeCache(budget: 500)
        let pinned = record(cache, [1, 2], snapshotAfter: [2: true])
        record(cache, [3], snapshotAfter: [1: false])
        record(cache, [4, 5])

        cache.evict()

        #expect(pinned[1].snapshot != nil)
        let match = cache.lookup(tokens([1, 2], tail: true))
        #expect(match.resumeBlockCount == 2)
        #expect(cache.snapshotCount == 1)
        #expect(cache.blockCount == 2)
    }

    @Test("After a snapshot is evicted a lookup resumes from a shallower one")
    func shallowerResumeAfterEviction() {
        let cache = makeCache(budget: 1500)
        let chain = record(cache, [1, 2, 3, 4], snapshotAfter: [4: false])
        cache.attachSnapshot(at: chain[1], ring: payload(Self.snapshotBytes), pinned: false)

        cache.evict()

        let match = cache.lookup(tokens([1, 2, 3, 4], tail: true))
        #expect(match.blocks.count == 4)
        #expect(match.resumeBlockCount == 2)
    }

    @Test("Recording under an evicted block adds nothing")
    func recordUnderEvictedParent() {
        let cache = makeCache(budget: 0)
        let chain = record(cache, [1])

        cache.evict()

        #expect(cache.blockCount == 0)
        #expect(cache.record(block: tokens([2])[...], under: chain[0], kv: payload(Self.blockBytes)) == nil)
        #expect(!cache.attachSnapshot(at: chain[0], ring: payload(Self.snapshotBytes), pinned: false))
    }
}
