// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CryptoKit
import Foundation

/// Reusable prompt state for an engine whose sliding-window ring cannot be rewound far.
///
/// A prompt is cut into blocks of ``blockSize`` tokens. A block is keyed by a SHA-256 chain over
/// its parent's key and its tokens, and holds the flat KV positions those tokens produced, so every
/// prompt that begins with the same tokens finds the same blocks whichever request computed them.
/// KV truncates anywhere; the ring does not, so it is saved whole as a snapshot at the blocks worth
/// resuming from. A position is restorable only where a snapshot sits.
///
/// Keys, tokens and payloads never change once written, and none refers to a process, a session or
/// an object, so a block or snapshot can be kept elsewhere and read back unchanged. The cache is
/// not thread-safe: the engine that owns it runs one request at a time.
final class PrefixCache {
    typealias Key = SHA256Digest

    /// Blocks a request must be past its resume point, with a snapshot on another branch below,
    /// before a junction is worth the ring it costs.
    static let minimumJunctionGain = 4

    /// A ring saved at the end of a block.
    struct Snapshot {
        let ring: [String: NDArray]
        /// A pinned snapshot is never evicted.
        let pinned: Bool
        fileprivate var byteCount: Int { payloadByteCount(ring) }
    }

    /// The tokens of one block and the KV they produced.
    final class Block {
        let key: Key
        let tokens: [Int32]
        /// Blocks from the root; the root is 0.
        let depth: Int
        let kv: [String: NDArray]
        fileprivate(set) var snapshot: Snapshot?

        fileprivate weak var parent: Block?
        fileprivate var children: [Key: Block] = [:]
        /// Snapshots strictly below this block.
        fileprivate var snapshotsBelow = 0
        fileprivate var lastUsed: UInt64 = 0
        fileprivate var snapshotLastUsed: UInt64 = 0
        /// False once evicted: the block no longer takes children or a snapshot.
        fileprivate(set) var isAttached = true

        fileprivate init(key: Key, tokens: [Int32], depth: Int, kv: [String: NDArray], parent: Block?) {
            self.key = key
            self.tokens = tokens
            self.depth = depth
            self.kv = kv
            self.parent = parent
        }
    }

    /// What a prompt shares with the cache.
    struct Match {
        let blockSize: Int
        /// The matched blocks, root first. The last token of the prompt is never matched, so the
        /// engine always has one real token to run for its logits.
        let blocks: [Block]
        /// Blocks through the deepest snapshot; 0 without one.
        let resumeBlockCount: Int
        /// Where to take a snapshot while prefilling: the end of the matched blocks, when the
        /// prompt leaves a path that another branch continues and that branch holds a snapshot.
        let junctionBlockCount: Int?

        var matchedTokens: Int { blocks.count * blockSize }
        var resumeTokens: Int { resumeBlockCount * blockSize }
        var junctionTokens: Int? { junctionBlockCount.map { $0 * blockSize } }
        /// The block whose snapshot the prompt resumes from.
        var resume: Block? { resumeBlockCount > 0 ? blocks[resumeBlockCount - 1] : nil }
    }

    let blockSize: Int
    let budgetBytes: Int
    private(set) var byteCount = 0
    var blockCount: Int { attached.count }
    var snapshotCount: Int { attached.values.reduce(0) { $0 + ($1.snapshot == nil ? 0 : 1) } }

    private let keyOf: (Key, ArraySlice<Int32>) -> Key
    private let root: Block
    private var attached: [ObjectIdentifier: Block] = [:]
    private var clock: UInt64 = 0

    /// - Parameter keyOf: The key of a block from its parent's key and its tokens.
    init(blockSize: Int, budgetBytes: Int, keyOf: @escaping (Key, ArraySlice<Int32>) -> Key = PrefixCache.chainKey) {
        precondition(blockSize > 0, "A block holds at least one token")
        self.blockSize = blockSize
        self.budgetBytes = budgetBytes
        self.keyOf = keyOf
        root = Block(key: SHA256.hash(data: [UInt8]()), tokens: [], depth: 0, kv: [:], parent: nil)
    }

    /// Finds the blocks `tokens` begins with, and the snapshot to resume from.
    func lookup(_ tokens: [Int32]) -> Match {
        clock += 1
        var chain: [Block] = []
        var parent = root
        for index in 0..<(Swift.max(0, tokens.count - 1) / blockSize) {
            let span = tokens[index * blockSize..<(index + 1) * blockSize]
            guard let child = parent.children[keyOf(parent.key, span)], child.tokens.elementsEqual(span)
            else { break }
            child.lastUsed = clock
            chain.append(child)
            parent = child
        }
        let resumeCount = (chain.lastIndex { $0.snapshot != nil } ?? -1) + 1
        if resumeCount > 0 { chain[resumeCount - 1].snapshotLastUsed = clock }
        var junction: Int?
        if let last = chain.last, chain.count - resumeCount >= Self.minimumJunctionGain, last.snapshotsBelow > 0 {
            junction = chain.count
        }
        return Match(blockSize: blockSize, blocks: chain, resumeBlockCount: resumeCount, junctionBlockCount: junction)
    }

    /// Adds the block of `tokens` under `parent` (the root when nil) and returns it, or returns the
    /// block already there. Nil when `parent` was evicted, or when another block holds the same key.
    @discardableResult
    func record(block tokens: ArraySlice<Int32>, under parent: Block?, kv: [String: NDArray]) -> Block? {
        precondition(tokens.count == blockSize, "A block holds exactly \(blockSize) tokens")
        let parent = parent ?? root
        guard parent.isAttached else { return nil }
        let key = keyOf(parent.key, tokens)
        if let existing = parent.children[key] {
            return existing.tokens.elementsEqual(tokens) ? existing : nil
        }
        clock += 1
        let block = Block(key: key, tokens: Array(tokens), depth: parent.depth + 1, kv: kv, parent: parent)
        block.lastUsed = clock
        parent.children[key] = block
        attached[ObjectIdentifier(block)] = block
        byteCount += payloadByteCount(kv)
        return block
    }

    /// Saves `ring` as the state after `block` and everything above it. False when the block was
    /// evicted or already holds a snapshot, which stays as it is.
    @discardableResult
    func attachSnapshot(at block: Block, ring: [String: NDArray], pinned: Bool) -> Bool {
        guard block.isAttached, block !== root, block.snapshot == nil else { return false }
        clock += 1
        let snapshot = Snapshot(ring: ring, pinned: pinned)
        block.snapshot = snapshot
        block.snapshotLastUsed = clock
        byteCount += snapshot.byteCount
        adjustSnapshotsBelow(of: block, by: 1)
        return true
    }

    /// Brings the cache within its budget. Blocks that no snapshot sits at or below go first, least
    /// recently used and deepest first; then unpinned snapshots, least recently used first, which
    /// leaves their blocks to go the same way. Whatever is evicted, every other block and snapshot
    /// is unchanged, so a later lookup only finds less. Pinned snapshots and the blocks above them
    /// stay even when they exceed the budget.
    func evict() {
        while byteCount > budgetBytes {
            let leaf = attached.values
                .filter { $0.children.isEmpty && $0.snapshot == nil }
                .min { ($0.lastUsed, -$0.depth) < ($1.lastUsed, -$1.depth) }
            if let leaf {
                detach(leaf)
                continue
            }
            let holder = attached.values
                .filter { $0.snapshot.map { !$0.pinned } ?? false }
                .min { $0.snapshotLastUsed < $1.snapshotLastUsed }
            guard let holder, let snapshot = holder.snapshot else { return }
            holder.snapshot = nil
            byteCount -= snapshot.byteCount
            adjustSnapshotsBelow(of: holder, by: -1)
        }
    }

    // MARK: - Internals

    private func detach(_ block: Block) {
        block.parent?.children[block.key] = nil
        block.parent = nil
        block.isAttached = false
        attached[ObjectIdentifier(block)] = nil
        byteCount -= payloadByteCount(block.kv)
    }

    private func adjustSnapshotsBelow(of block: Block, by delta: Int) {
        var ancestor = block.parent
        while let current = ancestor {
            current.snapshotsBelow += delta
            ancestor = current.parent
        }
    }

    /// SHA-256 of the parent's key followed by the tokens as 32-bit integers.
    static func chainKey(_ parent: Key, _ tokens: ArraySlice<Int32>) -> Key {
        var hasher = SHA256()
        parent.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        tokens.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return hasher.finalize()
    }
}

private func payloadByteCount(_ arrays: [String: NDArray]) -> Int {
    arrays.values.reduce(0) { $0 + estimatedArrayBytes($1) }
}
