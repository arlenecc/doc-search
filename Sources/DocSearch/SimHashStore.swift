import Foundation

/// 全局文本相似排重：从 tools/SimHashDedupe.swift 移植 shingles 模式的
/// MurmurHash3 + SimHash(64) + 4 表 LSH + 内容复核，索引改为 SQLite 持久化。
///
/// 用途：Markdown 生成完毕、切文本入库之前，对全文算指纹并查已入库文档索引；
/// 海明距离 ≤ ham 且内容复核通过 → 判重复，直接跳过切片入库。
enum SimHashStore {
    /// 与 tools/SimHashDedupe.swift Options 默认值一致（仅 shingles 模式）。
    struct Options {
        var sampleChars = 20000
        var top3 = 300
        var top2 = 200
        var ham = 3
        var verify = true
        var verifyJaccard = 0.5
        var verifySampleChars = 20000
        var hashSeed: UInt64 = 0
        var useStopwords = true
    }

    struct Match {
        let path: String
        let hamming: Int
    }

    // MARK: - MurmurHash3 x64_128 (h1)

    @inline(__always) private static func rotl(_ x: UInt64, _ r: Int) -> UInt64 {
        (x << r) | (x >> (64 - r))
    }

    @inline(__always) private static func fmix64(_ k: UInt64) -> UInt64 {
        var k = k
        k ^= k >> 33
        k = k &* 0xff51afd7ed558ccd
        k ^= k >> 33
        k = k &* 0xc4ceb9fe1a85ec53
        k ^= k >> 33
        return k
    }

    static func hash64(_ data: [UInt8], seed: UInt64 = 0) -> UInt64 {
        let c1: UInt64 = 0x87c37b91114253d5
        let c2: UInt64 = 0x4cf5ad432745937f
        var h1 = seed, h2 = seed
        let blocks = data.count / 16
        for b in 0..<blocks {
            var k1: UInt64 = 0, k2: UInt64 = 0
            for j in 0..<8 { k1 |= UInt64(data[b * 16 + j]) << (8 * j) }
            for j in 0..<8 { k2 |= UInt64(data[b * 16 + 8 + j]) << (8 * j) }
            k1 = k1 &* c1; k1 = rotl(k1, 31); k1 = k1 &* c2; h1 ^= k1
            h1 = rotl(h1, 27); h1 = h1 &+ h2; h1 = h1 &* 5 &+ 0x52dce729
            k2 = k2 &* c2; k2 = rotl(k2, 33); k2 = k2 &* c1; h2 ^= k2
            h2 = rotl(h2, 31); h2 = h2 &+ h1; h2 = h2 &* 5 &+ 0x38495ab5
        }
        let off = blocks * 16
        let tail = data.count - off
        if tail > 0 {
            var k1: UInt64 = 0, k2: UInt64 = 0
            if tail >= 9 {
                for j in 8..<tail { k2 ^= UInt64(data[off + j]) << (8 * (j - 8)) }
                k2 = k2 &* c2; k2 = rotl(k2, 33); k2 = k2 &* c1; h2 ^= k2
            }
            if tail >= 1 {
                for j in 0..<min(tail, 8) { k1 ^= UInt64(data[off + j]) << (8 * j) }
                k1 = k1 &* c1; k1 = rotl(k1, 31); k1 = k1 &* c2; h1 ^= k1
            }
        }
        h1 ^= UInt64(data.count); h2 ^= UInt64(data.count)
        h1 = h1 &+ h2; h2 = h2 &+ h1
        h1 = fmix64(h1); h2 = fmix64(h2)
        h1 = h1 &+ h2; h2 = h2 &+ h1
        return h1
    }

    static func hash64(_ s: String, seed: UInt64 = 0) -> UInt64 {
        hash64(Array(s.utf8), seed: seed)
    }

    // MARK: - 文本清理 / 特征

    @inline(__always) private static func kept(_ s: Unicode.Scalar) -> Character? {
        let v = s.value
        if (v >= 0x4E00 && v <= 0x9FFF) || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF) || (v >= 0xF900 && v <= 0xFAFF) {
            return Character(s)
        } else if v >= 0x41 && v <= 0x5A {
            return UnicodeScalar(v + 32).map { Character($0) }
        } else if v >= 0x61 && v <= 0x7A {
            return Character(s)
        }
        return nil
    }

    private static func cleanAll(_ text: String) -> [Character] {
        var out: [Character] = []
        for s in text.unicodeScalars {
            if let c = kept(s) { out.append(c) }
        }
        return out
    }

    private static func segments(_ text: String, sampleChars: Int) -> [[Character]] {
        let clean = cleanAll(text)
        if sampleChars <= 0 || clean.count <= 2 * sampleChars { return [clean] }
        return [Array(clean.prefix(sampleChars)), Array(clean.suffix(sampleChars))]
    }

    private static let weakBoundaryChars: Set<Character> = Set(
        "的 了 着 过 得 地 是 在 和 与 或 也 就 都 而 之 以 于 其 这 那 一 不 很 更 最 可 能 会 要 将 被 把 从 对 为 所 由 并 且 但 则 者 矣 焉 乎 哉 又 再 还 才 只 已 曾 乃 若 即 便 却 竟 偏"
            .filter { !$0.isWhitespace })

    private static let stopwords: Set<String> = {
        var set: Set<String> = []
        if let url = Bundle.module.url(forResource: "stopwords_full", withExtension: "txt"),
           let text = try? String(contentsOf: url, encoding: .utf8) {
            for line in text.split(whereSeparator: \.isNewline) {
                let w = String(line).trimmingCharacters(in: .whitespaces).lowercased()
                if !w.isEmpty { set.insert(w) }
            }
        }
        return set
    }()

    private static func grams(_ segs: [[Character]], k: Int, useStopwords: Bool) -> [String: Int] {
        var counts: [String: Int] = [:]
        for seg in segs {
            guard seg.count >= k else { continue }
            for i in 0...(seg.count - k) {
                let g = String(seg[i..<(i + k)])
                if useStopwords {
                    if stopwords.contains(g) { continue }
                    if g.allSatisfy({ weakBoundaryChars.contains($0) }) { continue }
                }
                counts[g, default: 0] += 1
            }
        }
        return counts
    }

    private static func top(_ counts: [String: Int], _ n: Int) -> [(String, Int)] {
        guard n > 0 else { return [] }
        return counts.sorted {
            if $0.value != $1.value { return $0.value > $1.value }
            return $0.key < $1.key
        }.prefix(n).map { ($0.key, $0.value) }
    }

    /// 代表文本 → 双窗口 k=3/k=2 特征（权重=TF）。
    static func features(_ text: String, _ opts: Options) -> [(String, Int)] {
        let segs = segments(text, sampleChars: opts.sampleChars)
        return top(grams(segs, k: 3, useStopwords: opts.useStopwords), opts.top3)
            + top(grams(segs, k: 2, useStopwords: opts.useStopwords), opts.top2)
    }

    /// SimHash 指纹：64 维逐位 +w/−w 累加，>0 记 1。
    static func fingerprint(_ text: String, _ opts: Options = Options()) -> UInt64 {
        var v = [Int64](repeating: 0, count: 64)
        for (word, weight) in features(text, opts) {
            let h = hash64(word, seed: opts.hashSeed)
            let w = Int64(weight)
            for i in 0..<64 { v[i] += ((h >> i) & 1 == 1) ? w : -w }
        }
        var fp: UInt64 = 0
        for i in 0..<64 where v[i] > 0 { fp |= (UInt64(1) << i) }
        return fp
    }

    @inline(__always) private static func band(_ fp: UInt64, _ b: Int) -> UInt16 {
        UInt16(truncatingIfNeeded: fp >> (16 * b))
    }

    @inline(__always) private static func hamming(_ a: UInt64, _ b: UInt64) -> Int {
        (a ^ b).nonzeroBitCount
    }

    /// 内容复核：头尾代表文本的 3-gram Jaccard（独立于指纹，抗模板碰撞）。
    static func contentSimilar(_ a: String, _ b: String, _ opts: Options) -> Bool {
        guard opts.verify else { return true }
        func grams3(_ text: String) -> Set<String> {
            var s = Set<String>()
            for seg in segments(text, sampleChars: opts.verifySampleChars) {
                guard seg.count >= 3 else { continue }
                for i in 0...(seg.count - 3) { s.insert(String(seg[i..<(i + 3)])) }
            }
            return s
        }
        let ga = grams3(a), gb = grams3(b)
        let inter = ga.intersection(gb).count
        let union = ga.count + gb.count - inter
        guard union > 0 else { return false }
        return Double(inter) / Double(union) >= opts.verifyJaccard
    }

    // MARK: - SQLite 持久化索引

    static func createSchema(_ db: SQLiteDB) throws {
        try db.execute("""
            CREATE TABLE IF NOT EXISTS simhash_docs (
                path TEXT PRIMARY KEY,
                fp TEXT NOT NULL,
                hamming_check INTEGER NOT NULL DEFAULT 0
            )
            """)
        try db.execute("""
            CREATE TABLE IF NOT EXISTS simhash_bands (
                band INTEGER NOT NULL,
                key TEXT NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY (band, key, path)
            )
            """)
        try db.execute("CREATE INDEX IF NOT EXISTS simhash_bands_key ON simhash_bands(band, key)")
    }

    /// 查重复：4 表 LSH 粗排 → XOR+popcount 精排 → 内容复核。
    static func findDuplicate(text: String, path: String, in db: SQLiteDB,
                              opts: Options = Options()) throws -> Match? {
        let fp = fingerprint(text, opts)
        var candidates = Set<String>()
        for b in 0..<4 {
            let key = String(band(fp, b))
            let rows = try db.query("SELECT path FROM simhash_bands WHERE band = ? AND key = ?",
                                    [.int(Int64(b)), .text(key)])
            for row in rows {
                if let p = row["path"]?.stringValue { candidates.insert(p) }
            }
        }
        var best: Match?
        for p in candidates {
            if p == path { continue }  // 同一路径重同步不算重复（正文变了应重新切片）
            guard let rows = try? db.query("SELECT fp FROM simhash_docs WHERE path = ?", [.text(p)]),
                  let hex = rows.first?["fp"]?.stringValue,
                  let other = UInt64(hex, radix: 16) else { continue }
            let d = hamming(fp, other)
            guard d <= opts.ham else { continue }
            if opts.verify, let existingText = FileManager.default.contents(atPath: p).flatMap({ String(data: $0, encoding: .utf8) }),
               !contentSimilar(text, existingText, opts) { continue }
            if let cur = best {
                if d < cur.hamming { best = Match(path: p, hamming: d) }
            } else {
                best = Match(path: p, hamming: d)
            }
        }
        return best
    }

    /// 登记新文档指纹与 4 段键值（仅在切片入库成功后调用）。
    static func insert(text: String, path: String, into db: SQLiteDB,
                       opts: Options = Options()) throws {
        let fp = fingerprint(text, opts)
        try db.transaction {
            try db.execute("INSERT OR REPLACE INTO simhash_docs(path, fp) VALUES(?,?)",
                           [.text(path), .text(String(fp, radix: 16))])
            for b in 0..<4 {
                try db.execute("INSERT OR REPLACE INTO simhash_bands(band, key, path) VALUES(?,?,?)",
                               [.int(Int64(b)), .text(String(band(fp, b))), .text(path)])
            }
        }
    }

    /// 已登记文档数（展示/诊断用）。
    static func count(in db: SQLiteDB) -> Int {
        guard let rows = try? db.query("SELECT COUNT(*) AS n FROM simhash_docs"),
              let n = rows.first?["n"]?.intValue else { return 0 }
        return Int(n)
    }
}
