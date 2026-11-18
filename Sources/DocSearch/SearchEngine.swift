import Foundation

struct SearchHit {
    let docID: String
    let title: String
    let content: String
    let score: Float
    let vectorRank: Int?
    let bm25Rank: Int?
    let rrf: Double
}

struct Candidate {
    var docID: String
    var rrf: Double = 0
    var vectorRank: Int? = nil
    var bm25Rank: Int? = nil
    var vectorDistance: Double? = nil
    var bm25Score: Double? = nil
}

/// 建库 / 入库 / 检索。三张要求的表 + 一个 FTS5 BM25 索引（不是数据表）。
final class SearchEngine {
    let db: SQLiteDB
    let colbert: ColBERT
    let options: ChunkOptions
    private let rrfK = 60.0
    /// AND 命中少于该数量时换用 OR，避免中文长 query 只剩一两条候选。
    private let minBM25Hits = 5
    /// 最近一次检索的向量通道（ann / vec0-exact），供 CLI 输出。
    private(set) var lastVectorBackend = "vec0-exact"

    init(db: SQLiteDB, colbert: ColBERT, options: ChunkOptions = ChunkOptions()) {
        self.db = db
        self.colbert = colbert
        self.options = options
    }

    static func createSchema(_ db: SQLiteDB) throws {
        try db.exec("""
        CREATE TABLE IF NOT EXISTS document (
            doc_id TEXT PRIMARY KEY,
            title  TEXT NOT NULL,
            content TEXT NOT NULL
        );
        """)
        try db.exec("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
            doc_id UNINDEXED,
            title_tokens,
            content_tokens,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """)
        // sqlite-vec 0.1.9 的 vec0 只支持向量列；用 rowid 与 document 行一一对齐。
        // embedding 列不能 SELECT 出原始字节，另建普通镜像表供 ANN/反量化读取。
        try db.exec("""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_summary_vec USING vec0(
            embedding float[\(ColBERT.dimension)] distance_metric=cosine
        );
        CREATE TABLE IF NOT EXISTS doc_vectors (
            rowid INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL
        );
        """)
        try db.exec("""
        CREATE TABLE IF NOT EXISTS doc_residuals (
            doc_id TEXT PRIMARY KEY,
            token_count INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            quant_bits INTEGER NOT NULL,
            scale REAL NOT NULL,
            offset REAL NOT NULL,
            residual_blob BLOB NOT NULL
        );
        """)
        try ANNIndex(db: db).createSchema()
        if Self.meta(db, "doc_count") == nil {
            let count = (try? db.query("SELECT COUNT(*) AS n FROM document").first?["n"]?.intValue) ?? 0
            Self.setMeta(db, "doc_count", String(count))
        }
        if Self.meta(db, "doc_rowid_seq") == nil {
            let maxID = (try? db.query("SELECT COALESCE(MAX(rowid),0) AS n FROM document")
                .first?["n"]?.intValue) ?? 0
            Self.setMeta(db, "doc_rowid_seq", String(maxID))
        }
    }

    // MARK: - meta helpers

    private static func meta(_ db: SQLiteDB, _ key: String) -> String? {
        try? db.query("SELECT value FROM doc_meta WHERE key = ?", [.text(key)])
            .first?["value"]?.stringValue
    }

    private static func setMeta(_ db: SQLiteDB, _ key: String, _ value: String) {
        try? db.execute("""
        INSERT INTO doc_meta(key,value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(key), .text(value)])
    }

    // MARK: - 入库

    func addFile(path: String, title: String? = nil, bits: Int = 4,
                 progress: ((Int, Int) -> Void)? = nil,
                 timing: ((String, Double) -> Void)? = nil,
                 cancelled: (() -> Bool)? = nil) throws -> Int {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else {
            throw DocSearchError.argument("无法读取 UTF-8 文本文件: \(path)")
        }
        let base = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        let sourceID = base + "-" + hash8(text)
        let docTitle = title ?? base
        let addStarted = Date()
        let chunks = try Chunker.split(text, sourceName: docTitle, model: colbert, options: options)
        let chunkingTime = Date().timeIntervalSince(addStarted)
        timing?("chunking", chunkingTime)
        guard !chunks.isEmpty else { throw DocSearchError.chunking("没有产生任何切片") }

        var pruneTotal = 0.0, encodeTotal = 0.0, quantTotal = 0.0, dbTotal = 0.0
        var termTotal = 0, tokenTotal = 0
        let ann = ANNIndex(db: db)

        // 清理旧切片单独成事务：只删不编码，避免长事务把 ONNX 推理也包进去。
        try db.transaction {
            let old = try db.query("SELECT doc_id, rowid FROM document WHERE doc_id LIKE ?",
                                   [.text(sourceID + "#%")])
            var staleIDs: [Int64] = []
            for row in old {
                guard let id = row["doc_id"]?.stringValue else { continue }
                if let rowID = row["rowid"]?.intValue {
                    staleIDs.append(rowID)
                    try db.execute("DELETE FROM doc_summary_vec WHERE rowid = ?", [.int(rowID)])
                    try db.execute("DELETE FROM doc_vectors WHERE rowid = ?", [.int(rowID)])
                }
                try db.execute("DELETE FROM document WHERE doc_id = ?", [.text(id)])
                try db.execute("DELETE FROM document_fts WHERE doc_id = ?", [.text(id)])
                try db.execute("DELETE FROM doc_residuals WHERE doc_id = ?", [.text(id)])
            }
            try ann.recordStale(rowIDs: staleIDs)
            let previous = Int(Self.meta(db, "doc_count") ?? "") ?? 0
            Self.setMeta(db, "doc_count", String(max(0, previous - old.count)))
        }

        // 逐片编码、逐片提交：取消能及时响应，写锁也不再覆盖整个文件。
        var added = 0
        for (i, chunk) in chunks.enumerated() {
            if cancelled?() == true { throw CancellationError() }
            let id = String(format: "%@#%04d", sourceID, i)
            var stage = Date()
            var terms = TokenPruner.shared.prune(chunk.content)
            if terms.isEmpty { terms = TextAnalysis.words(chunk.content).filter { !TokenPruner.shared.isStopword($0) } }
            pruneTotal += Date().timeIntervalSince(stage)
            guard !terms.isEmpty else { continue }
            termTotal += terms.count

            stage = Date()
            let vectors = try colbert.encodeTerms(terms)
            encodeTotal += Date().timeIntervalSince(stage)
            tokenTotal += vectors.count

            stage = Date()
            let record = try ResidualCodec.quantize(docID: id, vectors: vectors,
                                                    dim: ColBERT.dimension, bits: bits)
            quantTotal += Date().timeIntervalSince(stage)

            stage = Date()
            try db.transaction {
                let rowSeq = (Int64(Self.meta(db, "doc_rowid_seq") ?? "") ?? 0) + 1
                try db.execute("INSERT INTO document(rowid,doc_id,title,content) VALUES(?,?,?,?)",
                               [.int(rowSeq), .text(id), .text(chunk.title), .text(chunk.content)])
                try db.execute("INSERT INTO document_fts(doc_id,title_tokens,content_tokens) VALUES(?,?,?)",
                               [.text(id), .text(TextAnalysis.ftsIndexText(chunk.title)),
                                .text(TextAnalysis.ftsIndexText(chunk.content))])
                try db.execute("INSERT INTO doc_summary_vec(rowid,embedding) VALUES(?,?)",
                               [.int(rowSeq), .floats(record.center)])
                try db.execute("INSERT INTO doc_vectors(rowid,embedding) VALUES(?,?)",
                               [.int(rowSeq), .floats(record.center)])
                try db.execute("""
                    INSERT INTO doc_residuals(doc_id,token_count,dim,quant_bits,scale,offset,residual_blob)
                    VALUES(?,?,?,?,?,?,?)
                    """, [.text(id), .int(Int64(record.tokenCount)), .int(Int64(record.dim)), .int(Int64(record.bits)),
                          .double(Double(record.scale)), .double(Double(record.offset)), .blob(record.blob)])
                let count = Int(Self.meta(db, "doc_count") ?? "") ?? 0
                Self.setMeta(db, "doc_count", String(count + 1))
                Self.setMeta(db, "doc_rowid_seq", String(rowSeq))
            }
            dbTotal += Date().timeIntervalSince(stage)
            added += 1
            progress?(i + 1, chunks.count)
        }

        let total = Date().timeIntervalSince(addStarted)
        timing?("prune_total", pruneTotal)
        timing?("encode_total", encodeTotal)
        timing?("quantize_total", quantTotal)
        timing?("db_write_total", dbTotal)
        timing?("add_total", total)
        timing?("terms", Double(termTotal))
        timing?("tokens", Double(tokenTotal))
        timing?("chunks", Double(added))
        return chunks.count
    }

    // MARK: - 检索

    func search(query: String, topN: Int = 5, coarse: Int = 20,
                useANN: Bool = true, nprobe: Int = ANNIndex.defaultNprobe,
                rescore: Int = ANNIndex.defaultRescore, forceANN: Bool = false,
                minScoreRatio: Double = 0.9,
                timing: ((String, Double) -> Void)? = nil) throws -> [SearchHit] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let searchStarted = Date()
        var stage = Date()
        let encodedQuery = try colbert.queryEncoding(query)
        let center = encodedQuery.center
        let queryElapsed = Date().timeIntervalSince(stage)
        timing?("query_center", queryElapsed)
        timing?("query_tokens", queryElapsed)
        let fetchK = max(60, coarse * 3)
        var candidates: [String: Candidate] = [:]

        stage = Date()
        var pairs: [(id: String, score: Double, distance: Double?)] = []
        let ann = ANNIndex(db: db)
        let totalRows = Int(Self.meta(db, "doc_count") ?? "") ?? 0
        if useANN, (forceANN || totalRows >= ANNIndex.minRowsForANN), ann.status().usable,
           let found = ann.candidates(query: center, topK: fetchK,
                                      nprobe: nprobe, rescore: rescore), !found.isEmpty {
            lastVectorBackend = "ANN(nprobe=\(nprobe),rescore=\(rescore))"
            for item in found {
                guard let id = try db.query("SELECT doc_id FROM document WHERE rowid = ?",
                                            [.int(item.rowID)]).first?["doc_id"]?.stringValue else { continue }
                pairs.append((id, item.score, nil))
            }
        }
        if pairs.isEmpty {
            lastVectorBackend = "vec0-exact"
            let rows = try db.query("""
                SELECT rowid, distance FROM doc_summary_vec
                WHERE embedding MATCH ? AND k = ? ORDER BY distance
                """, [.floats(center), .int(Int64(fetchK))])
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let id = try db.query("SELECT doc_id FROM document WHERE rowid = ?",
                                            [.int(rowID)]).first?["doc_id"]?.stringValue else { continue }
                pairs.append((id, 1.0 / (1.0 + (row["distance"]?.doubleValue ?? 2.0)),
                              row["distance"]?.doubleValue ?? nil))
            }
        }
        timing?("vector_coarse", Date().timeIntervalSince(stage))
        stage = Date()
        for (rank, pair) in pairs.enumerated() {
            var c = candidates[pair.id] ?? Candidate(docID: pair.id)
            c.vectorRank = rank + 1
            c.vectorDistance = pair.distance
            c.rrf += 1.0 / (rrfK + Double(rank + 1))
            candidates[pair.id] = c
        }

        func searchBM25(_ mode: String) throws -> [[String: SQLValue]] {
            let match = TextAnalysis.ftsQuery(query, mode: mode)
            guard !match.isEmpty else { return [] }
            return try db.query("""
                SELECT doc_id, bm25(document_fts) AS score FROM document_fts
                WHERE document_fts MATCH ? ORDER BY score LIMIT ?
                """, [.text(match), .int(Int64(fetchK))])
        }
        var bm25Rows = try searchBM25("AND")
        // 中文长 query 的 AND 常只剩一两条命中，OR 兜底给 RRF 足够候选；
        // OR 仍按 bm25 打分排序，多词命中的文档会排在前面。
        if bm25Rows.count < minBM25Hits {
            let orRows = try searchBM25("OR")
            if orRows.count > bm25Rows.count { bm25Rows = orRows }
        }
        for (rank, row) in bm25Rows.enumerated() {
            guard let id = row["doc_id"]?.stringValue else { continue }
            var c = candidates[id] ?? Candidate(docID: id)
            c.bm25Rank = rank + 1
            c.bm25Score = row["score"]?.doubleValue ?? nil
            c.rrf += 1.0 / (rrfK + Double(rank + 1))
            candidates[id] = c
        }

        timing?("bm25_coarse", Date().timeIntervalSince(stage))
        stage = Date()
        let shortlist = candidates.values.sorted {
            if $0.rrf != $1.rrf { return $0.rrf > $1.rrf }
            return $0.docID < $1.docID
        }.prefix(max(1, coarse))
        timing?("rrf", Date().timeIntervalSince(stage))
        timing?("coarse_total", Date().timeIntervalSince(searchStarted))

        let queryTokens = encodedQuery.tokens
        guard !queryTokens.isEmpty else { return [] }

        var loadTotal = 0.0, maxsimTotal = 0.0
        var hits: [SearchHit] = []
        for candidate in shortlist {
            var itemStart = Date()
            guard let record = loadResidual(candidate.docID),
                  let meta = try loadMeta(candidate.docID) else { continue }
            loadTotal += Date().timeIntervalSince(itemStart)
            itemStart = Date()
            let score = try ResidualCodec.maxSim(query: queryTokens, record: record)
            maxsimTotal += Date().timeIntervalSince(itemStart)
            hits.append(SearchHit(docID: candidate.docID, title: meta.title, content: meta.content,
                                  score: score, vectorRank: candidate.vectorRank,
                                  bm25Rank: candidate.bm25Rank, rrf: candidate.rrf))
        }
        hits.sort { $0.score != $1.score ? $0.score > $1.score : $0.docID < $1.docID }
        // 低分切片不再凑数：只保留分数达到最高分一定比例的结果，至少留 1 片。
        var result = hits
        if let best = hits.first?.score, best > 0, minScoreRatio > 0 {
            let kept = hits.filter { $0.score >= best * Float(minScoreRatio) }
            if !kept.isEmpty { result = kept }
        }
        timing?("residual_load", loadTotal)
        timing?("maxsim", maxsimTotal)
        timing?("rerank", loadTotal + maxsimTotal)
        timing?("shortlist", Double(shortlist.count))
        timing?("search_total", Date().timeIntervalSince(searchStarted))
        return Array(result.prefix(max(1, topN)))
    }

    private func loadMeta(_ docID: String) throws -> (title: String, content: String)? {
        let rows = try db.query("SELECT title, content FROM document WHERE doc_id = ?", [.text(docID)])
        guard let row = rows.first else { return nil }
        return (row["title"]?.stringValue ?? "", row["content"]?.stringValue ?? "")
    }

    private func loadResidual(_ docID: String) -> ResidualRecord? {
        guard let rows = try? db.query("""
            SELECT token_count, dim, quant_bits, scale, offset, residual_blob
            FROM doc_residuals WHERE doc_id = ?
            """, [.text(docID)]), let row = rows.first,
              let count = row["token_count"]?.intValue,
              let dim = row["dim"]?.intValue,
              let bits = row["quant_bits"]?.intValue,
              let scale = row["scale"]?.doubleValue,
              let offset = row["offset"]?.doubleValue,
              let blob = row["residual_blob"]?.dataValue,
              let rowRows = try? db.query("SELECT rowid FROM document WHERE doc_id = ?", [.text(docID)]),
              let rowID = rowRows.first?["rowid"]?.intValue,
              let vecRows = try? db.query("SELECT embedding FROM doc_vectors WHERE rowid = ?", [.int(rowID)]),
              let centerData = vecRows.first?["embedding"]?.dataValue else { return nil }
        guard centerData.count == Int(dim) * MemoryLayout<Float>.stride else { return nil }
        let centerFloats = centerData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let center = centerFloats.map { Float16($0) }
        guard center.count == dim else { return nil }
        return ResidualRecord(docID: docID, tokenCount: Int(count), dim: Int(dim), bits: Int(bits),
                              scale: Float(scale), offset: Float(offset),
                              centerFP16: center, blob: blob)
    }

    // MARK: - helpers

    private func blob16(_ values: [Float16]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func decode16(_ data: Data) -> [Float16] {
        guard data.count % 2 == 0 else { return [] }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }

    private func hash8(_ text: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(format: "%08llx", hash & 0xffff_ffff)
    }
}

/// 检索只在 AppModel 的后台 Task 中调用，且 UI 侧用 isSearching 串行化；
/// SQLiteDB/ORTSession 自身加锁，故显式标记为可跨线程传递。
extension SearchEngine: @unchecked Sendable {}
