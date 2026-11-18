import Foundation
import OnnxRuntimeBindings
import Tokenizers

/// ColBERT 词向量引擎：严格复用模型目录 swift-example 的前缀、query 增强与输出约定。
final class ColBERT {
    enum Kind { case query, document }

    static let queryPrefix = "[QueryMarker]"
    static let documentPrefix = "[DocumentMarker]"
    static let queryMarkerID: Int = 250002
    static let documentMarkerID: Int = 250003
    static let maskID: Int = 250001
    static let padID: Int = 1
    static let queryMaxLen = 32
    static let dimension = 64

    private static let envLock = NSLock()
    private static var sharedEnv: ORTEnv?

    private let tokenizer: any Tokenizers.Tokenizer
    private let session: ORTSession
    private let outputName: String

    private init(tokenizer: any Tokenizers.Tokenizer, session: ORTSession, outputName: String) {
        self.tokenizer = tokenizer
        self.session = session
        self.outputName = outputName
    }

    static func load(modelDir: String, threads: Int = 4) async throws -> ColBERT {
        let dir = URL(fileURLWithPath: modelDir, isDirectory: true)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) else {
            throw DocSearchError.model("缺少 tokenizer.json: \(modelDir)")
        }
        let model = dir.appendingPathComponent("model_int8.onnx").path
        guard FileManager.default.fileExists(atPath: model) else {
            throw DocSearchError.model("缺少 model_int8.onnx: \(modelDir)")
        }

        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: dir, strict: false)
        let env: ORTEnv
        envLock.lock()
        if sharedEnv == nil { sharedEnv = try ORTEnv(loggingLevel: .warning) }
        let cached = sharedEnv!
        envLock.unlock()
        env = cached

        let options = try ORTSessionOptions()
        if threads > 0 { try options.setIntraOpNumThreads(Int32(threads)) }
        try options.setGraphOptimizationLevel(.all)
        let session = try ORTSession(env: env, modelPath: model, sessionOptions: options)
        guard let output = try session.outputNames().first else {
            throw DocSearchError.model("ONNX 图没有输出")
        }
        return ColBERT(tokenizer: tokenizer, session: session, outputName: output)
    }

    // MARK: - tokenisation

    private func encodeIDs(_ text: String, kind: Kind) -> [Int] {
        let prefix = kind == .query ? Self.queryPrefix : Self.documentPrefix
        // 查询侧与文档侧一样先剪枝（标点/空白/停用词/低信息词），
        // 但保留 query marker + <mask> 增强，不走逐词编码。
        let body = kind == .query ? TextAnalysis.prunedQueryText(text) : text
        return tokenizer.encode(text: prefix + " " + body, addSpecialTokens: true)
    }

    /// 实际送进 ONNX 的 token id 序列。marker 已在 encodeIDs 里由字符串映射为
    /// 250002/250003（紧跟 <s> 之后）；query 另按 ColBERT 约定补 <mask> 到 32。
    private func augmentedIDs(_ text: String, kind: Kind) -> [Int] {
        var ids = encodeIDs(text, kind: kind)
        if kind == .query {
            let target = max(Self.queryMaxLen, ids.count)
            ids.append(contentsOf: Array(repeating: Self.maskID, count: target - ids.count))
        }
        return ids
    }

    /// 返回每条文本的非特殊 token 位置（跳过 <s>/</s>/前缀标记）。
    private func contentIndices(_ ids: [Int], kind: Kind) -> [Int] {
        let marker = kind == .query ? Self.queryMarkerID : Self.documentMarkerID
        return ids.indices.filter { ids[$0] != 0 && ids[$0] != 2 && ids[$0] != marker }
    }

    // MARK: - ONNX

    private func run(ids: [[Int64]], mask: [[Int64]]) throws -> [[[Float]]] {
        guard let first = ids.first else { return [] }
        let seq = first.count
        func tensor(_ rows: [[Int64]]) throws -> ORTValue {
            var flat: [Int64] = []
            flat.reserveCapacity(rows.count * seq)
            for row in rows { flat.append(contentsOf: row) }
            let data = NSMutableData(capacity: flat.count * MemoryLayout<Int64>.stride)!
            flat.withUnsafeBytes { data.append($0.baseAddress!, length: $0.count) }
            return try ORTValue(tensorData: data,
                                elementType: .int64,
                                shape: [NSNumber(value: rows.count), NSNumber(value: seq)])
        }
        let outputs = try session.run(
            withInputs: ["input_ids": try tensor(ids), "attention_mask": try tensor(mask)],
            outputNames: [outputName],
            runOptions: nil)
        guard let value = outputs[outputName] else { throw DocSearchError.model("推理没有返回 \(outputName)") }
        let info = try value.tensorTypeAndShapeInfo()
        let shape = info.shape.map { $0.intValue }
        guard shape.count == 3, shape[2] == Self.dimension else {
            throw DocSearchError.model("输出形状异常: \(shape)")
        }
        let data = try value.tensorData()
        let count = shape[0] * shape[1] * shape[2]
        guard data.length >= count * MemoryLayout<Float>.stride else {
            throw DocSearchError.model("输出数据长度不足")
        }
        var flat = [Float](repeating: 0, count: count)
        flat.withUnsafeMutableBytes { data.getBytes($0.baseAddress!, length: count * MemoryLayout<Float>.stride) }
        var result = [[[Float]]]()
        result.reserveCapacity(shape[0])
        for b in 0..<shape[0] {
            var seqOut = [[Float]]()
            seqOut.reserveCapacity(shape[1])
            for s in 0..<shape[1] {
                let off = (b * shape[1] + s) * shape[2]
                seqOut.append(Array(flat[off..<(off + shape[2])]))
            }
            result.append(seqOut)
        }
        return result
    }

    /// 批量编码。query 先用 <mask> 补齐到至少 32，且 mask 位置参与注意力并保留在输出里
    /// （与模型目录 swift-example / infer.py 的 query augmentation 一致）；
    /// doc 用 PAD_ID 补齐并屏蔽 padding。输出保留每条文本的真实长度（含 query 的 mask）。
    private func encodeWithIDs(_ texts: [String], kind: Kind, batchSize: Int)
        throws -> (sequences: [[Int]], embeddings: [[[Float]]]) {
        guard !texts.isEmpty else { return ([], []) }
        let sequences = texts.map { augmentedIDs($0, kind: kind) }
        let target = sequences.map(\.count).max() ?? 0
        var out = [[[Float]]]()
        out.reserveCapacity(texts.count)
        var offset = 0
        while offset < sequences.count {
            let end = min(offset + max(1, batchSize), sequences.count)
            let ids = (offset..<end).map { i -> [Int64] in
                sequences[i].map(Int64.init)
                    + Array(repeating: Int64(Self.padID), count: target - sequences[i].count)
            }
            let mask = (offset..<end).map { i -> [Int64] in
                Array(repeating: Int64(1), count: sequences[i].count)
                    + Array(repeating: 0, count: target - sequences[i].count)
            }
            let batch = try run(ids: ids, mask: mask)
            for (j, seq) in batch.enumerated() {
                out.append(Array(seq.prefix(sequences[offset + j].count)))
            }
            offset = end
        }
        return (sequences, out)
    }

    func encode(_ texts: [String], kind: Kind, batchSize: Int = 32) throws -> [[[Float]]] {
        try encodeWithIDs(texts, kind: kind, batchSize: batchSize).embeddings
    }

    /// 对一组词/短语分别编码，取其内容 token 向量均值作为「这个词的向量」。
    func encodeTerms(_ terms: [String], batchSize: Int = 128) throws -> [[Float]] {
        guard !terms.isEmpty else { return [] }
        let seqs = terms.map { encodeIDs($0, kind: .document) }
        let target = seqs.map(\.count).max() ?? 0
        var output = [[Float]]()
        output.reserveCapacity(terms.count)
        var offset = 0
        while offset < seqs.count {
            let end = min(offset + max(1, batchSize), seqs.count)
            let ids = (offset..<end).map { i -> [Int64] in
                seqs[i].map { Int64($0) } + Array(repeating: Int64(Self.padID), count: target - seqs[i].count)
            }
            let mask = (offset..<end).map { i -> [Int64] in
                Array(repeating: Int64(1), count: seqs[i].count)
                    + Array(repeating: 0, count: target - seqs[i].count)
            }
            let batch = try run(ids: ids, mask: mask)
            for (j, seq) in batch.enumerated() {
                let positions = contentIndices(seqs[offset + j], kind: .document)
                let chosen = positions.isEmpty ? Array(0..<seq.count) : positions
                var center = [Float](repeating: 0, count: Self.dimension)
                for p in chosen where p < seq.count {
                    for d in 0..<Self.dimension { center[d] += seq[p][d] }
                }
                let denom = Float(max(1, chosen.count))
                // 模型逐 token 输出已 L2 归一化；词向量取均值后长度会缩水，
                // 统一重归一化后才能与 query token 向量做可比的点积/MaxSim。
                output.append(VectorMath.normalize(center.map { $0 / denom }))
            }
            offset = end
        }
        return output
    }

    /// 查询中心向量：取 query 增强后所有「内容 token」（含 <mask>，排除 <s>/</s>/标记）的均值。
    /// 一次 ONNX 推理同时产出 query token 向量和中心向量。
    /// 粗排与 MaxSim 重排共用，避免同一问句编码两次。
    func queryEncoding(_ query: String) throws -> (tokens: [[Float]], center: [Float]) {
        let (sequences, embeddings) = try encodeWithIDs([query], kind: .query, batchSize: 32)
        guard let seq = embeddings.first, let ids = sequences.first, !seq.isEmpty else {
            throw DocSearchError.model("问句编码为空")
        }
        // <mask> 拼在 </s> 之后，所以不能按“最后一个位置”排除 EOS；
        // 按 token id 过滤，250002/250003 也由 id 显式排除。
        let positions = contentIndices(ids, kind: .query).filter { $0 < seq.count }
        guard !positions.isEmpty else { throw DocSearchError.model("问句内容 token 为空") }
        var center = [Float](repeating: 0, count: Self.dimension)
        for p in positions { for d in 0..<Self.dimension { center[d] += seq[p][d] } }
        let n = Float(positions.count)
        return (seq, VectorMath.normalize(center.map { $0 / n }))
    }

    func queryCenter(_ query: String) throws -> [Float] {
        try queryEncoding(query).center
    }

    /// 相邻文本块的语义相似度（ColBERT 词向量中心余弦）；失败返回 nil 供切片器回退词法相似度。
    func adjacentSimilarities(_ texts: [String]) -> [Double]? {
        guard texts.count >= 2 else { return [] }
        do {
            let centers = try encodeTerms(texts)
            guard centers.count == texts.count else { return nil }
            return (0..<(texts.count - 1)).map {
                VectorMath.cosine(centers[$0], centers[$0 + 1])
            }
        } catch {
            return nil
        }
    }
}

enum VectorMath {
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        let den = sqrt(na) * sqrt(nb)
        return den > 0 ? Double(dot / den) : 0
    }

    static func normalize(_ v: [Float]) -> [Float] {
        var n: Float = 0
        for x in v { n += x * x }
        n = sqrt(n)
        guard n > 0 else { return v }
        return v.map { $0 / n }
    }
}
