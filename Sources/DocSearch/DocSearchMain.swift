import Foundation
import ImageIO

enum DocSearchCLI {
    static let defaultModel = "/Volumes/SSD/embedding&reranker/jina-colbert-v2-64-onnx-int8"

    static let commands: Set<String> = [
        "init", "add", "sync", "search", "decision-search", "decision-verify", "sentence-test", "vector-fill", "self-test", "db-stats", "ann-status",
        "ann-build", "ann-drop", "ann-bench", "ocr-test", "chunk-test", "models",
        "-h", "--help", "help",
    ]

    static func isCLIInvocation(_ args: [String]) -> Bool {
        guard let first = args.first else { return false }
        return commands.contains(first)
    }

    static func run(_ args: [String]) async throws {
        guard let command = args.first else { print(usage); return }
        if command == "-h" || command == "--help" || command == "help" { print(usage); return }
        let options = parse(args.dropFirst())
        // CLI 与 GUI 共用 ToolConfig 的模型目录探测/回退，避免配置里的自定义路径失效。
        let config = ToolConfig.load()
        if command == "self-test" {
            try selfTest(withModel: options["model"] != nil)
            if options["model"] != nil { try await endToEnd(model: value(options, "model", config.modelDirectory)) }
            return
        }

        let dbPath = value(options, "db", config.databasePath)
        let modelDir = value(options, "model", ProcessInfo.processInfo.environment["DOCSEARCH_MODEL"] ?? config.modelDirectory)

        switch command {
        case "init":
            let db = try SQLiteDB(path: dbPath)
            try SearchEngine.createSchema(db)
            print("已建库: \(dbPath)")
            print("表: document / doc_summary_vec(vec0, float[\(ColBERT.dimension)] cosine) / doc_residuals")
            print("索引: document_fts(FTS5 BM25) / ANN 分块 IVF")
        case "add":
            guard let path = options["file"] else { throw DocSearchError.argument("add 需要 --file PATH") }
            let db = try SQLiteDB(path: dbPath)
            try SearchEngine.createSchema(db)
            var cfg = config
            if options["text-only"] != nil {
                // 纯文本入库：不加载 ColBERT，切片直接写 document/FTS，供决策判定即时用。
                let count = try SearchEngine.addFileTextOnly(
                    db: db, options: cfg.chunkOptions, path: path, title: options["title"],
                    progress: { done, total in
                        if done % 10 == 0 || done == total { print("  切片 \(done)/\(total) 已入库") }
                    })
                print("纯文本入库完成: \(path)，共 \(count) 个切片（词向量未计算）")
                return
            }
            let colbert = try await ColBERT.load(modelDir: modelDir,
                                                  threads: Int(options["threads"] ?? "") ?? 4)
            if let value = options["bits"] { cfg.quantBits = Int(value) ?? cfg.quantBits }
            let bits = cfg.quantBits
            var timings: [String: Double] = [:]
            let sink: (String, Double) -> Void = { key, value in timings[key] = value }
            let count = try SearchEngine(db: db, colbert: colbert,
                                         options: cfg.chunkOptions).addFile(
                path: path, title: options["title"], bits: bits,
                progress: { done, total in
                    if done % 10 == 0 || done == total { print("  切片 \(done)/\(total) 已编码入库") }
                },
                timing: options["timing"] != nil ? sink : nil)
            print("入库完成: \(path)，共 \(count) 个切片，量化 \(bits)bit")
            if options["timing"] != nil { printTiming(timings, prefix: "入库") }
        case "search":
            guard let query = options["query"] else { throw DocSearchError.argument("search 需要 --query TEXT") }
            let db = try SQLiteDB(path: dbPath, readOnly: true)
            let colbert = try await ColBERT.load(modelDir: modelDir,
                                                  threads: Int(options["threads"] ?? "") ?? 4)
            let engine = SearchEngine(db: db, colbert: colbert)
            let topN = Int(options["top"] ?? "5") ?? 5
            let coarse = Int(options["coarse"] ?? "20") ?? 20
            let hits = try engine.search(
                query: query, topN: topN, coarse: coarse,
                useANN: options["exact"] == nil,
                nprobe: Int(options["nprobe"] ?? "") ?? ANNIndex.defaultNprobe,
                rescore: Int(options["rescore"] ?? "") ?? ANNIndex.defaultRescore,
                forceANN: options["force-ann"] != nil,
                minScoreRatio: Double(options["min-score-ratio"] ?? "") ?? 0.9)
            print("问句: \(query)  (粗筛 \(coarse) 条 / 返回 \(hits.count) 条 / 向量通道 \(engine.lastVectorBackend))")
            for (i, hit) in hits.enumerated() {
                let vec = hit.vectorRank.map(String.init) ?? "-"
                let bm = hit.bm25Rank.map(String.init) ?? "-"
                print("\n[\(i + 1)] score=\(String(format: "%.4f", hit.score))  rrf=\(String(format: "%.4f", hit.rrf))  vec#\(vec) bm25#\(bm)")
                print("    doc_id=\(hit.docID)  title=\(hit.title)")
                let preview = hit.content.prefix(options["full"] != nil ? hit.content.count : 220)
                print("    \(preview)")
            }
        case "decision-search":
            guard let query = options["query"] else {
                throw DocSearchError.argument("decision-search 需要 --query TEXT")
            }
            let db = try SQLiteDB(path: dbPath)
            try SearchEngine.createSchema(db)
            let recall = Int(options["recall"] ?? "") ?? config.decisionRecallCount
            let topN = Int(options["top"] ?? "5") ?? 5
            let slices = try SearchEngine.bm25Slices(db: db, query: query, limit: recall)
            print("BM25 召回 \(slices.count) 个原文切片，开始决策判定…")
            let decisionPath = value(options, "model", config.decisionModelPath)
            let batchSize = Int(options["batch"] ?? "") ?? config.decisionBatchSize
            let loadStart = Date()
            let judge = try await DecisionJudge.load(
                modelPath: decisionPath,
                batchSize: batchSize,
                chunkChars: Int(options["chunk-chars"] ?? "") ?? config.decisionChunkChars,
                coreSentenceCount: Int(options["core"] ?? "") ?? config.decisionCoreSentences)
            let loadElapsed = Date().timeIntervalSince(loadStart)
            let judgeStart = Date()
            let verdicts = try await judge.judge(query: query, slices: slices, topN: topN) { done, total in
                print(String(format: "  判定进度 \(done)/\(total) · 已耗时 %.2fs", Date().timeIntervalSince(judgeStart)))
            }
            let judgeElapsed = Date().timeIntervalSince(judgeStart)
            let batches = (slices.count + batchSize - 1) / batchSize
            print("问句: \(query)  (BM25 \(slices.count) → 判定 \(verdicts.count) 条)")
            print(String(format: "[决策计时] 模型加载 %.3fs · 判定 %.3fs（\(batches) 批 × batch=\(batchSize)）· 合计 %.3fs",
                         loadElapsed, judgeElapsed, loadElapsed + judgeElapsed))
            for (i, verdict) in verdicts.enumerated() {
                let slice = verdict.slice
                print("\n[\(i + 1)] P(相关)=\(String(format: "%.4f", verdict.probability))  bm25#\(slice.bm25Rank)  score=\(String(format: "%.3f", slice.bm25Score))")
                print("    doc_id=\(slice.docID)  title=\(slice.title)")
                let preview = slice.content.prefix(options["full"] != nil ? slice.content.count : 220)
                print("    \(preview)")
            }
        case "vector-fill":
            // 单独运行后台词向量回填（同步未完成或中断后用）。
            var cfg = ToolConfig.load()
            if let value = options["model"] { cfg.modelDirectory = value }
            if let value = options["bits"] { cfg.quantBits = Int(value) ?? cfg.quantBits }
            if let value = options["threads"] { cfg.vectorBackfillThreads = Int(value) ?? cfg.vectorBackfillThreads }
            print("回填数据库: \(dbPath)")
            let filled = try await VectorBackfill.run(config: cfg, dbPathOverride: dbPath, cancelled: { false }) { done, total in
                if done == total || done % 20 == 0 { print("  词向量回填 \(done)/\(total)") }
            }
            print("词向量回填完成：\(filled) 片")
        case "sentence-test":
            guard let query = options["query"], let file = options["file"] else {
                throw DocSearchError.argument("sentence-test 需要 --query TEXT --file PATH")
            }
            guard let text = FileManager.default.contents(atPath: file).flatMap({ String(data: $0, encoding: .utf8) }) else {
                throw DocSearchError.argument("无法读取 UTF-8 文件: \(file)")
            }
            let core = SentenceBM25.coreSentences(query: query, content: text,
                                                  maxSentences: Int(options["max"] ?? "") ?? 5,
                                                  maxSentenceChars: Int(options["chars"] ?? "") ?? config.decisionChunkChars)
            print("query: \(query)")
            print("核心句（按原文顺序）:")
            for (i, line) in core.split(separator: "\n").enumerated() {
                print("  [\(i + 1)] \(line)")
            }
        case "decision-verify":
            // 读取 Python 参考实现保存的 token ids / marker / 概率，对比 Swift backbone 输出。
            guard let probePath = options["probe"] else {
                throw DocSearchError.argument("decision-verify 需要 --probe PATH（Python 参考 JSON）")
            }
            guard let data = FileManager.default.contents(atPath: probePath),
                  let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let ids = json["ids"] as? [Int],
                  let markers = json["markers"] as? [Int],
                  let expected = json["probs"] as? [Double] else {
                throw DocSearchError.argument("无法解析 probe JSON")
            }
            let decisionPath = value(options, "model", config.decisionModelPath)
            let judge = try await DecisionJudge.load(modelPath: decisionPath,
                                                     batchSize: Int(options["batch"] ?? "") ?? config.decisionBatchSize,
                                                     chunkChars: Int(options["chunk-chars"] ?? "") ?? config.decisionChunkChars)
            // 先核对 tokenizer：Swift 对同一渲染文本的编码是否与 Python 一致。
            if let rendered = json["rendered"] as? String {
                let swiftIDs = judge.tokenizeRendered(rendered)
                let tokenMatch = swiftIDs == ids
                print("Tokenizer 一致性：Swift \(swiftIDs.count) tokens vs Python \(ids.count) tokens \(tokenMatch ? "✓ 完全一致" : "✗ 不一致")")
                if !tokenMatch {
                    let prefix = zip(swiftIDs, ids).first(where: { $0.0 != $0.1 })
                    print("  首个差异：swift=\(prefix?.0 ?? -1) python=\(prefix?.1 ?? -1)")
                }
            }
            // 核对完整构造路径（模板 + chat template）：与探针同样的 query/chunks 是否产生同样 ids。
            let probeChunks = [
                "Order 1182: customer returns blender bought 45 days ago.",
                "What is the capital of France?",
                "The transformer uses self-attention mechanisms.",
                "Refund requests must be submitted within 30 days of purchase.",
            ]
            let built = try judge.probeIDs(query: "returns policy within 30 days", chunkTexts: probeChunks)
            let buildMatch = built == ids
            print("模板构造一致性：Swift \(built.count) tokens vs Python \(ids.count) tokens \(buildMatch ? "✓ 完全一致" : "✗ 不一致")")
            if !buildMatch {
                var firstDiff = -1
                for i in 0..<min(built.count, ids.count) where built[i] != ids[i] {
                    firstDiff = i
                    break
                }
                print("  首个差异 index=\(firstDiff) swift=\(firstDiff >= 0 ? built[firstDiff] : -1) python=\(firstDiff >= 0 ? ids[firstDiff] : -1)")
            }
            let actual = try await judge.judgeTokenIDs(ids, markers: markers)
            print("Swift vs Python 参考 P(true)（token 数 \(ids.count)，marker \(markers)）：")
            var maxDiff = 0.0
            for (i, (a, e)) in zip(actual, expected).enumerated() {
                let diff = abs(Double(a) - e)
                maxDiff = max(maxDiff, diff)
                print(String(format: "  #%d swift=%.4f python=%.4f diff=%.4f", i + 1, a, e, diff))
            }
            print(String(format: "最大差值 %.4f（<0.02 视为对齐）", maxDiff))
        case "sync":
            var cfg = ToolConfig.load()
            if let value = options["source"] { cfg.sourceDirectory = value }
            if let value = options["workdir"] {
                cfg.workDirectory = value
            } else if let value = options["output"] {
                cfg.workDirectory = (value as NSString).deletingLastPathComponent
            } else if let value = options["db"] {
                let parent = (value as NSString).deletingLastPathComponent
                cfg.workDirectory = (parent as NSString).lastPathComponent == "db"
                    ? (parent as NSString).deletingLastPathComponent : parent
            }
            if let value = options["model"] { cfg.modelDirectory = value }
            if let value = options["bits"] { cfg.quantBits = Int(value) ?? 4 }
            if let value = options["threads"] { cfg.syncConcurrency = Int(value) ?? 3 }
            if options["no-vector"] != nil { cfg.useVectorSearch = false }
            guard !cfg.sourceDirectory.isEmpty else {
                throw DocSearchError.argument("sync 需要 --source 目录或在配置面板指定源目录")
            }
            var lastPhase = ""
            let report = try await KnowledgeSyncService().run(config: cfg) { info in
                if info.phase != lastPhase || info.fileName.isEmpty {
                    print("  [\(info.phase)] \(info.fileName)\(info.message.isEmpty ? "" : " — \(info.message)")")
                    lastPhase = info.phase
                }
            }
            print("同步完成：文件 \(report.succeeded)/\(report.files)，跳过 \(report.skipped)，"
                  + "失败 \(report.failed)，切片 \(report.chunks)，耗时 \(String(format: "%.2f", report.elapsed))s")
            if cfg.useVectorSearch {
                print("文本切片已可用于决策判定；开始后台回填词向量…")
                let filled = try await VectorBackfill.run(config: cfg, cancelled: { false }) { done, total in
                    if done == total || done % 20 == 0 {
                        print("  词向量回填 \(done)/\(total)")
                    }
                }
                print("词向量回填完成：\(filled) 片")
            } else {
                print("向量检索已关闭：未计算词向量，仅保留文本切片供决策判定。")
            }
        case "db-stats":
            try dbStats(dbPath)
        case "ann-status":
            let db = try SQLiteDB(path: dbPath, readOnly: true)
            let s = ANNIndex(db: db).status()
            print("ANN: \(s.summary)")
            print("  可用=\(s.usable) 需重建=\(s.needsRebuild) 簇/块=\(s.clusters) "
                + "块=\(s.blocks) 已索引=\(s.indexedRows) 尾块=\(s.tailRows) 陈旧=\(s.staleRows)")
        case "ann-build":
            let db = try SQLiteDB(path: dbPath)
            let ann = ANNIndex(db: db)
            let result = try ann.build(
                clusters: Int(options["clusters"] ?? "") ?? ANNIndex.clusters,
                clusterSize: Int(options["cluster-size"] ?? "") ?? ANNIndex.clusterSize,
                iterations: Int(options["iterations"] ?? "") ?? ANNIndex.iterations,
                force: options["force"] != nil) { done, total in
                print("  已构建 \(done)/约\(total) 块")
            }
            print(result.summary)
        case "ann-drop":
            let db = try SQLiteDB(path: dbPath)
            try ANNIndex(db: db).drop()
            print("ANN 索引已删除")
        case "ann-bench":
            try annBench(options)
        case "ocr-test":
            try await ocrTest(options)
        case "chunk-test":
            try chunkTest(options)
        case "models":
            try await modelsCommand(options)
        default:
            print(usage)
        }
    }

    /// 验证文档解析 OCR 通道：OvisOCR3（MLX VLM）/ Apple Vision / 完整注册表路由。
    /// - `--image PATH`：单图识别（用 ImageIO 加载为 CGImage）。
    /// - `--pdf PATH`：通过注册表 parse 完整 PDF（按 --engine 路由）。
    private static func ocrTest(_ options: [String: String]) async throws {
        guard let path = options["image"] ?? options["pdf"] else {
            throw DocSearchError.argument("ocr-test 需要 --image PATH 或 --pdf PATH")
        }
        let isPDF = options["pdf"] != nil
        let engine = (options["engine"] ?? "ovis").lowercased()

        if isPDF {
            let cfg = ToolConfig.load()
            let config = cfg.parserConfig
            let registry = DocumentParserRegistry.makeDefault(config: config)
            let url = URL(fileURLWithPath: path)
            var options = ParserOptions(config: config)
            if engine == "vision" { options.pdfEngine = .vision }
            else if engine == "ovis" { options.pdfEngine = .ovis }
            else if engine == "hybrid" { options.pdfEngine = .hybrid }
            guard let parser = registry.freshParser(for: url, options: options) else {
                throw DocSearchError.argument("没有可用的 PDF 解析器")
            }
            let started = Date()
            let doc = try await parser.parse(url: url, options: options, progress: nil)
            print("PDF 解析成功: \(parser.name) 引擎=\(engine)")
            print("  页数=\(doc.pageCount) 字符=\(doc.text.count) 耗时=\(String(format: "%.2f", Date().timeIntervalSince(started)))s")
            print("  正文预览: \(doc.text.prefix(200))")
            return
        }

        // 单图：ImageIO 加载 CGImage。
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DocSearchError.argument("无法加载图片: \(path)")
        }

        let started = Date()
        switch engine {
        case "vision":
            let result = try await VisionPDFParser.recognizeSingle(
                image: image, preprocess: true, binarize: true,
                languages: ["zh-Hans", "en-US"])
            print("Vision OCR 结果 (\(String(format: "%.2f", Date().timeIntervalSince(started)))s):")
            print("  \(String((result.text ?? "").prefix(400)))")
        case "ovis", "vlm":
            var cfg = ToolConfig.load()
            let backend = (options["backend"] ?? cfg.ocrBackend).lowercased()
            guard backend == "mlx" || backend == "gguf" || backend == "onnx" else {
                throw DocSearchError.argument("--backend 仅支持 mlx / gguf / onnx")
            }
            cfg.ocrBackend = backend
            if let model = options["model"] { cfg.ovisModelPath = model }
            if let mmproj = options["mmproj"] { cfg.ocrMMProjPath = mmproj }
            let config = cfg.parserConfig
            let vlm = LocalModelFactory.shared.makeRAGOCRVLM(config: config)
            guard vlm.isAvailable else {
                throw DocSearchError.argument("OCR VLM 不可用: \(vlm.identifier)")
            }
            await LocalModelFactory.shared.beginOCRJob()
            defer { Task { await LocalModelFactory.shared.endOCRJob() } }
            let text = try await vlm.recognize(
                image: image, prompt: OvisPrompt.pageToMarkdown, maxTokens: 512)
            let cleaned = OvisPDFParser.cleanVLMOutput(text)
            print("OCR VLM 结果 [\(backend)] (\(String(format: "%.2f", Date().timeIntervalSince(started)))s):")
            print("  \(String(cleaned.prefix(400)))")
        default:
            throw DocSearchError.argument("--engine 仅支持 vision / ovis / hybrid")
        }
    }

    /// 验证 potion 语义切片：对比语义相关/无关句的余弦，并检查模型确实加载推理。
    private static func chunkTest(_ options: [String: String]) throws {
        let cfg = ToolConfig.load()
        let trimmed = cfg.potionModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            PotionSentenceSimilarity.configuredDirectory =
                URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
        }
        guard let dir = PotionSentenceSimilarity.modelDirectory() else {
            throw DocSearchError.model("未找到 potion 模型；请在配置中设置 potion 模型目录")
        }
        print("potion 模型目录: \(dir.path)")

        let potion = PotionSentenceSimilarity()
        let sentences = [
            "机器学习需要大量标注数据来训练深度神经网络。",
            "训练数据的质量直接影响模型在测试集上的准确率。",
            "今天下午三点在公司楼下咖啡店见面喝咖啡。",
            "这家咖啡店的拿铁和卡布奇诺味道非常不错。",
        ]
        guard let vectors = potion.sentenceVectors(sentences),
              vectors.count == sentences.count,
              vectors.allSatisfy({ $0.count == PotionSentenceSimilarity.dimension }) else {
            throw DocSearchError.model("potion 句向量推理失败（维度应为 \(PotionSentenceSimilarity.dimension)）")
        }
        func cosine(_ a: [Float], _ b: [Float]) -> Double {
            let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            let na = a.reduce(Float(0)) { $0 + $1 * $1 }
            let nb = b.reduce(Float(0)) { $0 + $1 * $1 }
            return na > 0 && nb > 0 ? Double(dot / (sqrt(na) * sqrt(nb))) : 0
        }
        let related = cosine(vectors[0], vectors[1])   // 机器学习 / 训练数据：应高
        let unrelated = cosine(vectors[0], vectors[2]) // 机器学习 / 咖啡店见面：应低
        print(String(format: "  语义相关句余弦 = %.4f（机器学习 / 训练数据）", related))
        print(String(format: "  语义无关句余弦 = %.4f（机器学习 / 咖啡店）", unrelated))
        guard related > unrelated + 0.1 else {
            throw DocSearchError.model("potion 语义区分度不足：相关句余弦未显著高于无关句")
        }
        print("  potion 语义信号有效（相关句相似度显著高于无关句）")

        // 端到端切一片长文本，确认走 potion 且切片完整。
        let paragraph = String(repeating:
            "机器学习需要大量标注数据来训练深度神经网络，训练数据的质量直接影响模型准确率。", count: 130)
        let chunks = try Chunker.split(paragraph, sourceName: "语义切片测试",
                                       model: nil, options: cfg.chunkOptions)
        print("  切片: \(chunks.count) 片，长度 \(chunks.map(\.content.count).min() ?? 0)...\(chunks.map(\.content.count).max() ?? 0)")
    }

    /// 模型管理：models（等价于 --status）/ models --status / models --download。
    private static func modelsCommand(_ options: [String: String]) async throws {
        let config = ToolConfig.load()
        let action = options["download"] != nil ? "download" : "status"
        let required = DocSearchModels.required(for: config)
        let missing = DocSearchModels.missing(for: config)

        if action == "status" {
            print("模型状态（共 \(required.count) 项，缺失 \(missing.count) 项）：")
            for model in required {
                let ready = DocSearchModels.isComplete(model)
                print(String(format: "  [%@] %@  %@", ready ? "就绪" : "缺失",
                             model.kind, model.repo))
                print("        落点: \(model.directory.path)")
            }
            if !missing.isEmpty {
                print("运行 `DocSearch models --download` 自动补全缺失模型（hf-mirror，支持断点续传）。")
            }
            return
        }

        guard !missing.isEmpty else { print("所有模型已就绪，无需下载。"); return }
        let fetcher = DocSearchModelDownloader()
        for model in missing {
            print("下载 \(model.kind): \(model.repo)")
            print("  落点: \(model.directory.path)")
            do {
                try await fetcher.download(model) { received, total in
                    let pct = total > 0 ? Double(received) / Double(total) * 100 : 0
                    print(String(format: "\r  进度: %.1f%% (%llu/%llu MB)",
                                 pct, received / 1_048_576, total / 1_048_576),
                          terminator: "")
                    fflush(stdout)
                }
                print("") // 换行结束进度行
                print("  ✓ \(model.kind) 下载完成")
            } catch {
                print("  ✗ \(model.kind) 下载失败: \(error.localizedDescription)")
                throw error
            }
        }
        print("全部缺失模型补全完成。")
    }

    private static func parse(_ args: ArraySlice<String>) -> [String: String] {
        let list = Array(args)
        var out: [String: String] = [:]
        var i = 0
        while i < list.count {
            let arg = list[i]
            i += 1
            guard arg.hasPrefix("--") else { continue }
            let body = String(arg.dropFirst(2))
            // 支持 --key=value，避免值以 -- 开头或含空格时被误判为下一个选项。
            if let eq = body.firstIndex(of: "=") {
                out[String(body[..<eq])] = String(body[body.index(after: eq)...])
                continue
            }
            let key = body
            if i < list.count, !list[i].hasPrefix("--") {
                out[key] = list[i]
                i += 1
            } else {
                out[key] = ""
            }
        }
        return out
    }

    private static func value(_ options: [String: String], _ key: String, _ fallback: String) -> String {
        options[key] ?? fallback
    }

    // MARK: - 计时与存储统计

    private static func printTiming(_ values: [String: Double], prefix: String) {
        let order = prefix == "入库"
            ? ["chunking", "prune_total", "encode_total", "quantize_total",
               "db_write_total", "add_total", "terms", "tokens", "chunks"]
            : ["query_center", "vector_coarse", "bm25_coarse", "rrf", "coarse_total",
               "query_tokens", "residual_load", "maxsim", "rerank", "search_total", "shortlist"]
        print("  [\(prefix)计时]")
        for key in order {
            guard let value = values[key] else { continue }
            if ["terms", "tokens", "chunks", "shortlist"].contains(key) {
                print("    \(key)=\(Int(value))")
            } else {
                print(String(format: "    %@ %.4f s",
                             key.padding(toLength: 16, withPad: " ", startingAt: 0), value))
            }
        }
        if let chunks = values["chunks"], chunks > 0 {
            for key in ["prune_total", "encode_total", "quantize_total", "db_write_total"] {
                guard let v = values[key] else { continue }
                let label = (key + "_per_chunk").padding(toLength: 16, withPad: " ", startingAt: 0)
                print(String(format: "    %@ %.4f s/片", label, v / chunks))
            }
        }
    }

    private static func dbStats(_ dbPath: String) throws {
        let db = try SQLiteDB(path: dbPath, readOnly: true)
        let docs = try db.query("SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(content)),0) AS chars FROM document")
            .first ?? [:]
        let residuals = try db.query("""
            SELECT COUNT(*) AS n, COALESCE(SUM(token_count),0) AS tokens,
                   COALESCE(SUM(LENGTH(residual_blob)),0) AS bytes,
                   COALESCE(AVG(token_count),0) AS avg_tokens
            FROM doc_residuals
            """).first ?? [:]
        let centers = try db.query("SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(embedding)),0) AS bytes FROM doc_vectors")
            .first ?? [:]
        let chunkCount = docs["n"]?.intValue ?? 0
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: dbPath)[.size] as? NSNumber)?.intValue ?? 0
        print("存储统计: \(dbPath)")
        print("  切片数=\(chunkCount) 文档文字=\(docs["chars"]?.intValue ?? 0) 字符")
        print("  每片平均: 正文 \(chunkCount > 0 ? (docs["chars"]?.intValue ?? 0) / chunkCount : 0) 字符, "
              + "词向量 \(Int(residuals["avg_tokens"]?.doubleValue ?? 0)) 个")
        print("  残差 BLOB 总计 \(residuals["bytes"]?.intValue ?? 0) B, "
              + "每片 \(chunkCount > 0 ? (residuals["bytes"]?.intValue ?? 0) / chunkCount : 0) B "
              + "(4bit: token_count × dim / 2)")
        print("  中心向量 BLOB 总计 \(centers["bytes"]?.intValue ?? 0) B, "
              + "每片 \(chunkCount > 0 ? (centers["bytes"]?.intValue ?? 0) / chunkCount : 0) B (fp32 64维)")
        print("  数据库文件 \(fileSize) B, 每片均摊 \(chunkCount > 0 ? fileSize / Int(chunkCount) : 0) B "
              + "(含 FTS5/vec0/ANN/页开销)")
        if let pages = try? db.query("SELECT name, SUM(pgsize) AS bytes FROM dbstat GROUP BY name ORDER BY bytes DESC") {
            for row in pages.prefix(8) {
                print("    \(row["name"]?.stringValue ?? "?") : \(row["bytes"]?.intValue ?? 0) B")
            }
        }
    }

    // MARK: - ANN 基准

    private static func annBench(_ options: [String: String]) throws {
        let rows = Int(options["rows"] ?? "50000") ?? 50_000
        let topK = Int(options["k"] ?? "20") ?? 20
        let nprobe = Int(options["nprobe"] ?? "") ?? ANNIndex.defaultNprobe
        let clusters = Int(options["clusters"] ?? "") ?? ANNIndex.clusters
        let temporaryDB = options["db"] == nil
        let dbPath = options["db"] ?? (NSTemporaryDirectory() + "docsearch-annbench-\(UUID().uuidString).db")
        let fm = FileManager.default
        defer {
            if temporaryDB {
                try? fm.removeItem(atPath: dbPath)
                try? fm.removeItem(atPath: dbPath + "-wal")
                try? fm.removeItem(atPath: dbPath + "-shm")
            }
        }
        print("ANN 基准: \(rows) 行 / \(clusters) 簇，库 \(dbPath)")
        let db = try SQLiteDB(path: dbPath)
        try SearchEngine.createSchema(db)
        let dim = ColBERT.dimension
        let centerCount = 32

        var seed: UInt64 = 0xDEAD_BEEF_1234_5678
        func next() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int(seed >> 40) % 2000 - 1000) / 1000.0
        }
        func center(_ c: Int) -> [Float] {
            (0..<dim).map { Float(cos(Double(c) * 0.7 + Double($0) * 0.31)) }
        }
        func normalized(_ v: [Float]) -> [Float] {
            var n: Float = 0
            for x in v { n += x * x }
            n = n > 0 ? sqrt(n) : 1
            return v.map { $0 / n }
        }
        func vector(_ c: Int) -> [Float] {
            let base = center(c)
            return normalized(base.enumerated().map { $0.element + next() * 0.15 })
        }

        try db.transaction {
            var buffer: [(id: Int64, vec: [Float])] = []
            buffer.reserveCapacity(500)
            func flush() throws {
                guard !buffer.isEmpty else { return }
                let docSQL = "INSERT INTO document(rowid,doc_id,title,content) VALUES "
                let vecSQL = "INSERT INTO doc_summary_vec(rowid,embedding) VALUES "
                let mirrorSQL = "INSERT INTO doc_vectors(rowid,embedding) VALUES "
                var dValues: [String] = []
                var vValues: [String] = []
                for item in buffer {
                    dValues.append("(\(item.id),'b\(item.id)','t','c')")
                    let hex = item.vec.withUnsafeBufferPointer { p in
                        Data(buffer: p).map { String(format: "%02x", $0) }.joined()
                    }
                    vValues.append("(\(item.id),x'\(hex)')")
                }
                try db.exec(docSQL + dValues.joined(separator: ","))
                try db.exec(vecSQL + vValues.joined(separator: ","))
                try db.exec(mirrorSQL + vValues.joined(separator: ","))
                buffer.removeAll(keepingCapacity: true)
            }
            for i in 1...rows {
                buffer.append((Int64(i), vector(i % centerCount)))
                if buffer.count == 500 { try flush() }
            }
            try flush()
            for (key, value) in [("doc_count", String(rows)), ("doc_rowid_seq", String(rows))] {
                try db.execute("""
                INSERT INTO doc_meta(key,value) VALUES(?,?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, [.text(key), .text(value)])
            }
        }

        let ann = ANNIndex(db: db)
        let t0 = Date()
        let built = try ann.build(clusters: clusters)
        print("  建索引: \(built.summary)")
        _ = built

        let query = normalized(center(0).enumerated().map { $0.element + next() * 0.05 })
        let centerQuery = query

        let v0 = Date()
        let exactRows = try db.query("""
            SELECT rowid FROM doc_summary_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance
            """, [.floats(centerQuery), .int(Int64(topK))])
        let exactTime = -v0.timeIntervalSinceNow
        let exactIDs = exactRows.compactMap { $0["rowid"]?.intValue }

        _ = ann.candidates(query: centerQuery, topK: topK, nprobe: nprobe) // warm cache
        let a0 = Date()
        let approx = ann.candidates(query: centerQuery, topK: topK, nprobe: nprobe) ?? []
        let annTime = -a0.timeIntervalSinceNow
        let approxIDs = approx.map(\.rowID)
        let hitCount = Set(exactIDs).intersection(Set(approxIDs)).count
        let recall = topK > 0 ? Double(hitCount) / Double(topK) : 0
        print(String(format: "  检索: ANN %.4fs vs vec0 精确 %.4fs (%.1fx)",
                     annTime, exactTime, exactTime / max(annTime, 1e-6)))
        print(String(format: "  recall@%d = %.0f%% (nprobe=%d)", topK, recall * 100, nprobe))
        print("  \(ANNIndex(db: db).status().summary)")
    }

    // MARK: - 离线自测

    private static func selfTest(withModel: Bool) throws {
        print("== 离线自测 ==")
        try quantRoundTrip(bits: 4)
        try quantRoundTrip(bits: 2)
        try vdsPCheck()
        try chunkCheck()
        try sqliteCheck()
        try simHashCheck()
        print(withModel ? "（带模型端到端测试将在建库后执行）" : "全部离线测试通过（加 --model PATH 可跑端到端）")
    }

    private static func simHashCheck() throws {
        // MurmurHash3 x64_128 (h1) 权威向量（与 tools/SimHashDedupe 自检一致）。
        let vectors: [(String, String)] = [
            ("", "0000000000000000"),
            ("foo", "e271865701f54561"),
            ("hello", "cbd8a7b341bd9b02"),
            ("量子力学", "dc380f4d0ae2fc34"),
            ("abcdefghijklmnopq", "7564747f88bda657"),
        ]
        for (s, expected) in vectors {
            let raw = String(SimHashStore.hash64(s), radix: 16)
            let actual = String(repeating: "0", count: max(0, 16 - raw.count)) + raw
            guard actual == expected else {
                throw DocSearchError.quant("MurmurHash 向量不符: \(s) \(actual) != \(expected)")
            }
        }
        // SQLite 索引：相同文本判重复，不同文本不判重复。
        let tmp = NSTemporaryDirectory() + "/simhash-check-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        let db = try SQLiteDB(path: tmp)
        try SimHashStore.createSchema(db)
        let body = String(repeating: "量子力学与机器学习文本内容", count: 40)
        let dir = (tmp as NSString).deletingLastPathComponent
        let a = dir + "/simhash-a.md"
        let b = dir + "/simhash-b.md"
        let c = dir + "/simhash-c.md"
        try body.write(toFile: a, atomically: true, encoding: .utf8)
        try body.write(toFile: b, atomically: true, encoding: .utf8)
        try String(repeating: "完全不同的另一段内容描述", count: 40)
            .write(toFile: c, atomically: true, encoding: .utf8)
        try SimHashStore.insert(text: body, path: a, into: db)
        guard let dup = try SimHashStore.findDuplicate(text: body, path: b, in: db), dup.hamming == 0 else {
            throw DocSearchError.quant("SimHash 重复判定失败")
        }
        guard try SimHashStore.findDuplicate(text: String(repeating: "完全不同的另一段内容描述", count: 40),
                                              path: c, in: db) == nil else {
            throw DocSearchError.quant("SimHash 误判不同文本为重复")
        }
        print("  SimHash 排重: Murmur 向量 5 项 + SQLite 索引重复/非重复判定通过")
    }

    private static func quantRoundTrip(bits: Int) throws {
        var seed: UInt64 = 0x1234_5678_9abc_def0
        func next() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int(seed >> 33) % 2000 - 1000) / 1000.0
        }
        let vectors = (0..<37).map { _ in (0..<64).map { _ in next() } }
        let record = try ResidualCodec.quantize(docID: "test", vectors: vectors, bits: bits)
        let expectedBytes = (vectors.count * 64 + (8 / bits) - 1) / (8 / bits)
        guard record.blob.count == expectedBytes, record.tokenCount == vectors.count else {
            throw DocSearchError.quant("打包长度错误")
        }
        var error: Float = 0
        let restored = decode(record)
        for (a, b) in zip(vectors, restored) {
            for i in a.indices { error += abs(a[i] - b[i]) }
        }
        error /= Float(vectors.count * 64)
        guard error < (bits == 4 ? 0.05 : 0.25) else {
            throw DocSearchError.quant("\(bits)bit 反量化误差过大: \(error)")
        }
        print("  \(bits)bit 残差往返: 平均绝对误差 \(String(format: "%.5f", error))，blob \(record.blob.count) B")
    }

    private static func vdsPCheck() throws {
        var seed: UInt64 = 42
        func next() -> Float {
            seed = seed &* 2862933555777941757 &+ 3037000493
            return Float(Int(seed >> 40) % 1000) / 1000.0
        }
        let query = (0..<9).map { _ in (0..<64).map { _ in next() } }
        let docs = (0..<53).map { _ in (0..<64).map { _ in next() } }
        let record = try ResidualCodec.quantize(docID: "v", vectors: docs, bits: 4)
        let fast = try ResidualCodec.maxSim(query: query, record: record, batchTokens: 7)
        let slow = ResidualCodec.naiveMaxSim(query: query, vectors: decode(record))
        let diff = abs(fast - slow)
        guard diff < 1e-3 else { throw DocSearchError.quant("vDSP MaxSim 与朴素实现不一致: \(diff)") }
        print("  vDSP 流式 MaxSim: fast=\(String(format: "%.4f", fast)) naive=\(String(format: "%.4f", slow))")

        let bigQuery = (0..<32).map { _ in (0..<64).map { _ in next() } }
        let bigDocs = (0..<4096).map { _ in (0..<64).map { _ in next() } }
        let bigRecord = try ResidualCodec.quantize(docID: "bench", vectors: bigDocs, bits: 4)
        let bigRestored = decode(bigRecord)
        let t0 = Date()
        _ = try ResidualCodec.maxSim(query: bigQuery, record: bigRecord, batchTokens: 512)
        let fastTime = -t0.timeIntervalSinceNow
        let t1 = Date()
        _ = ResidualCodec.naiveMaxSim(query: bigQuery, vectors: bigRestored)
        let naiveTime = -t1.timeIntervalSinceNow
        print("  vDSP 加速: \(String(format: "%.3f", fastTime))s vs 朴素 \(String(format: "%.3f", naiveTime))s "
              + "(\(String(format: "%.1f", naiveTime / max(fastTime, 1e-6)))x)")
    }

    private static func chunkCheck() throws {
        let longParagraph = Array(repeating: "语义检索系统需要把长文档切成语义完整的片段。", count: 2000).joined()
        let text = """
        # 第一章 检索系统

        \(longParagraph)

        ## 表格数据

        | 方法 | 分数 |
        | --- | --- |
        | ColBERT | 0.9 |

        ## 代码

        ```swift
        let score = maxSim(query, document)
        print(score)
        ```
        """
        let chunks = try Chunker.split(text, sourceName: "测试文档", model: nil)
        // 短章节（表格/代码块）按设计不再跨章节拼长，允许保留为独立短片；
        // 但任何切片都不能超过 maxChars，长章节必须产出至少一片 7500+ 的切片。
        guard !chunks.isEmpty,
              chunks.allSatisfy({ $0.content.count <= 3000 }),
              chunks.contains(where: { $0.content.count >= 2500 }),
              chunks.allSatisfy({ $0.content.count >= 2500
                || $0.content.contains("| 方法 |") || $0.content.contains("```") }) else {
            throw DocSearchError.chunking("切片长度异常: \(chunks.map(\.content.count))")
        }
        guard chunks.contains(where: { $0.content.contains("| 方法 | 分数 |") && $0.content.contains("| ColBERT | 0.9 |") }) else {
            throw DocSearchError.chunking("表格被拆开")
        }
        guard chunks.contains(where: { $0.content.contains("let score = maxSim") && $0.content.contains("```") }) else {
            throw DocSearchError.chunking("代码块被拆开")
        }
        print("  切片: \(chunks.count) 片，长度 \(chunks.map(\.content.count).min() ?? 0)...\(chunks.map(\.content.count).max() ?? 0)，表格/代码块完整")
    }

    private static func sqliteCheck() throws {
        let dir = NSTemporaryDirectory() + "docsearch-selftest-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let db = try SQLiteDB(path: dir + "/test.db")
        try SearchEngine.createSchema(db)
        try db.execute("INSERT INTO document(doc_id,title,content) VALUES('d1','t','检索系统')")
        try db.execute("INSERT INTO document_fts(doc_id,title_tokens,content_tokens) VALUES('d1','t','检索 系统')")
        try db.execute("INSERT INTO doc_summary_vec(rowid,embedding) VALUES(1,?)",
                       [.floats([Float](repeating: 0, count: 64))])
        let vec = try db.query("SELECT rowid, distance FROM doc_summary_vec WHERE embedding MATCH ? AND k = 1",
                               [.floats([Float](repeating: 0, count: 64))])
        let fts = try db.query("SELECT doc_id FROM document_fts WHERE document_fts MATCH ?", [.text("\"检索\" OR \"系统\"")])
        guard !vec.isEmpty, !fts.isEmpty else { throw DocSearchError.storage("vec0/FTS5 自测失败") }
        print("  SQLite: document/doc_summary_vec/doc_residuals/document_fts 建表与查询成功")
    }

    private static func decode(_ record: ResidualRecord) -> [[Float]] {
        let bytes = [UInt8](record.blob)
        let perByte = 8 / record.bits
        let mask = (1 << record.bits) - 1
        return (0..<record.tokenCount).map { token in
            (0..<record.dim).map { d in
                let pos = token * record.dim + d
                let code = Int((bytes[pos / perByte] >> UInt8((pos % perByte) * record.bits)) & UInt8(mask))
                return Float(Float16(Float(code) * record.scale + record.offset) + record.centerFP16[d])
            }
        }
    }

    private static func endToEnd(model: String) async throws {
        print("== 带模型端到端 ==")
        let dir = NSTemporaryDirectory() + "docsearch-e2e-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let file = dir + "/sample.md"
        let body = Array(repeating: "ColBERT 使用延迟交互计算查询与文档的细粒度相似度。向量检索结合 BM25 做混合召回。", count: 120).joined()
        try ("# 混合检索\n\n" + body).write(toFile: file, atomically: true, encoding: .utf8)
        let db = try SQLiteDB(path: dir + "/e2e.db")
        try SearchEngine.createSchema(db)
        let colbert = try await ColBERT.load(modelDir: model)
        let engine = SearchEngine(db: db, colbert: colbert)
        let count = try engine.addFile(path: file)
        let hits = try engine.search(query: "ColBERT 的延迟交互如何计算？", topN: 2)
        guard count > 0, let first = hits.first, first.score.isFinite else {
            throw DocSearchError.model("端到端检索失败")
        }
        print("  入库 \(count) 片；查询返回 \(hits.count) 条，top1 score=\(String(format: "%.4f", first.score))")
    }

    private static let usage = """
    doc_search — ColBERT 问句直接检索长文档（独立 Swift 工具）

      swift run -c release DocSearch sync  --source 源目录 [--workdir 工作目录] [--bits 4] [--threads 3] [--no-vector]
        # 同步先入库文本切片（决策判定立即可用），词向量随后后台回填；--no-vector 完全跳过词向量省空间
      swift run -c release DocSearch init  --db PATH
      swift run -c release DocSearch add   --db PATH --file 文本_path [--title 标题] [--bits 4] [--model DIR] [--timing] [--text-only]
      swift run -c release DocSearch search --db PATH --query "问句" [--top 5] [--coarse 20] [--nprobe 32] [--rescore 64] [--min-score-ratio 0.9] [--exact] [--force-ann]
      swift run -c release DocSearch decision-search --db PATH --query "问句" [--top 5] [--recall 50] [--batch 8] [--chunk-chars 160] [--model DIR] [--full]
        # 决策判定路径：BM25 从原文切片表召回 → 字符级 BM25 提核心句 → karpo 决策模型判定 → 取分数最高 topN
      swift run -c release DocSearch vector-fill [--db PATH] [--model DIR] [--bits 4] [--threads 2]
        # 单独运行后台词向量回填（同步中断或 App 重启后补算）
      swift run -c release DocSearch sentence-test --query "问句" --file 切片文本 [--max 5] [--chars 160]
        # 查看字符级 BM25 从切片里挑出的核心句（按原文顺序）
      swift run -c release DocSearch decision-verify --probe /path/to/probe.json
      swift run -c release DocSearch db-stats --db PATH
      swift run -c release DocSearch ann-status --db PATH
      swift run -c release DocSearch ann-build  --db PATH [--clusters 512] [--cluster-size 512] [--iterations 8] [--force]
      swift run -c release DocSearch ann-drop   --db PATH
      swift run -c release DocSearch ann-bench  [--rows 50000] [--k 20] [--clusters 512] [--nprobe 32]
      swift run -c release DocSearch ocr-test  --image 图片路径 [--engine ovis|vision] [--backend mlx|gguf|onnx] [--model PATH] [--mmproj PATH]
      swift run -c release DocSearch ocr-test  --pdf PDF路径 [--engine ovis|vision|hybrid]
      swift run -c release DocSearch chunk-test
      swift run -c release DocSearch models [--download]    # 检查/自动补全缺失模型
      swift run -c release DocSearch self-test [--model DIR]

    默认模型: 配置 modelDirectory（首次默认 \(defaultModel)；可用 --model 或 DOCSEARCH_MODEL 覆盖；也支持 --key=value 参数）
    表: document / doc_summary_vec(vec0) / doc_residuals；索引: document_fts(BM25) + ann_blocks/ann_cells(ANN)
    大库（≥20000 切片）且 ANN 可用时自动走 ANN 粗筛，小库走 vec0 精确检索
    每次检索返回前自动打印整体耗时与各阶段耗时（无需 --timing；该参数保留兼容但已忽略）
    配置只保留一个工作目录：Markdown = 工作目录/markdown，数据库 = 工作目录/db/doc_search.db
    sync 的旧参数 --output/--db 仍接受，会自动换算成工作目录；CLI 的 --db 仍可临时指定检索库
    """
}
