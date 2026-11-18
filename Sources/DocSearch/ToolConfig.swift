import Foundation

/// 独立工具配置：只保留知识同步与检索阅读相关设置，JSON 持久化。
struct ToolConfig: Codable {
    var sourceDirectory = ""
    var outputDirectory = ""
    var databasePath = ""
    var modelDirectory = "/Volumes/SSD/embedding&reranker/jina-colbert-v2-64-onnx-int8"

    // 知识同步
    var syncConcurrency = 3
    var incrementalConversion = true
    var parseRenderDPI = 200.0
    var parseMaxPages = 0
    var parsePreprocessOCR = true
    var parseBinarizeOCR = true
    var parseKeepHeadersFooters = false

    // 文档解析 OCR 引擎：vision（Apple Vision）/ ovis（OvisOCR3）/ hybrid（两者结合）
    var pdfOCREngine = "vision"
    var epubOCREngine = "vision"
    /// OCR 视觉模型后端：mlx / gguf / onnx（按模型格式选择）。
    var ocrBackend = "mlx"
    /// OCR 模型路径：mlx 目录 / .gguf 文件 / onnx 目录或 .onnx 文件。
    var ovisModelPath = "models/OvisOCR3-MLX-4bit-mixed46"
    /// GGUF 视觉模型的多模态投影（mmproj）文件；留空则同目录自动查找。
    var ocrMMProjPath = ""
    /// OvisOCR3 并行实例数（1...2）。
    var ovisInstances = 2

    // 切片 / 词向量 / 残差
    var chunkMinChars = 7500
    var chunkMaxChars = 8000
    var chunkOverlapRatio = 0.015
    var semanticThreshold = 0.55
    var semanticFloor = 0.15
    /// potion-multilingual-128M（int4 ONNX）真语义切片；关闭则回退 ColBERT 词向量中心。
    var usePotionSemantic = true
    /// potion 模型目录（含 model.onnx/tokenizer.json/config.json）；留空自动查找。
    var potionModelPath = ""
    var quantBits = 4

    // 检索 / 阅读
    var resultTopN = 5
    var coarseCount = 20
    var nprobe = ANNIndex.defaultNprobe
    var rescore = ANNIndex.defaultRescore
    var annClusters = ANNIndex.clusters
    var annClusterSize = ANNIndex.clusterSize
    var useANN = true
    /// 重排后只保留分数 ≥ 最高分 × 该比例的结果（至少保留 1 片）。
    /// 可选类型保证旧配置文件缺字段时仍能正常解码。
    var minScoreRatio: Double?
    var effectiveMinScoreRatio: Double {
        min(max(minScoreRatio ?? 0.9, 0.0), 1.0)
    }

    static func defaultPaths() -> ToolConfig {
        var config = ToolConfig()
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first!
            .appendingPathComponent("DocSearch", isDirectory: true)
        config.outputDirectory = appSupport.appendingPathComponent("markdown", isDirectory: true).path
        config.databasePath = appSupport.appendingPathComponent("doc_search.db").path
        return config
    }

    static var configURL: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first!
            .appendingPathComponent("DocSearch", isDirectory: true)
        return root.appendingPathComponent("config.json")
    }

    static func load() -> ToolConfig {
        var result = defaultPaths()
        guard let data = try? Data(contentsOf: configURL),
              let decoded = try? JSONDecoder().decode(ToolConfig.self, from: data) else { return result }
        result = decoded
        // 兼容旧配置文件：空路径补默认值。
        if result.outputDirectory.isEmpty { result.outputDirectory = defaultPaths().outputDirectory }
        if result.databasePath.isEmpty { result.databasePath = defaultPaths().databasePath }
        if result.modelDirectory.isEmpty { result.modelDirectory = defaultPaths().modelDirectory }
        var changed = false
        // 旧配置里的 1% 是历史默认值，按本次调整迁移到 1.5%。
        if result.chunkOverlapRatio == 0.01 { result.chunkOverlapRatio = 0.015; changed = true }
        // 模型目录可能被移动到常见落点；配置的旧路径失效时自动纠正，避免检索直接报错。
        let fm = FileManager.default
        if !fm.fileExists(atPath: result.modelDirectory + "/tokenizer.json") {
            let name = (result.modelDirectory as NSString).lastPathComponent
            let home = fm.homeDirectoryForCurrentUser
            let candidates = [
                home.appendingPathComponent("app-space/models/\(name)"),
                URL(fileURLWithPath: "/Volumes/SSD/embedding&reranker/\(name)"),
                home.appendingPathComponent(".base-agent/models/\(name)"),
                home.appendingPathComponent("Library/Application Support/DocSearch/models/\(name)"),
            ]
            if let found = candidates.first(where: {
                fm.fileExists(atPath: $0.appendingPathComponent("tokenizer.json").path)
            }) {
                result.modelDirectory = found.path
                changed = true
            }
        }
        if changed { result.save() }
        return result
    }

    func save() {
        var copy = self
        if copy.sourceDirectory.isEmpty,
           let workspace = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            copy.sourceDirectory = workspace.path
        }
        guard let data = try? JSONEncoder().encode(copy) else { return }
        try? FileManager.default.createDirectory(at: Self.configURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: Self.configURL, options: .atomic)
    }

    var parserConfig: AgentConfig {
        var value = AgentConfig()
        value.ragIngestConcurrency = max(1, syncConcurrency)
        value.ragOCRPreprocess = parsePreprocessOCR
        value.ragOCRBinarize = parseBinarizeOCR
        value.ragPDFEngine = PDFEngine(stored: pdfOCREngine)
        value.ragEPUBOCREngine = PDFEngine(stored: epubOCREngine)
        // OvisOCR3：仅当选用 ovis/hybrid 时才真正解析/加载模型。
        value.ragOCRModelPath = ovisModelPath
        value.ragOCRModelName = "OvisOCR3-MLX-4bit-mixed46"
        value.ragOCRMMProjPath = ocrMMProjPath
        value.ragOvisInstances = max(1, min(2, ovisInstances))
        value.localVLMBackend = ocrBackend.isEmpty ? "mlx" : ocrBackend
        value.ragChunkSize = chunkMaxChars
        value.ragChunkOverlap = Int(Double(chunkMaxChars) * chunkOverlapRatio)
        value.ragSemanticChunking = true
        return value
    }

    var chunkOptions: ChunkOptions {
        var options = ChunkOptions(minChars: chunkMinChars, maxChars: chunkMaxChars,
                                   targetChars: (chunkMinChars + chunkMaxChars) / 2,
                                   overlapRatio: chunkOverlapRatio,
                                   semanticThreshold: semanticThreshold,
                                   similarityFloor: semanticFloor)
        options.usePotionSemantic = usePotionSemantic
        // 把显式 potion 目录注入相似度 provider。
        let trimmed = potionModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        PotionSentenceSimilarity.configuredDirectory = trimmed.isEmpty
            ? nil : URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
        return options
    }
}
