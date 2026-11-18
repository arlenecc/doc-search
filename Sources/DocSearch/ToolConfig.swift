import Foundation

/// 独立工具配置：只保留知识同步与检索阅读相关设置，JSON 持久化。
struct ToolConfig {
    var sourceDirectory = ""
    /// 唯一的工作目录：Markdown 输出为 <工作目录>/markdown，数据库为 <工作目录>/db/doc_search.db。
    var workDirectory = ""
    var modelDirectory = "/Volumes/SSD/embedding&reranker/jina-colbert-v2-64-onnx-int8"
    /// 决策判定模型目录（默认 karpo-decision-0.8B-mlx-int4，含 readout.safetensors）。
    var decisionModelPath = "/Volumes/SSD/Decision Model/karpo-decision-0.8B-mlx-int4"
    /// 检索路径：vector=向量检索（默认），decision=BM25 召回 + 决策模型判定。
    var searchMode = "vector"
    /// 决策模式 BM25 召回切片数。
    var decisionRecallCount = 50
    /// 决策模式单次前向的切片数（越大越快，但占内存）。
    var decisionBatchSize = 8
    /// 决策模式每片送入判定的最大正文字数。
    var decisionChunkChars = 160
    /// 决策模式每个切片提取的核心句数量。
    var decisionCoreSentences = 3

    var expandedWorkDirectory: String {
        let raw = workDirectory.isEmpty ? Self.defaultWorkDirectory() : workDirectory
        return (raw as NSString).expandingTildeInPath
    }

    var outputDirectory: String {
        URL(fileURLWithPath: expandedWorkDirectory, isDirectory: true)
            .appendingPathComponent("markdown", isDirectory: true).path
    }

    var databasePath: String {
        URL(fileURLWithPath: expandedWorkDirectory, isDirectory: true)
            .appendingPathComponent("db", isDirectory: true)
            .appendingPathComponent("doc_search.db").path
    }

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
    /// 是否启用向量检索模式。关闭时同步只入库文本切片（供决策判定用），不计算词向量/残差以节省空间。
    var useVectorSearch = true
    /// 后台词向量回填并发/线程数。
    var vectorBackfillThreads = 2
    var chunkMinChars = 2500
    var chunkMaxChars = 3000
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

    static func defaultWorkDirectory() -> String {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("DocSearch", isDirectory: true).path
    }

    static func defaultPaths() -> ToolConfig {
        var config = ToolConfig()
        config.workDirectory = defaultWorkDirectory()
        return config
    }

    static var configURL: URL {
        URL(fileURLWithPath: defaultWorkDirectory()).appendingPathComponent("config.json")
    }

    static func load() -> ToolConfig {
        var result = defaultPaths()
        var changed = false
        var hasNewWorkDir = false
        var legacyDatabasePath: String?
        if let data = try? Data(contentsOf: configURL),
           let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            for (key, value) in dict {
                switch key {
                case "sourceDirectory": result.sourceDirectory = value as? String ?? result.sourceDirectory
                case "workDirectory":
                    if let string = value as? String, !string.isEmpty {
                        result.workDirectory = string
                        hasNewWorkDir = true
                    }
                case "outputDirectory":
                    if !hasNewWorkDir, let string = value as? String, !string.isEmpty {
                        result.workDirectory = (string as NSString).deletingLastPathComponent
                        changed = true
                    }
                case "databasePath":
                    if let string = value as? String, !string.isEmpty {
                        legacyDatabasePath = string
                        if !hasNewWorkDir {
                            let ns = string as NSString
                            let parent = ns.deletingLastPathComponent
                            result.workDirectory = (parent as NSString).lastPathComponent == "db"
                                ? (parent as NSString).deletingLastPathComponent : parent
                            changed = true
                        }
                    }
                case "modelDirectory": result.modelDirectory = value as? String ?? result.modelDirectory
                case "decisionModelPath": result.decisionModelPath = value as? String ?? result.decisionModelPath
                case "searchMode": result.searchMode = value as? String ?? result.searchMode
                case "decisionRecallCount": result.decisionRecallCount = value as? Int ?? result.decisionRecallCount
                case "decisionBatchSize": result.decisionBatchSize = value as? Int ?? result.decisionBatchSize
                case "decisionChunkChars": result.decisionChunkChars = value as? Int ?? result.decisionChunkChars
                case "decisionCoreSentences": result.decisionCoreSentences = value as? Int ?? result.decisionCoreSentences
                case "syncConcurrency": result.syncConcurrency = value as? Int ?? result.syncConcurrency
                case "incrementalConversion": result.incrementalConversion = value as? Bool ?? result.incrementalConversion
                case "parseRenderDPI": result.parseRenderDPI = value as? Double ?? result.parseRenderDPI
                case "parseMaxPages": result.parseMaxPages = value as? Int ?? result.parseMaxPages
                case "parsePreprocessOCR": result.parsePreprocessOCR = value as? Bool ?? result.parsePreprocessOCR
                case "parseBinarizeOCR": result.parseBinarizeOCR = value as? Bool ?? result.parseBinarizeOCR
                case "parseKeepHeadersFooters": result.parseKeepHeadersFooters = value as? Bool ?? result.parseKeepHeadersFooters
                case "pdfOCREngine": result.pdfOCREngine = value as? String ?? result.pdfOCREngine
                case "epubOCREngine": result.epubOCREngine = value as? String ?? result.epubOCREngine
                case "ocrBackend": result.ocrBackend = value as? String ?? result.ocrBackend
                case "ovisModelPath": result.ovisModelPath = value as? String ?? result.ovisModelPath
                case "ocrMMProjPath": result.ocrMMProjPath = value as? String ?? result.ocrMMProjPath
                case "ovisInstances": result.ovisInstances = value as? Int ?? result.ovisInstances
                case "chunkMinChars": result.chunkMinChars = value as? Int ?? result.chunkMinChars
                case "chunkMaxChars": result.chunkMaxChars = value as? Int ?? result.chunkMaxChars
                case "chunkOverlapRatio": result.chunkOverlapRatio = value as? Double ?? result.chunkOverlapRatio
                case "semanticThreshold": result.semanticThreshold = value as? Double ?? result.semanticThreshold
                case "semanticFloor": result.semanticFloor = value as? Double ?? result.semanticFloor
                case "usePotionSemantic": result.usePotionSemantic = value as? Bool ?? result.usePotionSemantic
                case "potionModelPath": result.potionModelPath = value as? String ?? result.potionModelPath
                case "quantBits": result.quantBits = value as? Int ?? result.quantBits
                case "useVectorSearch": result.useVectorSearch = value as? Bool ?? result.useVectorSearch
                case "vectorBackfillThreads": result.vectorBackfillThreads = value as? Int ?? result.vectorBackfillThreads
                case "resultTopN": result.resultTopN = value as? Int ?? result.resultTopN
                case "coarseCount": result.coarseCount = value as? Int ?? result.coarseCount
                case "nprobe": result.nprobe = value as? Int ?? result.nprobe
                case "rescore": result.rescore = value as? Int ?? result.rescore
                case "annClusters": result.annClusters = value as? Int ?? result.annClusters
                case "annClusterSize": result.annClusterSize = value as? Int ?? result.annClusterSize
                case "useANN": result.useANN = value as? Bool ?? result.useANN
                case "minScoreRatio": result.minScoreRatio = value as? Double
                default: break
                }
            }
            // 旧数据库位于 <工作目录>/doc_search.db，迁移到 <工作目录>/db/doc_search.db。
            if let legacy = legacyDatabasePath,
               FileManager.default.fileExists(atPath: legacy),
               !FileManager.default.fileExists(atPath: result.databasePath) {
                let fm = FileManager.default
                try? fm.createDirectory(atPath: (result.databasePath as NSString).deletingLastPathComponent,
                                        withIntermediateDirectories: true)
                try? fm.moveItem(atPath: legacy, toPath: result.databasePath)
                for suffix in ["-wal", "-shm"] {
                    try? fm.moveItem(atPath: legacy + suffix, toPath: result.databasePath + suffix)
                }
                changed = true
            }
        }
        if result.workDirectory.isEmpty { result.workDirectory = Self.defaultWorkDirectory() }
        let fm = FileManager.default
        try? fm.createDirectory(atPath: result.expandedWorkDirectory, withIntermediateDirectories: true)
        try? fm.createDirectory(atPath: (result.databasePath as NSString).deletingLastPathComponent,
                                withIntermediateDirectories: true)
        if result.modelDirectory.isEmpty { result.modelDirectory = defaultPaths().modelDirectory }
        // 旧配置里的历史默认值迁移到新的切片尺寸。
        if result.chunkOverlapRatio == 0.01 { result.chunkOverlapRatio = 0.015; changed = true }
        if result.chunkMinChars == 7500 { result.chunkMinChars = 2500; changed = true }
        if result.chunkMaxChars == 8000 { result.chunkMaxChars = 3000; changed = true }
        // 决策模型默认从 4B 切到 0.8B：仅重写仍是旧默认路径的配置，显式自定义的不动。
        if result.decisionModelPath == "/Volumes/SSD/Decision Model/karpo-decision-4B-mlx-int4",
           fm.fileExists(atPath: defaultPaths().decisionModelPath + "/readout.safetensors") {
            result.decisionModelPath = defaultPaths().decisionModelPath
            changed = true
        }
        // 模型目录可能被移动到常见落点；配置的旧路径失效时自动纠正，避免检索直接报错。
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
        if copy.workDirectory.isEmpty { copy.workDirectory = Self.defaultWorkDirectory() }
        var dict: [String: Any] = [
            "sourceDirectory": copy.sourceDirectory,
            "workDirectory": copy.workDirectory,
            "modelDirectory": copy.modelDirectory,
            "decisionModelPath": copy.decisionModelPath,
            "searchMode": copy.searchMode,
            "decisionRecallCount": copy.decisionRecallCount,
            "decisionBatchSize": copy.decisionBatchSize,
            "decisionChunkChars": copy.decisionChunkChars,
            "decisionCoreSentences": copy.decisionCoreSentences,
            "syncConcurrency": copy.syncConcurrency,
            "incrementalConversion": copy.incrementalConversion,
            "parseRenderDPI": copy.parseRenderDPI,
            "parseMaxPages": copy.parseMaxPages,
            "parsePreprocessOCR": copy.parsePreprocessOCR,
            "parseBinarizeOCR": copy.parseBinarizeOCR,
            "parseKeepHeadersFooters": copy.parseKeepHeadersFooters,
            "pdfOCREngine": copy.pdfOCREngine,
            "epubOCREngine": copy.epubOCREngine,
            "ocrBackend": copy.ocrBackend,
            "ovisModelPath": copy.ovisModelPath,
            "ocrMMProjPath": copy.ocrMMProjPath,
            "ovisInstances": copy.ovisInstances,
            "chunkMinChars": copy.chunkMinChars,
            "chunkMaxChars": copy.chunkMaxChars,
            "chunkOverlapRatio": copy.chunkOverlapRatio,
            "semanticThreshold": copy.semanticThreshold,
            "semanticFloor": copy.semanticFloor,
            "usePotionSemantic": copy.usePotionSemantic,
            "potionModelPath": copy.potionModelPath,
            "quantBits": copy.quantBits,
            "useVectorSearch": copy.useVectorSearch,
            "vectorBackfillThreads": copy.vectorBackfillThreads,
            "resultTopN": copy.resultTopN,
            "coarseCount": copy.coarseCount,
            "nprobe": copy.nprobe,
            "rescore": copy.rescore,
            "annClusters": copy.annClusters,
            "annClusterSize": copy.annClusterSize,
            "useANN": copy.useANN,
        ]
        if let minScoreRatio = copy.minScoreRatio { dict["minScoreRatio"] = minScoreRatio }
        guard JSONSerialization.isValidJSONObject(dict),
              let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]) else { return }
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
