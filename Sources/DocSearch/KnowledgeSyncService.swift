import Foundation

struct SyncProgressInfo {
    var phase: String
    var completed = 0
    var total = 0
    var fileName = ""
    var message = ""
    var fraction = 0.0
}

struct SyncReport {
    var files = 0
    var succeeded = 0
    var failed = 0
    var skipped = 0
    var chunks = 0
    var elapsed: TimeInterval = 0
    var message = ""
}

/// 一次性目录同步：复用主应用的 Markdown 转换链路 → 本工具的切片/词向量/入库。
/// GUI 与 CLI 共用同一实现。
final class KnowledgeSyncService {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock(); cancelled = true; lock.unlock()
    }

    private var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    func run(config: ToolConfig,
             progress: @escaping @MainActor (SyncProgressInfo) -> Void) async throws -> SyncReport {
        lock.lock(); cancelled = false; lock.unlock()
        let source = URL(fileURLWithPath: (config.sourceDirectory as NSString).expandingTildeInPath,
                         isDirectory: true)
        let output = URL(fileURLWithPath: (config.outputDirectory as NSString).expandingTildeInPath,
                         isDirectory: true)
        let database = (config.databasePath as NSString).expandingTildeInPath

        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: (database as NSString).deletingLastPathComponent),
            withIntermediateDirectories: true)

        await progress(SyncProgressInfo(phase: "加载模型与数据库", message: config.modelDirectory))
        let colbert = try await ColBERT.load(modelDir: config.modelDirectory,
                                              threads: max(1, config.syncConcurrency))
        let db = try SQLiteDB(path: database)
        try SearchEngine.createSchema(db)
        let engine = SearchEngine(db: db, colbert: colbert, options: config.chunkOptions)

        let request = MarkdownConversionRequest(sourceURL: source, outputDirectory: output,
                                                config: config.parserConfig,
                                                incremental: config.incrementalConversion)
        let conversion = try await MarkdownConversionService().convert(request: request) { item in
            let info = SyncProgressInfo(
                phase: Self.phaseName(item.phase),
                completed: item.completedFiles,
                total: item.fileTotal,
                fileName: item.fileName,
                message: item.stage,
                fraction: item.overallFraction)
            Task { @MainActor in progress(info) }
        }

        var result = SyncReport()
        result.files = conversion.results.count
        result.succeeded = conversion.succeededCount
        result.failed = conversion.failedCount
        result.skipped = conversion.skippedCount
        result.elapsed = conversion.elapsed

        // 增量跳过的文件正文未变，切片/词向量也已在库中，不能再重编码入库。
        let successes = conversion.results.filter { $0.succeeded && !$0.skipped }
        for (index, item) in successes.enumerated() {
            try Task.checkCancellation()
            if isCancelled { throw CancellationError() }
            guard let url = item.outputURL else { continue }
            await progress(SyncProgressInfo(
                phase: "切片入库",
                completed: index + 1,
                total: successes.count,
                fileName: item.source.lastPathComponent,
                message: "词向量与 4bit 残差",
                fraction: Double(index + 1) / Double(max(1, successes.count))))
            do {
                let title = (item.source.lastPathComponent as NSString).deletingPathExtension
                let fileName = item.source.lastPathComponent
                let fileBase = Double(index) / Double(max(1, successes.count))
                let fileSpan = 1.0 / Double(max(1, successes.count))
                var lastReported = -5
                // 大文件切片很多时给 UI 报进度，但不写日志，避免日志刷屏。
                let ingestProgress: (Int, Int) -> Void = { done, total in
                    guard done == total || done - lastReported >= 5 else { return }
                    lastReported = done
                    let inner = Double(done) / Double(max(1, total))
                    let info = SyncProgressInfo(
                        phase: "切片入库",
                        completed: done,
                        total: total,
                        fileName: fileName,
                        message: "词向量与 4bit 残差 \(done)/\(total)",
                        fraction: fileBase + inner * fileSpan)
                    Task { @MainActor in progress(info) }
                }
                result.chunks += try engine.addFile(path: url.path, title: title,
                                                    bits: config.quantBits,
                                                    progress: ingestProgress,
                                                    cancelled: { [weak self] in self?.isCancelled ?? false })
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                result.failed += 1
                if result.message.isEmpty { result.message = error.localizedDescription }
            }
        }
        if result.message.isEmpty {
            result.message = "同步完成：\(result.succeeded) 个文件，\(result.chunks) 个切片"
        }
        return result
    }

    private static func phaseName(_ phase: MarkdownConversionProgress.Phase) -> String {
        switch phase {
        case .scanning: return "扫描文件"
        case .parsing: return "解析提取"
        case .writing: return "写入 Markdown"
        case .finished: return "完成"
        }
    }
}
