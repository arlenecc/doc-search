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

        // Markdown 生成后直接切文本入库，不加载 ColBERT。词向量由后台回填服务异步补算；
        // 关闭向量检索模式时则完全不计算词向量，节省空间。
        let db = try SQLiteDB(path: database)
        try SearchEngine.createSchema(db)
        try SimHashStore.createSchema(db)
        // 首次启用排重时，用已有 Markdown 文件初始化索引，使历史入库文档也参与比对。
        if SimHashStore.count(in: db) == 0,
           let enumerator = FileManager.default.enumerator(at: output, includingPropertiesForKeys: nil) {
            var seeded = 0
            for case let fileURL as URL in enumerator where fileURL.pathExtension.lowercased() == "md" {
                if let text = try? String(contentsOf: fileURL, encoding: .utf8) {
                    try? SimHashStore.insert(text: text, path: fileURL.path, into: db)
                    seeded += 1
                }
            }
            if seeded > 0 { AppLog.shared.log("[排重] 已用 \(seeded) 个历史 Markdown 初始化索引") }
        }
        await progress(SyncProgressInfo(
            phase: "准备数据库",
            message: config.useVectorSearch ? "文本切片入库（词向量后台填充）" : "纯文本入库（向量检索未启用）"))

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
            // 全局文本相似排重：Markdown 已生成、切片入库前，对全文算 SimHash，
            // 与已入库文档索引比对；重复则跳过切片入库。
            if let markdownText = try? String(contentsOf: url, encoding: .utf8),
               let duplicate = try? SimHashStore.findDuplicate(text: markdownText, path: url.path, in: db) {
                result.skipped += 1
                let reason = "重复于 \((duplicate.path as NSString).lastPathComponent)（海明 \(duplicate.hamming)）"
                AppLog.shared.log("[排重] 跳过相似文档：\(item.source.lastPathComponent) — \(reason)")
                await progress(SyncProgressInfo(
                    phase: "排重跳过",
                    completed: index + 1,
                    total: successes.count,
                    fileName: item.source.lastPathComponent,
                    message: reason,
                    fraction: Double(index + 1) / Double(max(1, successes.count))))
                continue
            }
            await progress(SyncProgressInfo(
                phase: "切片入库",
                completed: index + 1,
                total: successes.count,
                fileName: item.source.lastPathComponent,
                message: config.useVectorSearch ? "文本切片（词向量后台填充）" : "文本切片",
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
                        message: "文本切片 \(done)/\(total)",
                        fraction: fileBase + inner * fileSpan)
                    Task { @MainActor in progress(info) }
                }
                // 同步阶段只写文本切片；词向量由 AppModel/CLI 在同步完成后后台回填。
                result.chunks += try SearchEngine.addFileTextOnly(
                    db: db, options: config.chunkOptions, path: url.path, title: title,
                    progress: ingestProgress,
                    cancelled: { [weak self] in self?.isCancelled ?? false })
                // 入库成功才登记指纹，避免切片失败留下错误索引。
                if let markdownText = try? String(contentsOf: url, encoding: .utf8) {
                    try? SimHashStore.insert(text: markdownText, path: url.path, into: db)
                }
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
