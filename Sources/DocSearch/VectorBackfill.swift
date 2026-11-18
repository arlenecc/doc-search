import Foundation

/// 后台词向量回填：扫描尚无残差的文本切片，计算词向量并写回向量表。
/// 与 GUI/CLI 解耦，全部在调用方的后台 Task 中执行。
enum VectorBackfill {
    /// 执行回填，返回本次处理的切片总数（0 表示无待回填项）。
    static func run(config: ToolConfig, dbPathOverride: String? = nil,
                    cancelled: @escaping () -> Bool,
                    progress: @escaping @Sendable (Int, Int) -> Void) async throws -> Int {
        let db = try SQLiteDB(path: dbPathOverride ?? config.databasePath)
        try SearchEngine.createSchema(db)

        let initial = SearchEngine(db: db,
                                   colbert: try await ColBERT.load(modelDir: config.modelDirectory, threads: 1),
                                   options: config.chunkOptions).pendingVectorCount()
        progress(0, initial)
        guard initial > 0 else { return 0 }
        AppLog.shared.log("[回填] 开始：待填充 \(initial) 片")

        let colbert = try await ColBERT.load(modelDir: config.modelDirectory,
                                              threads: max(1, config.vectorBackfillThreads))
        let engine = SearchEngine(db: db, colbert: colbert, options: config.chunkOptions)
        var done = 0
        while true {
            if cancelled() { throw CancellationError() }
            try Task.checkCancellation()
            let report = try engine.backfillVectors(colbert: colbert, bits: config.quantBits,
                                                    limit: 20, cancelled: cancelled)
            let batch = report.filled + report.skipped
            done += batch
            progress(done, initial)
            if batch == 0 { break }  // 全部填完
        }
        AppLog.shared.log("[回填] 完成：填充 \(done) 片")
        return done
    }
}
