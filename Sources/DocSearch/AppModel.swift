import Foundation
import SwiftUI

struct ChatBubbleItem: Identifiable {
    enum Role { case user, assistant }
    let id = UUID()
    var role: Role
    var text: String
    var hits: [SearchHit]? = nil
}

/// 后台回填取消标志（跨 actor 共享，内部加锁）。
final class BackfillCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
}

struct ReaderState: Identifiable {
    let id = UUID()
    let messageID: UUID
    var index: Int
    let hits: [SearchHit]
    let query: String
}

@MainActor
final class AppModel: ObservableObject {
    @Published var config = ToolConfig.load()
    @Published var messages: [ChatBubbleItem] = [
        ChatBubbleItem(role: .assistant,
                       text: "直接输入问题即可检索知识库；检索结果会以分片气泡显示，点击可打开阅读。")
    ]
    @Published var input = ""
    @Published var searchMode = ToolConfig.load().searchMode == "decision" ? "decision" : "vector"
    @Published var isSyncing = false
    @Published var isSearching = false
    @Published var syncStatus = "尚未同步"
    @Published var syncFraction = 0.0
    /// 后台词向量回填进度（向量检索启用时，同步完成后异步进行）。
    @Published var isBackfilling = false
    @Published var backfillDone = 0
    @Published var backfillTotal = 0
    @Published var lastReport: SyncReport?
    @Published var reader: ReaderState?
    @Published var status = ""

    // 缺失模型自动补全
    @Published var missingModels: [RequiredModel] = []
    @Published var isDownloading = false
    @Published var downloadFraction = 0.0
    @Published var downloadStatus = ""

    private var engine: SearchEngine?
    private var decisionJudge: DecisionJudge?
    private var backfillTask: Task<Void, Never>?
    private var backfillFlag: BackfillCancelFlag?
    private let syncService = KnowledgeSyncService()
    private let modelDownloader = DocSearchModelDownloader()

    init() {
        checkModels()
    }

    /// 检测当前配置缺失的模型（不阻塞 UI）。
    func checkModels() {
        missingModels = DocSearchModels.missing(for: config)
        if !missingModels.isEmpty {
            status = "缺失 \(missingModels.count) 个模型，可在左侧下载补全"
        }
    }

    /// 后台逐个下载缺失模型，聚合进度写到 downloadFraction。
    func downloadMissingModels() async {
        guard !isDownloading else { return }
        let models = missingModels
        guard !models.isEmpty else { return }
        isDownloading = true
        defer { isDownloading = false }
        do {
            for model in models {
                downloadStatus = "下载 \(model.kind)…"
                downloadFraction = 0
                try await modelDownloader.download(model) { received, total in
                    Task { @MainActor in
                        self.downloadFraction = total > 0 ? Double(received) / Double(total) : 0
                    }
                }
                // 一个模型完成后重算剩余（就绪判定以磁盘为准）。
                missingModels = DocSearchModels.missing(for: config)
            }
            downloadStatus = "模型补全完成"
            status = downloadStatus
            checkModels()
        } catch is CancellationError {
            downloadStatus = "下载已取消"
            status = downloadStatus
        } catch {
            downloadStatus = "下载失败：\(error.localizedDescription)"
            status = downloadStatus
        }
    }

    func cancelDownload() {
        modelDownloader.cancel()
        downloadStatus = "正在取消…"
    }

    func updateConfig(_ newConfig: ToolConfig) {
        config = newConfig
        config.save()
        engine = nil
        decisionJudge = nil
        searchMode = newConfig.searchMode == "decision" ? "decision" : "vector"
        checkModels()
        status = "配置已保存；检索前将重新加载模型/数据库"
    }

    func setSearchMode(_ mode: String) {
        searchMode = mode
        config.searchMode = mode
        config.save()
        status = mode == "decision"
            ? "决策判定：BM25 召回 + 决策模型判定（首次会加载模型）"
            : "向量检索：ColBERT + MaxSim"
    }

    func ensureEngine() async throws {
        if engine == nil {
            let colbert = try await ColBERT.load(modelDir: config.modelDirectory, threads: 4)
            let db = try SQLiteDB(path: config.databasePath)
            try SearchEngine.createSchema(db)
            engine = SearchEngine(db: db, colbert: colbert, options: config.chunkOptions)
        }
        guard engine != nil else { throw DocSearchError.model("检索引擎初始化失败") }
    }

    /// 决策判定路径：BM25 从原文切片表召回 → 决策模型逐片判定 → 取分数最高 topN。
    private func decisionSearch(query: String) async throws -> [SearchHit] {
        let cfg = config
        if decisionJudge == nil {
            let judge = try await Task.detached(priority: .userInitiated) {
                try await DecisionJudge.load(modelPath: cfg.decisionModelPath,
                                             batchSize: cfg.decisionBatchSize,
                                             chunkChars: cfg.decisionChunkChars,
                                             coreSentenceCount: cfg.decisionCoreSentences)
            }.value
            decisionJudge = judge
        }
        guard let judge = decisionJudge else { throw DocSearchError.model("决策模型初始化失败") }

        let dbPath = cfg.databasePath
        let recall = max(1, cfg.decisionRecallCount)

        // 判定较重（4B 模型逐批前向），放后台线程避免阻塞 UI。
        let verdicts = try await Task.detached(priority: .userInitiated) {
            let db = try SQLiteDB(path: dbPath)
            try SearchEngine.createSchema(db)
            let slices = try SearchEngine.bm25Slices(db: db, query: query, limit: recall)
            return try await judge.judge(query: query, slices: slices, topN: cfg.resultTopN)
        }.value

        return verdicts.map { verdict in
            let slice = verdict.slice
            return SearchHit(docID: slice.docID, title: slice.title, content: slice.content,
                             score: verdict.probability, vectorRank: nil,
                             bm25Rank: slice.bm25Rank, rrf: 0)
        }
    }

    func send() async {
        let query = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, !isSearching else { return }
        input = ""
        messages.append(ChatBubbleItem(role: .user, text: query))
        isSearching = true
        status = "检索中…"
        defer { isSearching = false }
        do {
            let q = query
            let hits: [SearchHit]
            if searchMode == "decision" || !config.useVectorSearch {
                // 关闭向量检索时，即使开关在向量模式也自动走决策判定，保证可用。
                if !config.useVectorSearch && searchMode != "decision" {
                    messages.append(ChatBubbleItem(role: .assistant,
                        text: "向量检索已在配置中关闭，已自动改用决策判定模式。"))
                }
                status = decisionJudge == nil ? "决策判定：首次加载模型…" : "决策判定中…"
                hits = try await decisionSearch(query: q)
            } else {
                try await ensureEngine()
                guard let engine else { throw DocSearchError.model("检索引擎初始化失败") }
                let topN = config.resultTopN
                let coarse = config.coarseCount
                let useANN = config.useANN
                let nprobe = config.nprobe
                let rescore = config.rescore
                let minScoreRatio = config.effectiveMinScoreRatio
                // 检索含 ONNX 推理与 MaxSim 重排，放到后台线程，避免阻塞 @MainActor 的 UI。
                hits = try await Task.detached(priority: .userInitiated) {
                    try engine.search(query: q, topN: topN, coarse: coarse,
                                      useANN: useANN, nprobe: nprobe, rescore: rescore,
                                      minScoreRatio: minScoreRatio)
                }.value
            }
            messages.append(ChatBubbleItem(
                role: .assistant,
                text: hits.isEmpty ? "没有召回相关分片。"
                    : (searchMode == "decision" || !config.useVectorSearch)
                        ? "决策判定召回 \(hits.count) 个相关分片（按判定分数排序），点击气泡阅读："
                        : "召回 \(hits.count) 个分片，点击气泡阅读：",
                hits: hits))
            status = "检索完成：\(hits.count) 个分片"
        } catch {
            messages.append(ChatBubbleItem(role: .assistant, text: "检索失败：\(error.localizedDescription)"))
            status = "检索失败：\(error.localizedDescription)"
            AppLog.shared.log("[检索] 失败：\(error.localizedDescription)")
        }
    }

    func sync() async {
        guard !isSyncing, !config.sourceDirectory.isEmpty else {
            status = "请先在配置面板指定源目录"
            return
        }
        isSyncing = true
        syncStatus = "准备同步…"
        syncFraction = 0
        AppLog.shared.log("[同步] 开始")
        defer { isSyncing = false }
        var lastLoggedPhase = ""
        do {
            let report = try await syncService.run(config: config) { info in
                self.syncStatus = "\(info.phase)：\(info.fileName.isEmpty ? info.message : info.fileName)"
                self.syncFraction = info.fraction
                // 只在处理阶段切换时记一条，避免逐文件刷屏。
                if info.phase != lastLoggedPhase {
                    AppLog.shared.log("[同步] 进入 \(info.phase)")
                    lastLoggedPhase = info.phase
                }
            }
            lastReport = report
            syncStatus = report.message
            syncFraction = 1
            status = report.message
            AppLog.shared.log("[同步] 完成：文件 \(report.succeeded)/\(report.files)，切片 \(report.chunks)，跳过 \(report.skipped)")
            engine = nil
            // 文本切片已可立即用于决策判定；词向量在后台异步填充。
            startVectorBackfillIfNeeded()
        } catch is CancellationError {
            syncStatus = "同步已停止"
            status = "同步已停止"
            AppLog.shared.log("[同步] 已停止")
        } catch {
            syncStatus = "同步失败：\(error.localizedDescription)"
            status = syncStatus
            AppLog.shared.log("[同步] 失败：\(error.localizedDescription)")
        }
    }

    func stopSync() { syncService.cancel() }

    /// 同步完成后启动后台词向量回填；关闭向量检索模式则不启动。
    func startVectorBackfillIfNeeded() {
        guard config.useVectorSearch else { return }
        backfillTask?.cancel()
        let cfg = config
        // 用本地可变标志供后台读取，避免跨 actor 访问。
        let flag = BackfillCancelFlag()
        backfillFlag = flag
        let task = Task.detached(priority: .utility) { [weak self] in
            do {
                let total = try await VectorBackfill.run(config: cfg, cancelled: { flag.get() }) { done, all in
                    Task { @MainActor in
                        self?.isBackfilling = true
                        self?.backfillDone = done
                        self?.backfillTotal = all
                    }
                }
                await MainActor.run {
                    self?.isBackfilling = false
                    self?.backfillDone = total
                    self?.backfillTotal = total
                    // 回填完成后丢弃缓存引擎，后续向量检索重建以读取新向量。
                    self?.engine = nil
                }
            } catch is CancellationError {
                await MainActor.run { self?.isBackfilling = false }
            } catch {
                await MainActor.run {
                    self?.isBackfilling = false
                    AppLog.shared.log("[回填] 失败：\(error.localizedDescription)")
                }
            }
        }
        backfillTask = task
        Task { @MainActor in
            self.isBackfilling = true
            self.backfillDone = 0
        }
    }

    func stopVectorBackfill() {
        backfillFlag?.set(true)
        backfillTask?.cancel()
        isBackfilling = false
        AppLog.shared.log("[回填] 已停止")
    }

    func openReader(messageID: UUID, index: Int) {
        guard let message = messages.first(where: { $0.id == messageID }),
              let hits = message.hits, hits.indices.contains(index) else { return }
        reader = ReaderState(messageID: messageID, index: index, hits: hits, query: message.text)
    }

    func moveReader(_ delta: Int) {
        guard var state = reader else { return }
        let next = state.index + delta
        guard state.hits.indices.contains(next) else { return }
        state.index = next
        reader = state
    }
}
