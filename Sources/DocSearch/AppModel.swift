import Foundation
import SwiftUI

struct ChatBubbleItem: Identifiable {
    enum Role { case user, assistant }
    let id = UUID()
    var role: Role
    var text: String
    var hits: [SearchHit]? = nil
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
    @Published var isSyncing = false
    @Published var isSearching = false
    @Published var syncStatus = "尚未同步"
    @Published var syncFraction = 0.0
    @Published var lastReport: SyncReport?
    @Published var reader: ReaderState?
    @Published var status = ""

    // 缺失模型自动补全
    @Published var missingModels: [RequiredModel] = []
    @Published var isDownloading = false
    @Published var downloadFraction = 0.0
    @Published var downloadStatus = ""

    private var engine: SearchEngine?
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
        checkModels()
        status = "配置已保存；检索前将重新加载模型/数据库"
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

    func send() async {
        let query = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, !isSearching else { return }
        input = ""
        messages.append(ChatBubbleItem(role: .user, text: query))
        isSearching = true
        status = "检索中…"
        defer { isSearching = false }
        do {
            try await ensureEngine()
            guard let engine else { throw DocSearchError.model("检索引擎初始化失败") }
            let q = query
            let topN = config.resultTopN
            let coarse = config.coarseCount
            let useANN = config.useANN
            let nprobe = config.nprobe
            let rescore = config.rescore
            let minScoreRatio = config.effectiveMinScoreRatio
            // 检索含 ONNX 推理与 MaxSim 重排，放到后台线程，避免阻塞 @MainActor 的 UI。
            let hits = try await Task.detached(priority: .userInitiated) {
                try engine.search(query: q, topN: topN, coarse: coarse,
                                  useANN: useANN, nprobe: nprobe, rescore: rescore,
                                  minScoreRatio: minScoreRatio)
            }.value
            messages.append(ChatBubbleItem(
                role: .assistant,
                text: hits.isEmpty ? "没有召回相关分片。" : "召回 \(hits.count) 个分片，点击气泡阅读：",
                hits: hits))
            status = "检索完成：\(hits.count) 个分片"
        } catch {
            messages.append(ChatBubbleItem(role: .assistant, text: "检索失败：\(error.localizedDescription)"))
            status = "检索失败：\(error.localizedDescription)"
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
        defer { isSyncing = false }
        do {
            let report = try await syncService.run(config: config) { info in
                self.syncStatus = "\(info.phase)：\(info.fileName.isEmpty ? info.message : info.fileName)"
                self.syncFraction = info.fraction
            }
            lastReport = report
            syncStatus = report.message
            syncFraction = 1
            status = report.message
            engine = nil
        } catch is CancellationError {
            syncStatus = "同步已停止"
            status = "同步已停止"
        } catch {
            syncStatus = "同步失败：\(error.localizedDescription)"
            status = syncStatus
        }
    }

    func stopSync() { syncService.cancel() }

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
