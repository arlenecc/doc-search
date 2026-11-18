import SwiftUI

extension Notification.Name {
    static let docSearchOpenSettings = Notification.Name("docSearch.openSettings")
    static let docSearchOpenLog = Notification.Name("docSearch.openLog")
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.openWindow) private var openWindow
    @State private var showSettings = false
    @State private var mainWidth: CGFloat = 900

    var body: some View {
        NavigationSplitView {
            List {
                Section("工作区") {
                    Label("对话检索", systemImage: "bubble.left.and.bubble.right")
                        .foregroundColor(.accentColor)
                    Picker("检索方式", selection: Binding(
                        get: { model.searchMode },
                        set: { model.setSearchMode($0) }
                    )) {
                        Text("向量检索").tag("vector")
                        Text("决策判定").tag("decision")
                    }
                    .pickerStyle(.segmented)
                    .help("向量检索：ColBERT 向量召回；决策判定：BM25 召回 + 决策模型判定")
                }
                Section("知识库") {
                    Text(model.config.sourceDirectory.isEmpty
                         ? "源目录：未设置" : "源目录：\(model.config.sourceDirectory)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    Text(model.syncStatus)
                        .font(.caption)
                    ProgressView(value: model.syncFraction)
                    HStack {
                        Button("同步知识") { Task { await model.sync() } }
                            .disabled(model.isSyncing)
                        Button("停止") { model.stopSync() }
                            .disabled(!model.isSyncing)
                    }
                    if let report = model.lastReport {
                        Text("文件 \(report.succeeded)/\(report.files) · 切片 \(report.chunks) · 跳过 \(report.skipped)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    if model.config.useVectorSearch {
                        if model.isBackfilling || model.backfillTotal > 0 {
                            HStack {
                                Text("词向量后台填充 \(model.backfillDone)/\(max(1, model.backfillTotal))")
                                    .font(.caption2)
                                Spacer()
                                Button("停止") { model.stopVectorBackfill() }
                                    .font(.caption2)
                                    .disabled(!model.isBackfilling)
                            }
                            ProgressView(value: model.backfillTotal > 0
                                ? Double(model.backfillDone) / Double(model.backfillTotal) : 0)
                        }
                    } else {
                        Text("向量检索未启用（仅文本切片，决策判定可直接用）")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Section("模型") {
                    if model.missingModels.isEmpty {
                        Label("模型已就绪", systemImage: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.green)
                    } else {
                        ForEach(model.missingModels) { item in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.kind).font(.caption)
                                Text(item.repo).font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                        HStack {
                            Button("下载补全") { Task { await model.downloadMissingModels() } }
                                .disabled(model.isDownloading)
                            Button("停止") { model.cancelDownload() }
                                .disabled(!model.isDownloading)
                        }
                    }
                    if model.isDownloading || !model.downloadStatus.isEmpty {
                        Text(model.downloadStatus).font(.caption2).foregroundStyle(.secondary)
                        ProgressView(value: model.downloadFraction)
                    }
                    Button("重新检测") { model.checkModels() }
                        .font(.caption)
                }
                Section {
                    Button("配置…") { showSettings = true }
                }
            }
            .navigationTitle("Doc Search")
            .listStyle(.sidebar)
            .frame(minWidth: 260)
        } detail: {
            ChatView()
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            openWindow(id: "docsearch-log")
                        } label: {
                            Label("日志", systemImage: "doc.text.magnifyingglass")
                        }
                        .help("打开实时日志窗口")
                    }
                }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(config: model.config) { newConfig in
                model.updateConfig(newConfig)
            }
            .frame(minWidth: 680, minHeight: 620)
        }
        .background(
            GeometryReader { geo in
                Color.clear
                    .onAppear { mainWidth = geo.size.width }
                    .onChange(of: geo.size.width) { _, new in mainWidth = new }
            }
        )
        .sheet(item: $model.reader) { state in
            ChunkReaderView(state: state)
                .environmentObject(model)
                .frame(width: max(560, mainWidth * 0.8), height: 620)
        }
        .onReceive(NotificationCenter.default.publisher(for: .docSearchOpenSettings)) { _ in
            showSettings = true
        }
        .onReceive(NotificationCenter.default.publisher(for: .docSearchOpenLog)) { _ in
            openWindow(id: "docsearch-log")
        }
    }
}
