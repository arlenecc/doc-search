import SwiftUI
import AppKit

private struct MarkdownFile: Identifiable {
    let url: URL
    let relativePath: String
    let size: Int64
    let modified: Date
    var id: String { url.path }
    var name: String { url.lastPathComponent }
}

/// 独立窗口：浏览工作目录下 markdown/ 中转换生成的文件。
struct MarkdownFilesView: View {
    @EnvironmentObject var model: AppModel
    @State private var files: [MarkdownFile] = []
    @State private var search = ""
    @State private var sortByDate = true

    private var directory: String { model.config.outputDirectory }

    private var filteredFiles: [MarkdownFile] {
        let q = search.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let list = q.isEmpty ? files : files.filter {
            $0.name.lowercased().contains(q) || $0.relativePath.lowercased().contains(q)
        }
        return sortByDate
            ? list.sorted { $0.modified > $1.modified }
            : list.sorted { $0.relativePath.localizedCaseInsensitiveCompare($1.relativePath) == .orderedAscending }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Markdown：\(directory)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button("刷新") { load() }
                Button("在 Finder 中显示") { revealDirectory() }
                    .disabled(!FileManager.default.fileExists(atPath: directory))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            HStack {
                TextField("搜索文件名或路径", text: $search)
                    .textFieldStyle(.roundedBorder)
                Picker("排序", selection: $sortByDate) {
                    Text("修改时间").tag(true)
                    Text("名称").tag(false)
                }
                .pickerStyle(.segmented)
                .frame(width: 170)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 6)

            List {
                if filteredFiles.isEmpty {
                    Text(emptyMessage)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 120, alignment: .center)
                } else {
                    ForEach(filteredFiles) { file in
                        HStack(spacing: 8) {
                            Image(systemName: "doc.text")
                                .foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(file.name)
                                Text(file.relativePath)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer()
                            Text(formattedSize(file.size))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(formattedDate(file.modified))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) { NSWorkspace.shared.open(file.url) }
                        .contextMenu {
                            Button("打开") { NSWorkspace.shared.open(file.url) }
                            Button("在 Finder 中显示") {
                                NSWorkspace.shared.activateFileViewerSelecting([file.url])
                            }
                        }
                    }
                }
            }
            .listStyle(.inset)

            HStack {
                Text("\(filteredFiles.count) 个 Markdown 文件（双击打开）")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .frame(minWidth: 640, minHeight: 460)
        .onAppear { load() }
        .onReceive(model.$lastReport) { _ in load() }
    }

    private var emptyMessage: String {
        if !FileManager.default.fileExists(atPath: directory) {
            return "Markdown 目录不存在：\(directory)\n请先同步知识或检查工作目录配置。"
        }
        return files.isEmpty ? "该目录下暂无 Markdown 文件。" : "没有匹配的文件。"
    }

    private func load() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory),
              let subpaths = fm.subpaths(atPath: directory) else {
            files = []
            return
        }
        var result: [MarkdownFile] = []
        for subpath in subpaths {
            guard subpath.lowercased().hasSuffix(".md") else { continue }
            let full = (directory as NSString).appendingPathComponent(subpath)
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: full, isDirectory: &isDirectory), !isDirectory.boolValue else { continue }
            let attributes = try? fm.attributesOfItem(atPath: full)
            result.append(MarkdownFile(
                url: URL(fileURLWithPath: full),
                relativePath: subpath,
                size: (attributes?[.size] as? NSNumber)?.int64Value ?? 0,
                modified: (attributes?[.modificationDate] as? Date) ?? Date.distantPast))
        }
        files = result
    }

    private func revealDirectory() {
        let url = URL(fileURLWithPath: directory, isDirectory: true)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func formattedSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}
