import SwiftUI
import AppKit

struct LogWindowView: View {
    @ObservedObject private var log = AppLog.shared
    @State private var autoScroll = true

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("日志").font(.headline)
                Spacer()
                Toggle("自动滚动", isOn: $autoScroll)
                    .toggleStyle(.checkbox)
                    .font(.caption)
                Button("复制全部") { copyAll() }
                Button("清空") { log.clear() }
            }
            .padding(10)

            Divider()

            ScrollViewReader { proxy in
                List(log.entries) { entry in
                    Text(log.formatted(entry))
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .id(entry.id)
                }
                .listStyle(.plain)
                .onChange(of: log.entries.count) { _, _ in
                    guard autoScroll, let last = log.entries.last else { return }
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
        .frame(minWidth: 760, minHeight: 480)
    }

    private func copyAll() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(log.plainText, forType: .string)
    }
}
