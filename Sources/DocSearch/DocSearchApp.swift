import SwiftUI
import AppKit

@main
struct DocSearchApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 980, minHeight: 640)
        }
        Window("日志", id: "docsearch-log") {
            LogWindowView()
        }
        .defaultSize(width: 900, height: 620)
        Window("Markdown 文件", id: "docsearch-files") {
            MarkdownFilesView()
                .environmentObject(model)
        }
        .defaultSize(width: 760, height: 560)
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("配置…") {
                    NotificationCenter.default.post(name: .docSearchOpenSettings, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
                Button("日志") {
                    NotificationCenter.default.post(name: .docSearchOpenLog, object: nil)
                }
                .keyboardShortcut("l", modifiers: [.command, .shift])
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        let args = Array(CommandLine.arguments.dropFirst())
        guard DocSearchCLI.isCLIInvocation(args) else { return }
        // 保留原 CLI：有命令参数时只跑 CLI，不展示 GUI。
        NSApp.setActivationPolicy(.accessory)
        Task {
            do {
                try await DocSearchCLI.run(args)
            } catch {
                FileHandle.standardError.write(Data(("error: \(error.localizedDescription)\n").utf8))
                exit(1)
            }
            exit(0)
        }
    }
}
