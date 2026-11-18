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
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("配置…") {
                    NotificationCenter.default.post(name: .docSearchOpenSettings, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
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
