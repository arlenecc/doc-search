import Foundation

/// 全局日志缓冲区：GUI 检索、同步与计时的输出都会汇总到这里，日志窗口实时展示。
final class AppLog: ObservableObject {
    static let shared = AppLog()

    struct Entry: Identifiable {
        let id = UUID()
        let time = Date()
        let message: String
    }

    @Published private(set) var entries: [Entry] = []

    private let lock = NSLock()
    private var pending: [Entry] = []
    private var scheduled = false
    private let maxEntries = 2000

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    func log(_ message: String) {
        let entry = Entry(message: message)
        lock.lock()
        pending.append(entry)
        let needsSchedule = !scheduled
        scheduled = true
        lock.unlock()
        if needsSchedule {
            DispatchQueue.main.async { [weak self] in self?.flush() }
        }
    }

    func clear() {
        lock.lock()
        pending.removeAll()
        lock.unlock()
        DispatchQueue.main.async { [weak self] in self?.entries.removeAll() }
    }

    var plainText: String {
        entries.map { "\(Self.formatter.string(from: $0.time))  \($0.message)" }
            .joined(separator: "\n")
    }

    func formatted(_ entry: Entry) -> String {
        "\(Self.formatter.string(from: entry.time))  \(entry.message)"
    }

    private func flush() {
        lock.lock()
        let batch = pending
        pending.removeAll()
        scheduled = false
        lock.unlock()
        guard !batch.isEmpty else { return }
        entries.append(contentsOf: batch)
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
    }
}
