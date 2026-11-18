import Foundation
import CoreGraphics
import Darwin

// 主应用 AgentConfig / AgentError / AgentLog / PDFEngine 的最小裁剪版。
// 仅覆盖文档解析与 Markdown 转换服务使用的字段，避免拖入 LLM/MLX 依赖。

public struct Chunk: Sendable, Identifiable {
    public var id: String
    public var text: String
    public var source: String
    public var chunkIndex: Int
    public var heading: String
    public var metadata: [String: String]

    public init(text: String, source: String, chunkIndex: Int,
                heading: String = "", metadata: [String: String] = [:]) {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in source.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100000001b3 }
        self.id = String(format: "%08llx_%d", hash & 0xffff_ffff, chunkIndex)
        self.text = text
        self.source = source
        self.chunkIndex = chunkIndex
        self.heading = heading
        self.metadata = metadata
    }
}

public struct ParsedDocument: Sendable {
    public var text: String
    public var source: String
    public var outline: String
    public var metadata: [String: String]
    public var pageCount: Int

    public init(text: String, source: String, outline: String = "",
                metadata: [String: String] = [:], pageCount: Int = 0) {
        self.text = text
        self.source = source
        self.outline = outline
        self.metadata = metadata
        self.pageCount = pageCount
    }
}

final class AsyncSemaphore: @unchecked Sendable {
    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Bool, Never>
    }

    private let lock = NSLock()
    private var available: Int
    private var waiters: [Waiter] = []
    private var nextID: UInt64 = 0

    init(value: Int) { available = max(1, value) }

    func acquire() async -> Bool {
        lock.lock()
        if Task.isCancelled { lock.unlock(); return false }
        if available > 0 { available -= 1; lock.unlock(); return true }
        let id = nextID
        nextID &+= 1
        lock.unlock()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.lock()
                if Task.isCancelled { lock.unlock(); continuation.resume(returning: false); return }
                if available > 0 {
                    available -= 1
                    lock.unlock()
                    continuation.resume(returning: true)
                    return
                }
                waiters.append(Waiter(id: id, continuation: continuation))
                lock.unlock()
            }
        } onCancel: {
            lock.lock()
            if let index = waiters.firstIndex(where: { $0.id == id }) {
                let waiter = waiters.remove(at: index)
                lock.unlock()
                waiter.continuation.resume(returning: false)
            } else {
                lock.unlock()
            }
        }
    }

    func release() {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.continuation.resume(returning: true)
        } else {
            available += 1
            lock.unlock()
        }
    }

    func withPermit<T>(_ body: () async throws -> T) async throws -> T {
        guard await acquire() else { throw AgentError.cancelled }
        defer { release() }
        return try await body()
    }
}

enum ProcessMemory {
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}

public enum PDFEngine: String, Codable, Sendable, CaseIterable {
    case ovis, vision, hybrid, custom

    init(stored: String) {
        switch stored.lowercased() {
        case "vision", "pdfkit_vision", "apple_vision": self = .vision
        case "ovis", "ovisocr3": self = .ovis
        case "hybrid", "mixed", "auto": self = .hybrid
        case "custom", "custom_vlm": self = .custom
        default: self = .vision
        }
    }
}

public struct AgentConfig {
    var ragPDFEngine: PDFEngine = .vision
    var ragEPUBOCREngine: PDFEngine = .vision
    var ragOCRPreprocess = true
    var ragOCRBinarize = true
    var ragOCREngine = "mlx-vlm"
    var ragOCRModelName = "OvisOCR3-MLX-4bit-mixed46"
    var ragOCRModelPath = "models/OvisOCR3-MLX-4bit-mixed46"
    var ragOCRMMProjPath = ""
    var ragOCRTokenizerPath = ""
    /// OvisOCR3 并行实例数（1...2）；2 时返回双实例池，池级并行。
    var ragOvisInstances = 2
    /// 本地 VLM 后端；doc_search 仅实现 MLX（OvisOCR3 默认）。
    var localVLMBackend = "mlx"
    var localVLMModel = ""
    var llamaGpuLayers = 99
    var effectiveMainContextTokens = 8192
    /// GGUF 模型搜索根目录（workspace/models）。
    var modelsDirectory = ""
    var ragIngestConcurrency = 3
    var ragChunkSize = 1400
    var ragChunkOverlap = 180
    var ragSemanticChunking = true
}

public enum AgentError: LocalizedError {
    case cancelled
    case parse(String)
    case unsupported(String)
    case storage(String)
    case internalError(String)
    case timeout(String)

    public var errorDescription: String? {
        switch self {
        case .cancelled: return "已取消"
        case .parse(let s): return "解析错误: \(s)"
        case .unsupported(let s): return "不支持: \(s)"
        case .storage(let s): return "存储错误: \(s)"
        case .internalError(let s): return "内部错误: \(s)"
        case .timeout(let s): return "超时: \(s)"
        }
    }
}

enum AgentLog {
    static func info(_ message: String) { FileHandle.standardError.write(Data(("[info] " + message + "\n").utf8)) }
    static func warning(_ message: String) { FileHandle.standardError.write(Data(("[warn] " + message + "\n").utf8)) }
}

enum RAGEngine {
    static let supportedExtensions: Set<String> = [
        "pdf", "txt", "md", "markdown", "html", "htm", "xhtml", "epub",
        "docx", "xlsx", "pptx", "doc", "xls", "ppt", "csv", "json",
        "rst", "tex", "log", "png", "jpg", "jpeg", "tiff", "tif",
        "bmp", "heic", "heif", "webp",
    ]
}
