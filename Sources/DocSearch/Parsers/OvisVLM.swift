import Foundation
import CoreGraphics
import CoreImage
import MLX
import MLXVLM
import MLXLMCommon

// OCR/VLM 支撑层：从主应用裁剪而来，支持三种自定义 OCR 后端：
// - mlx：OvisOCR3（config.json model_type=qwen3_5），走 mlx-swift-lm 已注册的
//   Qwen3.5 VLM 实现；未注册的 Ovis2 架构会明确报错，绝不伪造结果。
// - gguf：llama.cpp mtmd（主模型 + mmproj），走 LlamaCppVLMService。
// - onnx：单一合并 VLM 图（ONNXVLMService）或检测+识别流水线（ONNXOCRService）。

// MARK: - 错误与模型类别

public enum LocalModelKind: String, Sendable {
    case vlm
    public var displayName: String { "VLM（OCR）模型" }
}

public enum LocalModelError: LocalizedError, Sendable {
    case notConfigured(String)
    case notConfiguredKind(LocalModelKind, hint: String)
    case pathNotFound(String)
    case notFound(value: String, searched: [String], hint: String)
    case notFoundKind(LocalModelKind, value: String, searched: [String], hint: String)
    case missingONNXModel(directory: String, expected: [String])
    case missingTokenizer(directory: String, file: String)
    case invalidConfiguration(String)
    case unsupportedArchitecture(String)
    case runtime(String)

    public var errorDescription: String? {
        switch self {
        case .notConfigured(let hint): return "未配置 VLM（OCR）模型：\(hint)"
        case .notConfiguredKind(let kind, let hint): return "未配置\(kind.displayName)：\(hint)"
        case .pathNotFound(let path): return "VLM 模型的显式路径不存在：\(path)"
        case .notFound(let value, let searched, let hint):
            let list = searched.map { "  - \($0)" }.joined(separator: "\n")
            return "找不到 VLM 模型：\(value)\n已检查以下位置：\n\(list)\n\(hint)"
        case .notFoundKind(let kind, let value, let searched, let hint):
            let list = searched.map { "  - \($0)" }.joined(separator: "\n")
            return "找不到\(kind.displayName)：\(value)\n已检查以下位置：\n\(list)\n\(hint)"
        case .missingONNXModel(let directory, let expected):
            return "OCR 模型目录中缺少 ONNX 权重：\(directory)\n期望文件名（按优先级）：\(expected.joined(separator: "、"))"
        case .missingTokenizer(let directory, let file):
            return "模型目录缺少 tokenizer 文件：\(directory)/\(file)"
        case .invalidConfiguration(let message): return "本地模型配置无效：\(message)"
        case .unsupportedArchitecture(let message):
            return "本地模型架构不受当前 mlx-swift-lm 支持：\(message)"
        case .runtime(let message): return "本地模型运行失败：\(message)"
        }
    }
}

// MARK: - 模型路径解析

/// 与本地模型工厂共用的目录发现逻辑（精简版，仅 VLM 用途）。
public enum LocalModelRegistry {
    /// 显式路径（/、~、./、../）之外，优先锚定应用所在目录的 models。
    public static func resolveDirectory(value: String, config: AgentConfig) throws -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let effective: String
        if trimmed.isEmpty {
            effective = "OvisOCR3-MLX-4bit-mixed46"
        } else {
            effective = trimmed
        }

        let fm = FileManager.default
        let expanded = expandPath(effective)

        // 1) 显式路径：路径不存在时不做猜测。
        if isExplicitPath(effective) {
            let url = URL(fileURLWithPath: expanded, isDirectory: true)
            guard fm.fileExists(atPath: url.path) else {
                throw LocalModelError.pathNotFound(expanded)
            }
            return url
        }

        // 2) `models/...` 锚定到可执行文件或 App 所在目录向上查找到的 models。
        if effective.hasPrefix("models/") {
            var roots: [URL] = []
            if let app = applicationModelsDirectory() { roots.append(app) }
            roots.append(configModelsDirectory())
            var searched: [String] = []
            for root in roots {
                let name = String(effective.dropFirst("models/".count))
                let candidate = root.appendingPathComponent(name, isDirectory: true)
                searched.append(candidate.path)
                if fm.fileExists(atPath: candidate.path) { return candidate }
            }
            // 也允许直接以 `models/<name>` 相对当前工作目录存在（便于独立工具调试）。
            let relative = URL(fileURLWithPath: expanded, isDirectory: true)
            if fm.fileExists(atPath: relative.path) { return relative }
            throw LocalModelError.notFound(
                value: effective, searched: searched,
                hint: "请将 OvisOCR3 模型放到可执行文件所在目录向上查找到的 models 目录，或在配置中填写绝对路径。")
        }

        // 3) 相对路径 / 当前工作目录。
        let relative = URL(fileURLWithPath: expanded, isDirectory: true)
        if fm.fileExists(atPath: relative.path) { return relative }

        // 4) 应用 models、配置 models 下按裸名查找。
        var searched: [String] = []
        var roots: [URL] = []
        if let app = applicationModelsDirectory() { roots.append(app) }
        roots.append(configModelsDirectory())
        for root in roots {
            let candidate = root.appendingPathComponent(effective, isDirectory: true)
            searched.append(candidate.path)
            if fm.fileExists(atPath: candidate.path) { return candidate }
        }
        throw LocalModelError.notFound(
            value: effective, searched: searched,
            hint: "请配置 rag_ocr_model_path 为模型目录的绝对路径。")
    }

    /// 读取 config.json 的 `model_type`（部分模型只有 `architectures`）。
    public static func modelType(in directory: URL) -> String? {
        guard let config = readJSON(directory.appendingPathComponent("config.json")) else {
            return nil
        }
        if let type = config["model_type"] as? String, !type.isEmpty { return type }
        if let architectures = config["architectures"] as? [String],
           let first = architectures.first { return first }
        return nil
    }

    /// 判断是否为 Ovis2 架构（mlx-swift-lm 未注册）。
    public static func isOvisArchitecture(directory: URL) -> Bool {
        guard let type = modelType(in: directory)?.lowercased() else { return false }
        return type.contains("ovis")
    }

    /// 从可执行文件或 App 包所在目录向上查找 models（受 BASE_AGENT_MODELS_DIR 覆盖）。
    public static func applicationModelsDirectory() -> URL? {
        let fm = FileManager.default
        if let override = ProcessInfo.processInfo.environment["BASE_AGENT_MODELS_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let url = URL(fileURLWithPath: expandPath(override), isDirectory: true)
            if fm.fileExists(atPath: url.path) { return url }
        }
        var directory = Bundle.main.bundleURL.standardizedFileURL
        if directory.pathExtension.lowercased() == "app" {
            directory = directory.deletingLastPathComponent()
        }
        for _ in 0..<8 {
            let candidate = directory.appendingPathComponent("models", isDirectory: true)
            if fm.fileExists(atPath: candidate.path) { return candidate }
            let parent = directory.deletingLastPathComponent()
            if parent.path == directory.path { break }
            directory = parent
        }
        return nil
    }

    public static func configModelsDirectory() -> URL {
        URL(fileURLWithPath: expandPath("~/.base-agent/models"), isDirectory: true)
    }

    // MARK: - ONNX 专用解析

    /// 解析 ONNX 目录：显式 `.onnx` 文件原样返回，否则回落到 `resolveDirectory`。
    public static func resolveONNXDirectory(
        value: String, config: AgentConfig
    ) throws -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, trimmed.lowercased().hasSuffix(".onnx") {
            let url = URL(fileURLWithPath: expandPath(trimmed))
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
               !isDir.boolValue {
                return url
            }
        }
        return try resolveDirectory(value: value, config: config)
    }

    /// 按约定顺序定位 ONNX 权重文件（量化优先，当前目录 → onnx/ 子目录）。
    public static func resolveONNXModel(in directory: URL) throws -> URL {
        let fm = FileManager.default
        if directory.pathExtension.lowercased() == "onnx",
           fm.fileExists(atPath: directory.path) {
            return directory
        }
        guard fm.fileExists(atPath: directory.path) else {
            throw LocalModelError.missingONNXModel(
                directory: directory.path, expected: onnxCandidateNames)
        }
        for relative in onnxCandidateNames {
            let url = directory.appendingPathComponent(relative)
            if fm.fileExists(atPath: url.path) { return url }
        }
        throw LocalModelError.missingONNXModel(
            directory: directory.path, expected: onnxCandidateNames)
    }

    /// ONNX 权重的 tokenizer/config 目录：权重可能在 onnx/ 子目录而配置在根目录。
    public static func onnxSupportDirectory(modelURL: URL, modelRoot: URL) -> URL {
        let fm = FileManager.default
        var root = modelRoot
        var isDirectory: ObjCBool = false
        if fm.fileExists(atPath: root.path, isDirectory: &isDirectory), !isDirectory.boolValue {
            root = root.deletingLastPathComponent()
        }
        let weightDir = modelURL.deletingLastPathComponent()
        for candidate in [weightDir, root] {
            if fm.fileExists(atPath: candidate.appendingPathComponent("tokenizer.json").path) {
                return candidate
            }
        }
        return root
    }

    public static let onnxCandidateNames: [String] = [
        "model_quantized.onnx", "model_int8.onnx", "model_int4.onnx", "model.onnx",
        "onnx/model_quantized.onnx", "onnx/model_int8.onnx",
        "onnx/model_int4.onnx", "onnx/model.onnx",
    ]

    private static func isExplicitPath(_ value: String) -> Bool {
        value.hasPrefix("/") || value.hasPrefix("~")
            || value.hasPrefix("./") || value.hasPrefix("../")
    }

    private static func expandPath(_ value: String) -> String {
        (value as NSString).expandingTildeInPath
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else { return nil }
        return dictionary
    }
}

// MARK: - 串行闸门

/// 保证同一 MLX VLM 实例同一时刻只处理一页 OCR。
final class AsyncGate: @unchecked Sendable {
    private let lock = NSLock()
    private var acquired = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        lock.lock()
        if !acquired {
            acquired = true
            lock.unlock()
            return
        }
        await withCheckedContinuation { continuation in
            lock.lock()
            if !acquired {
                acquired = true
                lock.unlock()
                continuation.resume()
                return
            }
            waiters.append(continuation)
            lock.unlock()
        }
    }

    func release() {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.resume()
        } else {
            acquired = false
            lock.unlock()
        }
    }

    func withLock<T>(_ body: () async throws -> T) async rethrows -> T {
        await acquire()
        do {
            let value = try await body()
            release()
            return value
        } catch {
            release()
            throw error
        }
    }
}

// MARK: - OvisOCR3 VLM 服务

/// OvisOCR3 的 VLM 服务（MLX 后端 OvisOCR3 默认）。
///
/// OvisOCR3 的 `config.json` 使用 `model_type=qwen3_5` /
/// `Qwen3_5ForConditionalGeneration`，可直接走 `mlx-swift-lm` 已注册的
/// Qwen3.5 VLM 实现。仅当模型确实是未注册的 Ovis2 架构时 `isAvailable`
/// 返回 false 并在 `recognize` 抛明确错误，绝不伪造结果。
public final class OvisVLMService: VLMService, @unchecked Sendable {
    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let modelType: String?
    private let preflightError: Error?
    private let state = VLMLoadState()
    private let gate = AsyncGate()

    public var isAvailable: Bool { preflightError == nil }

    public init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-vlm:\(modelName)"

        let type = LocalModelRegistry.modelType(in: directory)
        self.modelType = type

        if LocalModelRegistry.isOvisArchitecture(directory: directory) {
            preflightError = LocalModelError.unsupportedArchitecture(
                "检测到未注册的 Ovis2 架构（model_type=\(type ?? "unknown")，模型：\(modelName)）。"
                    + "mlx-swift-lm 未提供该架构实现，无法加载 \(directory.path)。"
                    + "OvisOCR3 请使用 model_type=qwen3_5 的 MLX 转换版本。"
            )
        } else if type == nil {
            preflightError = LocalModelError.invalidConfiguration(
                "无法从 \(directory.path)/config.json 读取 model_type 或 architectures"
            )
        } else {
            preflightError = nil
        }
    }

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        try await gate.withLock {
            try await recognizeLocked(image: image, prompt: prompt, maxTokens: maxTokens)
        }
    }

    private func recognizeLocked(image: CGImage, prompt: String,
                                 maxTokens: Int) async throws -> String {
        if let preflightError { throw preflightError }
        guard let modelType else {
            throw LocalModelError.invalidConfiguration("VLM 模型缺少 model_type：\(directory.path)")
        }

        let supported = await VLMTypeRegistry.shared.contains(modelType)
        guard supported else {
            throw LocalModelError.unsupportedArchitecture(
                "model_type=\(modelType) 未在 mlx-swift-lm 的 VLMTypeRegistry 中注册"
                    + "（模型：\(modelName)，目录：\(directory.path)）"
            )
        }

        let container: ModelContainer
        do {
            container = try await state.container(directory: directory)
        } catch let error as LocalModelError {
            throw error
        } catch {
            throw LocalModelError.runtime(
                "VLMModelFactory 加载 \(modelName) 失败：\(error.localizedDescription)"
            )
        }

        let userInput: UserInput = autoreleasepool {
            let ciImage = CIImage(cgImage: image)
            return UserInput(chat: [.user(prompt, images: [.ciImage(ciImage)])])
        }

        do {
            let input = try await container.prepare(input: userInput)
            let parameters = GenerateParameters(
                maxTokens: max(1, maxTokens),
                temperature: 0,
                topP: 1.0,
                topK: 0,
                minP: 0.0,
                repetitionPenalty: nil
            )
            let stream = try await container.generate(input: input, parameters: parameters)

            var text = ""
            for await generation in stream {
                try Task.checkCancellation()
                switch generation {
                case .chunk(let chunk): text += chunk
                case .info(_), .toolCall(_), .rejectedToolCall(_): break
                }
            }
            return Self.stripThinking(text).trimmingCharacters(in: .whitespacesAndNewlines)
        } catch is CancellationError {
            throw AgentError.cancelled
        } catch let error as LocalModelError {
            throw error
        } catch {
            throw LocalModelError.runtime(
                "\(modelName) OCR 推理失败：\(error.localizedDescription)"
            )
        }
    }

    public func unload() async {
        await state.unload()
    }

    private static func stripThinking(_ text: String) -> String {
        var result = text
        let pattern = "<think>[\\s\\S]*?</think>"
        let stripped = autoreleasepool {
            if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
                let range = NSRange(result.startIndex..<result.endIndex, in: result)
                return regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
            }
            return result
        }
        result = stripped
            .replacingOccurrences(of: "<think>", with: "")
            .replacingOccurrences(of: "</think>", with: "")
        return result
    }
}

/// VLM 容器只加载一次；失败后缓存错误，避免重复加载数 GB 权重。
actor VLMLoadState {
    private var container: ModelContainer?
    private var failure: Error?

    func container(directory: URL) async throws -> ModelContainer {
        if let container { return container }
        if let failure { throw failure }
        do {
            let container = try await VLMModelFactory.shared.loadContainer(
                from: directory, using: OvisTokenizerLoader())
            self.container = container
            return container
        } catch {
            self.failure = error
            throw error
        }
    }

    func unload() {
        container = nil
        failure = nil
    }
}

// MARK: - 明确抛错的占位实现

private final class FailingVLMService: VLMService, @unchecked Sendable {
    let identifier: String
    let error: Error
    var isAvailable: Bool { false }

    init(identifier: String, error: Error) {
        self.identifier = identifier
        self.error = error
    }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        throw error
    }
}

// MARK: - 多实例池

/// OvisOCR3 多实例池：实例独立懒加载、各自串行，池级借还并行。
public final class VLMServicePool: VLMService, @unchecked Sendable {
    public let identifier: String
    public var isAvailable: Bool { services.allSatisfy { $0.isAvailable } }

    private let services: [VLMService]
    private let gate: AsyncSemaphore
    private let lock = NSLock()
    private var idle: [VLMService]

    init(size: Int, factory: (Int) -> VLMService) {
        let built = (0..<max(1, size)).map(factory)
        self.services = built
        self.identifier = "vlm-pool(\(built.count)):\(built.first?.identifier ?? "empty")"
        self.gate = AsyncSemaphore(value: built.count)
        self.idle = built
    }

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        try await gate.withPermit {
            lock.lock()
            let service = idle.removeLast()
            lock.unlock()
            defer {
                lock.lock()
                idle.append(service)
                lock.unlock()
            }
            return try await service.recognize(image: image, prompt: prompt, maxTokens: maxTokens)
        }
    }

    /// 必须覆写：否则 endOCRJob 时 unloadOCRWeights 对池无效，权重常驻。
    public func unload() async {
        for service in services { await service.unload() }
    }
}

// MARK: - 缓存

private final class ModelCache<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: Value] = [:]

    func value(for key: String, build: () -> Value) -> Value {
        lock.lock()
        if let existing = storage[key] { lock.unlock(); return existing }
        lock.unlock()
        let built = build()
        lock.lock()
        if let existing = storage[key] { lock.unlock(); return existing }
        storage[key] = built
        lock.unlock()
        return built
    }

    func clear() {
        lock.lock()
        storage.removeAll()
        lock.unlock()
    }

    func allValues() -> [Value] {
        lock.lock()
        defer { lock.unlock() }
        return Array(storage.values)
    }
}

// MARK: - OCR 生命周期

/// 引用计数：最后一个在途任务结束时才释放权重；卸载期间阻塞新任务。
public actor OCRModelLifecycle {
    private var activeJobs = 0
    private var isReleasing = false
    private var beginWaiters: [CheckedContinuation<Void, Never>] = []
    private let unload: @Sendable () async -> Void

    public init(unload: @escaping @Sendable () async -> Void) {
        self.unload = unload
    }

    public func begin() async {
        while isReleasing {
            await withCheckedContinuation { beginWaiters.append($0) }
        }
        activeJobs += 1
    }

    public func end() async {
        guard activeJobs > 0 else { return }
        activeJobs -= 1
        guard activeJobs == 0 else { return }
        await releaseWeights()
    }

    public func unloadIfIdle() async {
        guard activeJobs == 0, !isReleasing else { return }
        await releaseWeights()
    }

    private func releaseWeights() async {
        guard !isReleasing else { return }
        isReleasing = true
        await unload()
        isReleasing = false
        let waiters = beginWaiters
        beginWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}

// MARK: - 本地模型工厂

/// RAG OCR 专用本地模型工厂（仅 MLX VLM 分支）。
public final class LocalModelFactory: @unchecked Sendable {
    public static let shared = LocalModelFactory()

    private let vlms = ModelCache<VLMService>()
    private let ocrLifecycle = OCRModelLifecycle {
        await LocalModelFactory.shared.unloadOCRWeights()
    }

    private init() {
        Self.configureMetallibPath()
        // 与参考实现一致：缓存 1024MB，总内存按 65% 物理内存自动取值。
        setCacheLimitMB(1024)
        setMemoryLimitMB(0)
    }

    /// 打包成 .app 后显式指定 metallib 路径；裸可执行文件由 MLX 同目录查找。
    private static func configureMetallibPath() {
        let relative = "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        let candidates: [String] = [
            Bundle.main.resourceURL?.appendingPathComponent(relative).path,
            Bundle.main.bundleURL.appendingPathComponent(relative).path,
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("Resources/\(relative)").path,
        ].compactMap { $0 }
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            MLX.GPU.metallib = URL(fileURLWithPath: path)
            AgentLog.info("[metal] metallib=\(path)")
            return
        }
    }

    public func setCacheLimitMB(_ megabytes: Int) {
        MLX.Memory.cacheLimit = max(256, min(2_048, megabytes)) * 1_024 * 1_024
    }

    public func setMemoryLimitMB(_ megabytes: Int) {
        let physicalMB = Int(ProcessInfo.processInfo.physicalMemory / 1_024 / 1_024)
        let automatic = max(1_024, min(physicalMB * 65 / 100, 9_830))
        let requested = megabytes > 0 ? megabytes : automatic
        let clamped = max(1_024, min(requested, max(1_024, physicalMB - 2_048)))
        MLX.Memory.memoryLimit = clamped * 1_024 * 1_024
    }

    /// 释放全部 OCR VLM 权重，provider 保留，下次识别按需重载。
    public func unloadOCRWeights() async {
        let values = vlms.allValues()
        for service in values { await service.unload() }
        let activeMB = MLX.Memory.activeMemory / 1_024 / 1_024
        let cacheMB = MLX.Memory.cacheMemory / 1_024 / 1_024
        AgentLog.info("[memory] 释放 OCR \(values.count) 个模型，"
            + "MLX active=\(activeMB)MB cache=\(cacheMB)MB")
    }

    public func unloadIdleOCRWeights() async {
        await ocrLifecycle.unloadIfIdle()
    }

    public func beginOCRJob() async {
        await ocrLifecycle.begin()
    }

    public func endOCRJob() async {
        await ocrLifecycle.end()
    }

    /// OCR VLM：按 `ragOvisInstances` 返回单实例或多实例池。
    public func makeRAGOCRVLM(config: AgentConfig) -> VLMService {
        let instances = max(1, min(2, config.ragOvisInstances))
        guard instances > 1 else { return makeOCRModel(config: config) }
        return VLMServicePool(size: instances) { index in
            LocalModelFactory.shared.makeOCRModel(config: config, instance: index)
        }
    }

    /// 按后端分派：mlx（OvisOCR3）/ gguf（llama.cpp mtmd）/
    /// onnx（合并 VLM 图或检测+识别 OCR 流水线）。自动按模型路径后缀识别。
    public func makeOCRModel(config: AgentConfig, instance: Int = 0) -> VLMService {
        let backend = config.localVLMBackend.lowercased()
        guard backend == "mlx" || backend == "onnx" || backend == "gguf" || backend.isEmpty else {
            return FailingVLMService(
                identifier: "local:\(backend)-vlm:unsupported",
                error: LocalModelError.invalidConfiguration(
                    "doc_search 的 OCR VLM 支持 mlx / onnx 后端，当前配置为 \(config.localVLMBackend)")
            )
        }

        let value: String
        if !config.ragOCRModelPath.isEmpty {
            value = config.ragOCRModelPath
        } else if !config.localVLMModel.isEmpty {
            value = config.localVLMModel
        } else {
            value = config.ragOCRModelName
        }
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return FailingVLMService(
                identifier: "local:vlm:unconfigured",
                error: LocalModelError.invalidConfiguration("未配置 rag_ocr_model_path / rag_ocr_model_name")
            )
        }

        let expanded = (value as NSString).expandingTildeInPath
        let looksGGUF = backend == "gguf"
            || value.lowercased().hasSuffix(".gguf")
            || GGUFFileResolver.ggufFile(atPath: expanded) != nil
        let looksONNX = !looksGGUF && (backend == "onnx"
            || value.lowercased().hasSuffix(".onnx")
            || (try? LocalModelRegistry.resolveONNXModel(in: URL(fileURLWithPath: expanded))) != nil)

        let key = "vlm:\(backend):\(value)#\(instance)"
        return vlms.value(for: key) {
            do {
                if looksGGUF {
                    let file = try GGUFFileResolver.resolve(
                        kind: .vlm, value: value, config: config)
                    let mmproj = GGUFFileResolver.resolveMMProj(
                        modelPath: file.path,
                        explicit: config.ragOCRMMProjPath.isEmpty
                            ? nil : config.ragOCRMMProjPath)
                    return LlamaCppVLMService(
                        modelPath: file.path, mmprojPath: mmproj,
                        gpuLayers: config.llamaGpuLayers)
                }
                if looksONNX {
                    let directory = try LocalModelRegistry.resolveONNXDirectory(
                        value: value, config: config)
                    // 检测+识别多阶段 OCR 包优先；单一合并 VLM 图作为后备。
                    if ONNXOCRService.isONNXOCRDirectory(directory) {
                        return ONNXOCRService(modelDirectory: directory, modelName: value)
                    }
                    let tokenizerPath = config.ragOCRTokenizerPath
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let tokenizerDirectory = tokenizerPath.isEmpty
                        ? nil : URL(fileURLWithPath: tokenizerPath)
                    return ONNXVLMService(
                        modelLocation: directory,
                        tokenizerDirectory: tokenizerDirectory,
                        modelName: value,
                        contextWindow: config.effectiveMainContextTokens,
                        defaultMaxTokens: 2_048)
                }
                let directory = try LocalModelRegistry.resolveDirectory(value: value, config: config)
                return OvisVLMService(directory: directory, modelName: value)
            } catch {
                AgentLog.warning("本地 VLM 不可用：\(error.localizedDescription)")
                return FailingVLMService(
                    identifier: "local:\(backend)-vlm:\(value)", error: error)
            }
        }
    }
}
