import Foundation

// MARK: - HF 镜像模型下载

/// 从 HuggingFace 兼容镜像（默认 hf-mirror.com）按需拉取模型文件。
///
/// - 文件列表走 REST API：`GET /api/models/{repo}/tree/main?recursive=true`；
/// - 单文件下载：`GET /{repo}/resolve/main/{path}`，HTTP `Range` 请求配合
///   `.part` 临时文件实现断点续传（重启应用后从已下载偏移继续）；
/// - 只拉取推理运行时会读取的文件（权重 / 配置 / tokenizer），忽略 README、
///   示例脚本等无关内容。
public final class HFModelDownloader: @unchecked Sendable {
    public struct FileEntry: Sendable, Equatable {
        public let path: String
        public let size: Int64
        /// 远程仓库内路径（含子目录前缀），用于拼接下载 URL。
        /// 子目录模型（如 potion 的 `int4/`）本地落点需剥掉前缀，但远程 URL
        /// 必须保留前缀，否则会指向仓库根下不存在的文件而 404。为空时等同 `path`。
        public let remotePath: String

        public init(path: String, size: Int64, remotePath: String? = nil) {
            self.path = path
            self.size = size
            self.remotePath = remotePath ?? path
        }
    }

    public enum HFError: LocalizedError {
        case badResponse(String)
        case httpStatus(Int)

        public var errorDescription: String? {
            switch self {
            case .badResponse(let message): return message
            case .httpStatus(let code): return "下载请求失败（HTTP \(code)）"
            }
        }
    }

    /// 默认镜像：国内网络可直接访问的 HuggingFace 反代。
    public static let defaultMirrorBase = "https://hf-mirror.com"
    /// 默认模型统一托管在 arlenecc/<模型名> 仓库。
    public static let repoOwner = "arlenecc"

    private let baseURL: URL
    private let session: URLSession
    private let lock = NSLock()
    private var activeTask: URLSessionTask?

    /// `HF_MIRROR_BASE` 环境变量可覆盖镜像地址（测试/内网镜像）。
    public init(base: String = HFModelDownloader.defaultMirrorBase) {
        let envBase = ProcessInfo.processInfo.environment["HF_MIRROR_BASE"] ?? ""
        self.baseURL = URL(string: envBase.isEmpty ? base : envBase)
            ?? URL(string: base)!
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        // 大模型下载可达数 GB，资源级超时必须放开。
        cfg.timeoutIntervalForResource = 7 * 24 * 3_600
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: cfg)
    }

    deinit {
        lock.lock()
        activeTask?.cancel()
        lock.unlock()
    }

    // MARK: 文件清单

    /// 列出仓库中推理需要的文件（过滤 README / 脚本 / 图片等）。
    ///
    /// 与单文件下载同样的 3 次重试：TLS/网络瞬时抖动（首次很快失败）
    /// 不应直接把错误抛给启动下载流程。
    public func listNeededFiles(repo: String) async throws -> [FileEntry] {
        var lastError: Error?
        for attempt in 0..<3 {
            if attempt > 0 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
            do {
                return try await listNeededFilesOnce(repo: repo)
            } catch {
                if error is CancellationError { throw error }
                lastError = error
            }
        }
        throw lastError ?? HFError.badResponse("获取模型文件列表失败：\(repo)")
    }

    private func listNeededFilesOnce(repo: String) async throws -> [FileEntry] {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("api/models/\(repo)/tree/main"),
            resolvingAgainstBaseURL: false)
        comps?.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        guard let url = comps?.url else {
            throw HFError.badResponse("无效的仓库地址：\(repo)")
        }
        let (data, response) = try await session.data(for: URLRequest(url: url))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw HFError.httpStatus(code) }
        guard let list = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw HFError.badResponse("模型文件列表格式异常：\(repo)")
        }
        var entries: [FileEntry] = []
        for item in list {
            // 远端清单不可信：先挡住 `../`、绝对路径、盘符等，否则会被
            // appendingPathComponent 落到模型目录之外（zip-slip 式写入）。
            guard (item["type"] as? String) == "file",
                  let path = item["path"] as? String,
                  Self.isSafeRelativePath(path),
                  Self.isNeededFile(path) else { continue }
            let size = (item["size"] as? NSNumber)?.int64Value ?? 0
            entries.append(FileEntry(path: path, size: size))
        }
        return entries
    }

    /// 远端清单里的路径能否安全落到模型目录内。
    ///
    /// 拒绝：绝对路径、`~`、空/`.`/`..` 段、Windows 反斜杠与盘符、URL scheme、
    /// NUL 和超长路径。`appendingPathComponent` 不会规范化 `..`，因此必须在
    /// 拼路径之前就过滤；`download` 里还有一次 containment 兜底。
    public static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, path.utf8.count <= 1024,
              !path.contains("\0"), !path.contains("\\"),
              !path.hasPrefix("/"), !path.hasPrefix("~") else { return false }
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !segments.isEmpty else { return false }
        for segment in segments {
            let part = String(segment)
            if part.isEmpty || part == "." || part == ".." { return false }
            if part.contains(":") { return false }
        }
        return true
    }

    /// 解析符号链接后，target 是否仍在 root 目录内（root 本身不算）。
    static func isContained(_ target: URL, in root: URL) -> Bool {
        let resolved = canonicalPath(target)
        let rootPath = canonicalPath(root)
        let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
        return resolved.hasPrefix(prefix)
    }

    /// 展开路径里所有**已存在**的符号链接，再拼回尚未创建的尾部组件。
    ///
    /// `URL.resolvingSymlinksInPath()` 在目标文件不存在时不会展开中间的
    /// symlink（下载前正是这种情形），直接用它会让 `root/link/model` 中的
    /// `link -> /elsewhere` 逃过前缀比对。所以从最深的不存在组件往上找到
    /// 第一个真实目录，对它做 realpath 后再拼回剩余组件。
    private static func canonicalPath(_ url: URL) -> String {
        let fm = FileManager.default
        var missing: [String] = []
        var cursor = url.standardizedFileURL
        while cursor.path != "/" {
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: cursor.path, isDirectory: &isDirectory) { break }
            missing.insert(cursor.lastPathComponent, at: 0)
            cursor = cursor.deletingLastPathComponent()
        }
        var resolved = cursor.resolvingSymlinksInPath().standardizedFileURL
        for component in missing {
            // 已解析部分本身可能是 symlink 目标；继续逐段 realpath，
            // 防止新拼上的组件里又出现指向外部的符号链接。
            resolved = resolved.appendingPathComponent(component)
                .resolvingSymlinksInPath().standardizedFileURL
        }
        return resolved.path
    }

    /// 推理运行时会读取的文件类型：权重（safetensors/onnx/gguf）、配置与
    /// tokenizer（json/jinja/model/txt）。README（md）、示例脚本、图片等排除。
    public static func isNeededFile(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        guard !name.hasPrefix(".") else { return false }
        let ext = (name as NSString).pathExtension.lowercased()
        switch ext {
        case "json", "jinja", "model", "txt", "safetensors", "onnx", "gguf":
            return true
        default:
            return false
        }
    }

    // MARK: 单文件下载（断点续传）

    /// 下载仓库内的一个文件到 `directory/<path>`。
    ///
    /// 目标文件已存在且大小一致时直接返回；存在 `.part` 临时文件时从其
    /// 当前偏移发 Range 请求续传。失败自动重试 3 次（每次从最新偏移继续）。
    public func download(repo: String, file: FileEntry, into directory: URL,
                         progress: @escaping @Sendable (_ received: Int64, _ total: Int64) -> Void
    ) async throws {
        // FileEntry 也可能由打包清单/子目录模型转换构造，不只来自远端 tree API；
        // 因此落盘前再校验一次，并确认解析符号链接后仍在目标目录内。
        guard Self.isSafeRelativePath(file.path),
              Self.isSafeRelativePath(file.remotePath) else {
            throw HFError.badResponse("非法的模型文件路径：\(file.path)")
        }
        let fm = FileManager.default
        let destination = directory.appendingPathComponent(file.path).standardizedFileURL
        guard Self.isContained(destination, in: directory) else {
            throw HFError.badResponse("模型文件路径越出目标目录：\(file.path)")
        }
        if Self.fileSize(at: destination) == file.size {
            progress(file.size, file.size)
            return
        }
        try fm.createDirectory(at: destination.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        let partial = destination.appendingPathExtension("part")

        var lastError: Error?
        for _ in 0..<3 {
            try Task.checkCancellation()
            do {
                try await downloadOnce(repo: repo, file: file, partial: partial,
                                       progress: progress)
                // 大小校验：不符则报错（.part 保留，下次从断点继续）。
                if file.size > 0 {
                    let got = Self.fileSize(at: partial)
                    guard got == file.size else {
                        throw HFError.badResponse(
                            "\(file.path) 下载不完整（\(got)/\(file.size) 字节）")
                    }
                }
                if fm.fileExists(atPath: destination.path) {
                    try fm.removeItem(at: destination)
                }
                try fm.moveItem(at: partial, to: destination)
                progress(file.size, file.size)
                return
            } catch {
                if error is CancellationError { throw error }
                lastError = error
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
        throw lastError ?? HFError.badResponse("下载失败：\(file.path)")
    }

    private func downloadOnce(repo: String, file: FileEntry, partial: URL,
                              progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        let fm = FileManager.default
        var offset = Self.fileSize(at: partial)
        if !fm.fileExists(atPath: partial.path) {
            fm.createFile(atPath: partial.path, contents: Data())
            offset = 0
        }
        let handle = try FileHandle(forWritingTo: partial)
        defer { try? handle.close() }
        if offset > 0 {
            try handle.seek(toOffset: UInt64(offset))
        } else {
            try handle.truncate(atOffset: 0)
        }

        var request = URLRequest(url: resolveURL(repo: repo, path: file.remotePath))
        if offset > 0 {
            request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
        }

        let sink = RangeDownloadSink(expectedSize: file.size, resumedAt: offset,
                                     handle: handle, progress: progress)
        let task = session.dataTask(with: request)
        task.delegate = sink
        lock.lock()
        activeTask = task
        lock.unlock()
        defer {
            lock.lock()
            if activeTask === task { activeTask = nil }
            lock.unlock()
        }

        let failure: Error? = await withCheckedContinuation { continuation in
            sink.onFinish = { continuation.resume(returning: $0) }
            task.resume()
        }
        if let failure { throw failure }
    }

    /// 取消当前正在下载的文件（.part 保留，下次从断点续传）。
    public func cancelCurrent() {
        lock.lock()
        let task = activeTask
        lock.unlock()
        task?.cancel()
    }

    private func resolveURL(repo: String, path: String) -> URL {
        var url = baseURL
            .appendingPathComponent(repo)
            .appendingPathComponent("resolve")
            .appendingPathComponent("main")
        for segment in path.split(separator: "/") {
            url = url.appendingPathComponent(String(segment))
        }
        return url
    }

    public static func fileSize(at url: URL) -> Int64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    }
}

/// 把 dataTask 的字节流写入 FileHandle 并上报进度的委托。
private final class RangeDownloadSink: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let expectedSize: Int64
    let resumedAt: Int64
    let handle: FileHandle
    let progress: @Sendable (Int64, Int64) -> Void
    /// 完成回调：nil 表示成功（由 didCompleteWithError / 内部失败判定决定）。
    var onFinish: (@Sendable (Error?) -> Void)?
    private var received: Int64
    private var failure: Error?
    private var lastReport = Date.distantPast
    private let lock = NSLock()

    init(expectedSize: Int64, resumedAt: Int64, handle: FileHandle,
         progress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.expectedSize = expectedSize
        self.resumedAt = resumedAt
        self.handle = handle
        self.progress = progress
        self.received = resumedAt
    }

    private func finish(_ error: Error?) {
        lock.lock()
        let alreadyDone = (onFinish == nil)
        let callback = onFinish
        onFinish = nil
        lock.unlock()
        guard !alreadyDone else { return }
        callback?(failure ?? error)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200...299).contains(code) else {
            if code == 416 {
                // Range 不可满足（本地残留多于远端文件）：删除 .part 重来。
                try? handle.truncate(atOffset: 0)
                received = 0
                failure = HFModelDownloader.HFError.httpStatus(code)
                completionHandler(.cancel)
                finish(nil)
                return
            }
            failure = HFModelDownloader.HFError.httpStatus(code)
            completionHandler(.cancel)
            finish(nil)
            return
        }
        if code == 200 && resumedAt > 0 {
            // 服务器忽略 Range 返回全量：从头重写。
            try? handle.truncate(atOffset: 0)
            received = 0
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        do {
            try handle.write(contentsOf: data)
        } catch {
            failure = error
            dataTask.cancel()
            return
        }
        received += Int64(data.count)
        // 进度回调限流到 ~10/s，避免 GB 级文件每块都刷新 UI。
        let now = Date()
        if now.timeIntervalSince(lastReport) >= 0.1 {
            lastReport = now
            progress(received, max(expectedSize, received))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let nsError = error as NSError?, nsError.code == NSURLErrorCancelled {
            // 主动取消：不算失败，也不重试。
            finish(CancellationError())
            return
        }
        if let error, failure == nil { failure = error }
        if failure == nil, expectedSize > 0, received != expectedSize {
            failure = HFModelDownloader.HFError.badResponse(
                "连接中断（\(received)/\(expectedSize) 字节），将从断点续传")
        }
        finish(nil)
    }
}

// MARK: - 默认配置的必需模型清单

/// 默认配置实际引用的本地模型（按当前配置解析目录与 HF 仓库名）。
public struct RequiredModel: Sendable, Equatable, Identifiable {
    public var id: String { repo }
    /// 用途显示名：主模型 / 辅助模型 / Embedding / Reranker / OCR / 语义切片。
    public let kind: String
    /// HF 仓库名：arlenecc/<模型名>。
    public let repo: String
    /// 本地目标目录（缺失时从这里补齐）。
    public let directory: URL
    /// 仓库内子目录：只下载该子目录下的文件并剥掉前缀（如 potion 的 `int4`）。
    public let subpath: String?
    /// 运行时真正会加载的权重文件名（basename 白名单）。
    /// `nil` = 不限制（按扩展名宽松判定，适用于分片 safetensors / gguf）。
    /// ONNX 后端只认 `resolveONNXModel` 候选名单里的名字，避免把仓库里
    /// `model_fp16.onnx` / 原版 `model.safetensors` 这类永远不读的权重拉下来。
    public let allowedWeightNames: Set<String>?
    /// 是否必须存在 tokenizer 文件（MLX / ONNX 模型的加载前置）。
    /// GGUF 的 tokenizer 内嵌在权重里，目录下载不带 tokenizer 文件，
    /// 这类模型必须为 false，否则会被永远判定缺失。
    public let requiresTokenizer: Bool

    public init(kind: String, repo: String, directory: URL,
                subpath: String? = nil,
                allowedWeightNames: Set<String>? = nil,
                requiresTokenizer: Bool = true) {
        self.kind = kind
        self.repo = repo
        self.directory = directory
        self.subpath = subpath
        self.allowedWeightNames = allowedWeightNames
        self.requiresTokenizer = requiresTokenizer
    }
}

// MARK: - doc_search 专用缺失模型清单

/// doc_search 运行时实际需要的本地模型（仅检索 ColBERT / 语义切片 potion / OCR VLM）。
/// 从主应用 RequiredModelCatalog 裁剪，去掉 LLM/Embedding/Reranker 等本工具不用的模型。
public enum DocSearchModels {
    /// 一个待补模型（复用下载器文件里的 RequiredModel 结构）。
    public typealias Model = RequiredModel

    /// 需要的权重扩展名（对应下载器 isNeededFile 放行范围）。
    private static let weightExtensions = ["onnx", "safetensors", "gguf", "npz"]

    /// 检索阅读 ColBERT：目录直接用 ToolConfig.modelDirectory（权重名固定 model_int8.onnx）。
    public static func colbert(directory: URL) -> Model {
        let name = directory.lastPathComponent
        return Model(
            kind: "检索阅读（ColBERT）",
            repo: "\(HFModelDownloader.repoOwner)/\(name)",
            directory: directory,
            allowedWeightNames: ["model_int8.onnx"],
            requiresTokenizer: true)
    }

    /// 语义切片 potion：仓库名与落点目录名不同，必须显式写 repo 与 subpath=int4。
    public static func potion(configuredDirectory: URL?) -> Model {
        let dir = configuredDirectory ?? PotionSentenceSimilarity.downloadDirectory()
        return Model(
            kind: "语义切片（potion）",
            repo: "\(HFModelDownloader.repoOwner)/potion-multilingual-128M-onnx",
            directory: dir,
            subpath: "int4",
            requiresTokenizer: true)
    }

    /// OCR VLM：按 ToolConfig 的模型路径解析落点；仅在选用 ovis/hybrid 时需要。
    public static func ocr(value: String, backend: String) -> Model {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let url: URL
        if !trimmed.isEmpty, (trimmed as NSString).isAbsolutePath {
            url = URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
        } else if !trimmed.isEmpty {
            url = URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
        } else {
            url = URL(fileURLWithPath: "models/OvisOCR3-MLX-4bit-mixed46")
        }
        let name = url.deletingPathExtension().lastPathComponent
        return Model(
            kind: "OCR 视觉模型（\(backend)）",
            repo: "\(HFModelDownloader.repoOwner)/\(name)",
            directory: url,
            requiresTokenizer: backend != "gguf") // GGUF tokenizer 内嵌在权重里
    }

    /// 当前 ToolConfig 需要的全部模型（去重）。OCR 仅在 PDF/EPUB 引擎选用 VLM 时纳入。
    static func required(for config: ToolConfig) -> [Model] {
        var models: [Model] = [
            colbert(directory: URL(fileURLWithPath: config.modelDirectory)),
        ]
        if config.usePotionSemantic {
            let trimmed = config.potionModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            let dir = trimmed.isEmpty ? nil
                : URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
            models.append(potion(configuredDirectory: dir))
        }
        let usesVLM = config.pdfOCREngine != "vision" || config.epubOCREngine != "vision"
        if usesVLM {
            models.append(ocr(value: config.ovisModelPath, backend: config.ocrBackend))
        }
        var seen = Set<String>()
        return models.filter { seen.insert($0.repo).inserted }
    }

    /// 缺失（未就绪）的模型。
    static func missing(for config: ToolConfig) -> [Model] {
        required(for: config).filter { !isComplete($0) }
    }

    /// 就绪判定：认可加载权重文件名 + 体积达标；MLX/ONNX/potion 还必须有 tokenizer。
    public static func isComplete(_ model: Model) -> Bool {
        guard isComplete(model.directory, allowedWeightNames: model.allowedWeightNames)
        else { return false }
        return !model.requiresTokenizer || hasTokenizerFile(in: model.directory)
    }

    public static func isComplete(_ directory: URL) -> Bool {
        isComplete(directory, allowedWeightNames: nil)
    }

    /// tokenizer.json 是否存在（potion/ColBERT/OCR 加载前置）。
    static func hasTokenizerFile(in directory: URL) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory) else { return false }
        guard isDirectory.boolValue else { return true }
        let scopes = [directory, directory.appendingPathComponent("onnx", isDirectory: true)]
        for scope in scopes {
            let url = scope.appendingPathComponent("tokenizer.json")
            if fm.fileExists(atPath: url.path), HFModelDownloader.fileSize(at: url) > 0 {
                return true
            }
        }
        return false
    }

    private static func isComplete(_ directory: URL,
                                   allowedWeightNames: Set<String>?) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory) else { return false }
        let minWeightBytes: Int64 = 4_096
        func usable(_ name: String, at url: URL) -> Bool {
            guard isWeightPath(name) else { return false }
            if let allowed = allowedWeightNames,
               !allowed.contains((name as NSString).lastPathComponent) { return false }
            return HFModelDownloader.fileSize(at: url) >= minWeightBytes
        }
        if !isDirectory.boolValue {
            return isWeightPath(directory.lastPathComponent)
                && HFModelDownloader.fileSize(at: directory) >= minWeightBytes
        }
        let scopes = [directory, directory.appendingPathComponent("onnx", isDirectory: true)]
        for scope in scopes {
            let items = (try? fm.contentsOfDirectory(atPath: scope.path)) ?? []
            for item in items where usable(item, at: scope.appendingPathComponent(item)) {
                return true
            }
        }
        return false
    }

    static func isWeightPath(_ path: String) -> Bool {
        weightExtensions.contains((path as NSString).pathExtension.lowercased())
    }

    /// 按模型的 subpath/白名单过滤下载清单：potion 剥 int4/ 前缀，ColBERT 只留 model_int8.onnx。
    public static func scopedFiles(
        for model: Model, in files: [HFModelDownloader.FileEntry]
    ) -> [HFModelDownloader.FileEntry] {
        let subpath = model.subpath?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !subpath.isEmpty || model.allowedWeightNames != nil else { return files }
        let prefix = subpath.isEmpty ? "" : (subpath.hasSuffix("/") ? subpath : subpath + "/")
        var out: [HFModelDownloader.FileEntry] = []
        for file in files {
            let localPath: String
            if prefix.isEmpty {
                localPath = file.path
            } else {
                guard file.path.hasPrefix(prefix) else { continue }
                localPath = String(file.path.dropFirst(prefix.count))
            }
            guard !localPath.isEmpty else { continue }
            if let allowed = model.allowedWeightNames,
               isWeightPath(localPath),
               !allowed.contains((localPath as NSString).lastPathComponent) { continue }
            out.append(HFModelDownloader.FileEntry(
                path: localPath, size: file.size, remotePath: file.remotePath))
        }
        return out
    }
}

/// 单个模型的下载编排：列清单 → 过滤 → 逐文件下载（断点续传）→ 就绪校验。
public struct DocSearchModelDownloader {
    public let downloader = HFModelDownloader()

    public init() {}

    /// 下载一个模型；progress 为聚合字节进度（received/total）。
    public func download(_ model: DocSearchModels.Model,
                         progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        let files = try await downloader.listNeededFiles(repo: model.repo)
        let scoped = DocSearchModels.scopedFiles(for: model, in: files)
        guard !scoped.isEmpty else {
            throw HFModelDownloader.HFError.badResponse("模型 \(model.repo) 没有可下载文件")
        }
        try FileManager.default.createDirectory(at: model.directory,
                                                withIntermediateDirectories: true)
        let total = scoped.reduce(Int64(0)) { $0 + $1.size }
        var done: Int64 = 0
        for file in scoped {
            try await downloader.download(repo: model.repo, file: file,
                                          into: model.directory) { received, _ in
                // 单文件进度叠加已完成文件体积，得到整模型进度。
                progress(min(total, done + received), total)
            }
            done += file.size
            progress(done, total)
        }
        guard DocSearchModels.isComplete(model) else {
            throw HFModelDownloader.HFError.badResponse(
                "模型 \(model.repo) 下载后仍不完整，请重试")
        }
    }

    public func cancel() {
        downloader.cancelCurrent()
    }
}
