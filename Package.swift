// swift-tools-version:5.9
import PackageDescription

/// 独立工具：问句 → 召回长文档（ColBERT 词向量 + 4bit 残差 + BM25/vec0 混合检索）。
/// 与主体 App 的 SPM 解耦；依赖仅 ONNX Runtime、swift-transformers、系统 SQLite3 与内置 sqlite-vec。
let package = Package(
    name: "DocSearch",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.5"),
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
        // OvisOCR3 视觉模型（MLX VLM），用于扫描件/图片内文字提取
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.0"),
        // 文档解析（复制自主应用，独立使用）
        .package(url: "https://github.com/weichsel/ZIPFoundation", exact: "0.9.20"),
        .package(url: "https://github.com/tadija/AEXML", exact: "4.7.3"),
        .package(url: "https://github.com/scinfu/SwiftSoup", exact: "2.14.3"),
        // GGUF 视觉模型（llama.cpp mtmd）与 chat template 渲染
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.2"),
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11815/llama-b11815-xcframework.zip",
            checksum: "a7d14f886fc7d15edcf505b84ddba2e05c1bf2222a8dc6d7bcf57b7810aa7174"
        ),
        .target(
            name: "CSQLiteVec",
            path: "CSQLiteVec",
            cSettings: [.define("SQLITE_CORE"), .define("SQLITE_VEC_STATIC")],
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .target(name: "CPDFShim", path: "CPDFShim"),
        .executableTarget(
            name: "DocSearch",
            dependencies: [
                "CSQLiteVec",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Jinja", package: "swift-jinja"),
                "llama",
                "CPDFShim",
                .product(name: "ZIPFoundation", package: "zipfoundation"),
                .product(name: "AEXML", package: "aexml"),
                .product(name: "SwiftSoup", package: "swiftsoup"),
            ],
            path: "Sources/DocSearch",
            resources: [.copy("Resources/stopwords_full.txt")],
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
    ]
)
