import Foundation
import Hub
import Tokenizers

/// 本地 tokenizer 精简封装（供 ONNX VLM 服务使用）。
/// 从主应用 TokenizerSupport.swift 裁剪；只保留 encode/decode 与特殊 token 访问。
/// tokenizer.json 必须存在；tokenizer_config.json 缺失时用最小合成配置加载。
public final class LocalTokenizer: @unchecked Sendable {
    public let base: any Tokenizers.Tokenizer

    private init(base: any Tokenizers.Tokenizer) {
        self.base = base
    }

    public static func load(directory: URL) async throws -> LocalTokenizer {
        let fm = FileManager.default
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard fm.fileExists(atPath: tokenizerURL.path) else {
            throw LocalModelError.missingTokenizer(
                directory: directory.path, file: "tokenizer.json")
        }
        let base: any Tokenizers.Tokenizer
        do {
            if fm.fileExists(atPath: directory.appendingPathComponent("tokenizer_config.json").path) {
                // strict:false：兼容 Qwen3_5Tokenizer 等新版本 transformers 类名。
                base = try await Tokenizers.AutoTokenizer.from(modelFolder: directory, strict: false)
            } else {
                // tokenizer_config 缺失：用 tokenizer.json 数据 + 合成最小配置加载。
                let tokenizerData = try HubApi.shared.configuration(fileURL: tokenizerURL)
                let synthetic = Config([
                    "tokenizer_class": Config("PreTrainedTokenizer"),
                    "model_max_length": Config(8192),
                ])
                base = try Tokenizers.AutoTokenizer.from(
                    tokenizerConfig: synthetic, tokenizerData: tokenizerData, strict: false)
            }
        } catch let error as LocalModelError {
            throw error
        } catch {
            throw LocalModelError.runtime(
                "tokenizer 加载失败（\(directory.path)）：\(error.localizedDescription)")
        }
        return LocalTokenizer(base: base)
    }

    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        base.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }
}
