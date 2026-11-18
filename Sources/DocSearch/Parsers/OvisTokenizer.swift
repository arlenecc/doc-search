import Foundation
import Tokenizers
import MLXLMCommon

// OvisOCR3 tokenizer 桥接：从主应用 TokenizerSupport.swift 裁剪。
// OvisOCR3 的 tokenizer_config.json 里 tokenizer_class=Qwen3_5Tokenizer（新版本
// transformers 类名），必须 strict:false 回退到通用 BPE；chat template 不在
// tokenizer.json / tokenizer_config.json 里，而在 chat_template.jinja，需显式注入。

/// 供 VLMModelFactory 本地目录加载使用的 tokenizer loader。
public struct OvisTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let base = try await Tokenizers.AutoTokenizer.from(modelFolder: directory, strict: false)
        let override = Self.jinjaTemplate(in: directory)
        return OvisTokenizerBridge(base, chatTemplateOverride: override)
    }

    /// 读取 chat_template.jinja（OvisOCR3 的模板落在这里）。
    private static func jinjaTemplate(in directory: URL) -> String? {
        let url = directory.appendingPathComponent("chat_template.jinja")
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return ChatTemplateCompatibility.strippingSeparators(text)
    }
}

/// HF chat template 与 swift-jinja 的兼容补丁（MLX tokenizer 与 GGUF runtime 共用）。
public enum ChatTemplateCompatibility {
    /// Spark-X2.6 等模板写了 `tojson(separators=(',', ':'))`，swift-jinja 的 tojson
    /// 不接受 `separators` 参数会抛错；这里去掉该关键字参数，JSON 内容等价。
    public static func strippingSeparators(_ template: String) -> String {
        template.replacingOccurrences(
            of: #"tojson\s*\(\s*separators\s*=\s*\([^)]*\)\s*\)"#,
            with: "tojson",
            options: .regularExpression)
    }
}

/// 把 swift-transformers 的 tokenizer 桥接给 MLXLMCommon.Tokenizer。
public final class OvisTokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let base: any Tokenizers.Tokenizer
    private let chatTemplateOverride: String?

    public init(_ base: any Tokenizers.Tokenizer, chatTemplateOverride: String? = nil) {
        self.base = base
        self.chatTemplateOverride = chatTemplateOverride
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        base.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        base.convertIdToToken(id)
    }

    public var bosToken: String? { base.bosToken }
    public var eosToken: String? { base.eosToken }
    public var unknownToken: String? { base.unknownToken }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let formatted: [Tokenizers.Message] = messages.map { message in
            // Message 是 [String: any Sendable] 的 typealias，字典直接返回即可。
            message
        }
        let template: Tokenizers.ChatTemplateArgument? = chatTemplateOverride.map { .literal($0) }
        // OCR 场景固定关闭思考模式；Qwen3.5 模板据此跳过 <think> 段。
        var context: [String: any Sendable] = ["enable_thinking": false]
        if let additionalContext {
            for (key, value) in additionalContext { context[key] = value }
        }
        return try base.applyChatTemplate(
            messages: formatted,
            chatTemplate: template,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil,
            additionalContext: context
        )
    }
}
