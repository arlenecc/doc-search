import Foundation

/// 字符级 N-gram BM25：在一个文本切片内部，把每个句子当作一篇文档，
/// query 当作查询，选出最能代表该切片与 query 相关性的若干句子。
///
/// 参数按需求固定：字符级 3-gram 滑动窗口、k1=1.6、b=0.4。
/// BM25 的语料就是该切片内的句子集合（IDF 在切片内计算）。
enum SentenceBM25 {
    static let k1 = 1.6
    static let b = 0.4
    static let ngram = 3

    /// 选出分数最高的 maxSentences 句，按它们在原文中出现的顺序返回。
    static func coreSentences(query: String, content: String,
                              maxSentences: Int = 5,
                              maxSentenceChars: Int = 160) -> String {
        let sentences = splitSentences(content)
        guard !sentences.isEmpty else { return "" }

        let queryGrams = Array(Set(grams(query)))
        guard !queryGrams.isEmpty else {
            return sentences.prefix(maxSentences)
                .map { clip($0, maxSentenceChars) }
                .joined(separator: "\n")
        }

        let docLists = sentences.map { grams($0) }
        let docSets = docLists.map { Set($0) }
        let count = sentences.count
        let avgdl = Double(docLists.reduce(0) { $0 + $1.count }) / Double(count)

        var idf: [String: Double] = [:]
        for g in queryGrams {
            let df = docSets.reduce(0) { $0 + ($1.contains(g) ? 1 : 0) }
            idf[g] = log(1.0 + (Double(count - df) + 0.5) / (Double(df) + 0.5))
        }

        var scores = [Double](repeating: 0, count: count)
        for (i, list) in docLists.enumerated() {
            var tf: [String: Int] = [:]
            for g in list { tf[g, default: 0] += 1 }
            let dl = Double(max(list.count, 1))
            let norm = k1 * (1 - b + b * dl / max(avgdl, 1))
            for g in queryGrams {
                guard let f = tf[g], f > 0 else { continue }
                scores[i] += idf[g]! * (Double(f) * (k1 + 1)) / (Double(f) + norm)
            }
        }

        let selected = scores.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(maxSentences)
            .map { $0.offset }
            .sorted()
        return selected.map { clip(sentences[$0], maxSentenceChars) }
                       .joined(separator: "\n")
    }

    /// 按中英文句末标点与换行切句，保留句末标点。
    private static func splitSentences(_ text: String) -> [String] {
        let delimiters: Set<Character> = ["。", "！", "？", "!", "?", "；", ";", "\n"]
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if delimiters.contains(ch) {
                let sentence = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if sentence.count >= 2 { out.append(sentence) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if tail.count >= 2 { out.append(tail) }
        return out
    }

    /// 字符级 3-gram 滑动窗口；不足 3 字的句子整体作为一个 gram。
    private static func grams(_ text: String) -> [String] {
        let chars = Array(text.trimmingCharacters(in: .whitespacesAndNewlines))
        guard chars.count >= ngram else { return chars.isEmpty ? [] : [String(chars)] }
        var out: [String] = []
        for i in 0...(chars.count - ngram) {
            out.append(String(chars[i..<(i + ngram)]))
        }
        return out
    }

    private static func clip(_ text: String, _ limit: Int) -> String {
        text.count <= limit ? text : String(text.prefix(limit))
    }
}
