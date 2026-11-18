import Foundation

struct ChunkRecord {
    let title: String
    var content: String
    /// 所属章节（目录上级）ID：同一章内的小节可以整段合并。
    let sectionID: Int
    /// 所属小节 ID：跨小节的句子搬运/重叠都被禁止，尽量不切开小节。
    let partID: Int
}

struct ChunkOptions {
    var minChars = 2500
    var maxChars = 3000
    var targetChars = 2750
    var overlapRatio = 0.015
    var semanticThreshold = 0.55
    var similarityFloor = 0.15
    /// 是否用 potion-multilingual-128M（int4 ONNX）句向量做语义低谷切分。
    var usePotionSemantic = true
}

/// 结构优先 → 语义低谷 → 递归回退的长文档切片器。
/// 表格（不超长）与代码块作为原子块保持完整；断点一律落在句子/原子边界。
enum Chunker {
    private enum AtomKind { case prose, table, code, heading }
    private struct Atom { let text: String; let kind: AtomKind }
    private struct Section {
        let id: Int
        let parentID: Int
        let level: Int
        let title: String
        let atoms: [Atom]
    }

    static func split(_ text: String, sourceName: String, model: ColBERT?,
                      options: ChunkOptions = ChunkOptions()) throws -> [ChunkRecord] {
        let sections = parseSections(text, sourceName: sourceName)
        var chunks: [ChunkRecord] = []
        for section in sections {
            let pieces = try split(section, model: model, options: options)
            let heading = section.title.trimmingCharacters(in: .whitespacesAndNewlines)
            for piece in pieces {
                // 同一章节后续切片未必带原标题，补上标题让检索与阅读都有章节上下文。
                let content = (!heading.isEmpty && !piece.hasPrefix("#"))
                    ? "# \(heading)\n" + piece : piece
                chunks.append(ChunkRecord(title: section.title, content: content,
                                          sectionID: section.parentID, partID: section.id))
            }
        }
        return coalesce(chunks, options: options)
    }

    // MARK: - 结构解析

    private static let headingPatterns: [String] = [
        "^\\s{0,3}#{1,6}\\s+(.+?)\\s*$",
        "^\\s*第[一二三四五六七八九十百千零〇0-9]+[章节篇部卷](?:[\\s：:、.．]+.*)?$",
        "^\\s*(?:chapter|section|part)\\s+[0-9ivxlcdm]+(?:\\.[0-9]+)*(?:[.:：)）].*)?$",
    ]

    private static func isHeading(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return false }
        guard headingPatterns.contains(where: { pattern in
            line.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
        }) else { return false }
        return isPlausibleHeading(trimmed)
    }

    /// 标题层级：Markdown # 数量优先，中文“第 N 章/篇”与 Chapter/Part 视为一级目录。
    private static func headingLevel(_ line: String) -> Int {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let hashes = trimmed.prefix(while: { $0 == "#" }).count
        if hashes > 0 { return min(6, hashes) }
        let lower = trimmed.lowercased()
        if lower.hasPrefix("chapter") || lower.hasPrefix("part")
            || trimmed.range(of: "^第[一二三四五六七八九十百千零〇0-9]+[章节篇部卷]",
                             options: .regularExpression) != nil {
            return 1
        }
        return 2
    }

    /// OCR 常把页眉、参考文献残片、URL 写成 `##` 标题；这类行不能当章节边界。
    private static func isPlausibleHeading(_ line: String) -> Bool {
        var title = line
        if let range = title.range(of: "^\\s{0,3}#{1,6}\\s+", options: .regularExpression) {
            title = String(title[range.upperBound...])
        }
        title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard title.count >= 2, title.count <= 80 else { return false }
        guard !title.contains(where: { "。！？；".contains($0) }) else { return false }
        guard !DocumentOutline.isPageHeaderLine(title),
              !DocumentOutline.isBareNumberTitle(title),
              !DocumentOutline.looksLikeDateProse(title) else { return false }
        // `p. 19.` 这类尾点页码：pageHeader 正则要求严格结尾，这里单独兜住。
        if title.hasSuffix("."), title.contains(where: { $0.isNumber }) { return false }
        let lower = title.lowercased()
        if lower.contains("http") || lower.contains(".pdf") || lower.contains("%20")
            || title.contains("/") || title.contains("\\") { return false }
        return true
    }

    private static func parseSections(_ text: String, sourceName: String) -> [Section] {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var sections: [Section] = []
        var title = sourceName
        var atoms: [Atom] = []
        var paragraph: [String] = []
        var inCode = false
        var codeLines: [String] = []
        var fence = ""
        var tableLines: [String] = []
        var currentID = 0
        var currentParentID = 0
        var currentLevel = 0
        var nextID = 1

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            let raw = paragraph.joined(separator: "\n")
            let sentences = TextAnalysis.sentences(raw)
            if sentences.isEmpty {
                atoms.append(Atom(text: raw + "\n", kind: .prose))
            } else {
                for (i, s) in sentences.enumerated() {
                    atoms.append(Atom(text: s + (i == sentences.count - 1 ? "\n" : " "), kind: .prose))
                }
            }
            paragraph.removeAll(keepingCapacity: true)
        }
        func flushTable() {
            guard !tableLines.isEmpty else { return }
            atoms.append(Atom(text: tableLines.joined(separator: "\n") + "\n", kind: .table))
            tableLines.removeAll(keepingCapacity: true)
        }
        func flushSection() {
            flushParagraph(); flushTable()
            if !atoms.isEmpty {
                sections.append(Section(id: currentID, parentID: currentParentID,
                                        level: currentLevel, title: title, atoms: atoms))
            }
            atoms.removeAll(keepingCapacity: true)
        }

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if inCode {
                codeLines.append(line)
                if trimmed.hasPrefix(fence) {
                    atoms.append(Atom(text: codeLines.joined(separator: "\n") + "\n", kind: .code))
                    codeLines.removeAll(); inCode = false; fence = ""
                }
                continue
            }
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushParagraph(); flushTable()
                fence = String(trimmed.prefix(3))
                codeLines = [line]
                inCode = true
                continue
            }
            if trimmed.contains("|") {
                flushParagraph()
                tableLines.append(line)
                continue
            }
            flushTable()
            if isHeading(trimmed) {
                flushSection()
                let level = Self.headingLevel(trimmed)
                currentID = nextID; nextID += 1
                currentLevel = level
                // 一级目录开启新章；小节沿用当前章 ID，便于同章内整段合并。
                currentParentID = level <= 1 ? currentID : currentParentID
                title = trimmed.replacingOccurrences(of: "^\\s*#+\\s*", with: "", options: .regularExpression)
                atoms.append(Atom(text: line + "\n", kind: .heading))
            } else if trimmed.isEmpty {
                flushParagraph()
            } else {
                paragraph.append(line)
            }
        }
        if inCode { atoms.append(Atom(text: codeLines.joined(separator: "\n"), kind: .code)) }
        flushSection()
        return sections.isEmpty
            ? [Section(id: 0, parentID: 0, level: 0, title: sourceName, atoms: [])]
            : sections
    }

    // MARK: - 章节内切分

    private static func split(_ section: Section, model: ColBERT?,
                              options: ChunkOptions) throws -> [String] {
        guard !section.atoms.isEmpty else { return [] }
        var atoms = section.atoms
        if atoms.allSatisfy({ $0.kind == .prose }), atoms.count == 1,
           let expanded = expandProse(atoms[0]) {
            atoms = expanded
        }
        let whole = atoms.map(\.text).joined()
        if whole.count <= options.maxChars { return [whole] }

        let texts = atoms.map(\.text)
        // 语义信号优先级：potion（int4 ONNX 多语言句向量，真正的语义相似度）
        // → ColBERT 词向量中心 → 词法 Jaccard 回退。potion 失败不阻断切片。
        let potion = options.usePotionSemantic
            ? PotionSentenceSimilarity().adjacentCosines(texts) : nil
        let semantic = potion ?? model?.adjacentSimilarities(texts)
        let similarities = (semantic?.count == texts.count - 1)
            ? semantic!
            : (0..<max(0, texts.count - 1)).map { i in TextAnalysis.lexicalSimilarity(texts[i], texts[i + 1]) }
        let threshold = adaptiveThreshold(similarities, fallback: options.semanticThreshold,
                                          floor: options.similarityFloor)
        var pieces: [String] = []
        var current: [Atom] = []
        var currentChars = 0

        func flush(at index: Int) {
            guard !current.isEmpty else { return }
            let cut = min(max(1, index), current.count)
            let head = current.prefix(cut).map(\.text).joined()
            pieces.append(head.trimmingCharacters(in: .whitespacesAndNewlines))
            // 不在切分阶段插 overlap：后续 mergeShort / ensureMin 会搬动句子，
            // 把重叠吞掉或推到中间。统一在 coalesce 完成后的 applyOverlap 里补。
            current = Array(current.dropFirst(cut))
            currentChars = current.map(\.text.count).reduce(0, +)
        }

        for (i, atom) in atoms.enumerated() {
            let count = atom.text.count
            if count > options.maxChars {
                flush(at: current.count)
                pieces.append(contentsOf: splitOversized(atom, options: options))
                current.removeAll(); currentChars = 0
                continue
            }
            if currentChars + count > options.maxChars, !current.isEmpty {
                flush(at: chooseBreak(current, similarities: similarities,
                                      offset: i - current.count, threshold: threshold,
                                      minChars: options.minChars, maxChars: options.maxChars))
            }
            current.append(atom)
            currentChars += count
            let boundary = i < similarities.count ? similarities[i] : Double.infinity
            if currentChars >= options.minChars, boundary <= threshold, i + 1 < atoms.count {
                flush(at: current.count)
            }
        }
        if !current.isEmpty {
            pieces.append(current.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return mergeShort(pieces, options: options)
    }

    /// 在长度范围内选相似度最低的边界；找不到低谷时退到最后一个原子（必为句末/原子末）。
    private static func chooseBreak(_ atoms: [Atom], similarities: [Double], offset: Int,
                                    threshold: Double, minChars: Int, maxChars: Int) -> Int {
        guard atoms.count > 1 else { return 1 }
        var best = atoms.count - 1
        var bestSim = Double.infinity
        var length = 0
        for j in 0..<atoms.count {
            length += atoms[j].text.count
            if length > maxChars { break }
            guard j >= 1, length >= minChars * 3 / 4 else { continue }
            let simIndex = offset + j
            let sim = simIndex < similarities.count ? similarities[simIndex] : Double.infinity
            if sim <= threshold, sim < bestSim { best = j + 1; bestSim = sim }
        }
        return min(best, atoms.count)
    }

    private static func expandProse(_ atom: Atom) -> [Atom]? {
        let sentences = TextAnalysis.sentences(atom.text)
        guard sentences.count > 1 else { return nil }
        return sentences.map { Atom(text: $0 + " ", kind: .prose) }
    }

    private static func splitOversized(_ atom: Atom, options: ChunkOptions) -> [String] {
        let lines = atom.text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        switch atom.kind {
        case .table where lines.count > 2:
            let header = lines.prefix(2).joined(separator: "\n")   // 表头 + 分隔行
            var out: [String] = []
            var current = header
            for row in lines.dropFirst(2) {
                if current.count + row.count + 1 > options.maxChars {
                    out.append(current); current = header
                }
                current += "\n" + row
            }
            out.append(current)
            return out
        case .code:
            let fence = String(lines.first?.trimmingCharacters(in: .whitespaces).prefix(3) ?? "```")
            var out: [String] = []
            var current = fence
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix(fence) { continue }
                if current.count + line.count + 1 > options.maxChars {
                    out.append(current + "\n" + fence); current = fence
                }
                current += "\n" + line
            }
            out.append(current + "\n" + fence)
            return out
        default:
            let sentences = TextAnalysis.sentences(atom.text)
            var out: [String] = []
            var current = ""
            for sentence in sentences {
                if !current.isEmpty, current.count + sentence.count > options.maxChars {
                    out.append(current); current = ""
                }
                current = TextAnalysis.join(current, sentence)
            }
            if !current.isEmpty { out.append(current) }
            return splitByCharactersIfNeeded(out, max: options.maxChars)
        }
    }

    private static func splitByCharactersIfNeeded(_ pieces: [String], max: Int) -> [String] {
        var out: [String] = []
        for piece in pieces {
            if piece.count <= max { out.append(piece); continue }
            var start = piece.startIndex
            while start < piece.endIndex {
                let end = piece.index(start, offsetBy: max, limitedBy: piece.endIndex) ?? piece.endIndex
                out.append(String(piece[start..<end]))
                start = end
            }
        }
        return out
    }

    // MARK: - 合并小片 / 自适应阈值

    private static func adaptiveThreshold(_ values: [Double], fallback: Double, floor: Double) -> Double {
        guard values.count >= 4 else { return fallback }
        let sorted = values.sorted()
        let index = Int(Double(sorted.count - 1) * 0.25)
        return max(floor, min(fallback, sorted[index]))
    }

    private static func mergeShort(_ pieces: [String], options: ChunkOptions) -> [String] {
        guard pieces.count > 1 else { return pieces }
        var merged: [String] = []
        for piece in pieces {
            if let last = merged.last,
               last.count < options.minChars || piece.count < options.minChars,
               last.count + piece.count + 1 <= options.maxChars {
                merged[merged.count - 1] = last + "\n" + piece
            } else {
                merged.append(piece)
            }
        }
        return merged
    }

    /// 只在同章节内合并/补齐：跨章节会把不同主题拼进同一片，还会让标题与正文错配。
    private static func coalesce(_ chunks: [ChunkRecord], options: ChunkOptions) -> [ChunkRecord] {
        guard chunks.count > 1 else { return chunks }
        var result: [ChunkRecord] = []
        var group: [ChunkRecord] = []
        for chunk in chunks {
            if let last = group.last, last.sectionID != chunk.sectionID {
                result.append(contentsOf: coalesceSameSection(group, options: options))
                group.removeAll()
            }
            group.append(chunk)
        }
        if !group.isEmpty { result.append(contentsOf: coalesceSameSection(group, options: options)) }
        return applyOverlap(result, options: options)
    }

    /// 合并/补齐全部完成后再补重叠：把上一片尾部按字符搬到下一片开头。
    /// 只在同章节相邻片之间做，避免跨主题重复，也不会被 ensureMin 之类的搬运吞掉。
    private static func applyOverlap(_ chunks: [ChunkRecord], options: ChunkOptions) -> [ChunkRecord] {
        let chars = max(0, Int(Double(options.targetChars) * options.overlapRatio))
        guard chars > 0, chunks.count > 1 else { return chunks }
        var out = chunks
        for i in stride(from: out.count - 1, through: 1, by: -1) {
            guard out[i].sectionID == out[i - 1].sectionID,
                  out[i].partID == out[i - 1].partID else { continue }
            var prefix = String(out[i - 1].content.suffix(chars))
            // 上一片末尾恰好是标题行时不要重复注入标题。
            while prefix.hasPrefix("#"), let newline = prefix.firstIndex(of: "\n") {
                prefix = String(prefix[prefix.index(after: newline)...])
            }
            guard !prefix.isEmpty else { continue }
            let content = out[i].content
            // 重叠不能把切片顶过 maxChars。
            let room = max(0, options.maxChars - content.count)
            guard room > 0 else { continue }
            if prefix.count > room { prefix = String(prefix.suffix(room)) }
            if content.hasPrefix(prefix) { continue }
            if content.hasPrefix("#"), let newline = content.firstIndex(of: "\n") {
                let head = String(content[...newline])
                let body = String(content[content.index(after: newline)...])
                out[i].content = head + prefix + body
            } else {
                out[i].content = prefix + content
            }
        }
        return out
    }

    private static func coalesceSameSection(_ chunks: [ChunkRecord], options: ChunkOptions) -> [ChunkRecord] {
        guard chunks.count > 1 else { return chunks }
        var out: [ChunkRecord] = []
        for chunk in chunks {
            if let last = out.last,
               last.content.count < options.minChars || chunk.content.count < options.minChars {
                if last.content.count + chunk.content.count + 1 <= options.maxChars {
                    out[out.count - 1] = ChunkRecord(title: last.title,
                                                    content: last.content + "\n" + chunk.content,
                                                    sectionID: last.sectionID, partID: last.partID)
                    continue
                }
                // 合成会超上限：只在同小节拆出的片之间搬句子，不切开另一个小节。
                if last.partID == chunk.partID, let split = rebalance(last, chunk, options: options) {
                    out.removeLast()
                    out.append(contentsOf: split)
                    continue
                }
            }
            out.append(chunk)
        }
        var result = out
        for i in stride(from: result.count - 1, through: 1, by: -1) { ensureMin(i, in: &result, options: options) }
        for i in 0..<max(0, result.count - 1) { ensureMinForward(i, in: &result, options: options) }
        return result
    }

    /// 首片偏短时从下一片头部递归搬句子（先保证下一片自身达标）。
    private static func ensureMinForward(_ index: Int, in chunks: inout [ChunkRecord],
                                         options: ChunkOptions) {
        guard index + 1 < chunks.count,
              chunks[index].partID == chunks[index + 1].partID,
              chunks[index].content.count < options.minChars else { return }
        ensureMinForward(index + 1, in: &chunks, options: options)
        while chunks[index].content.count < options.minChars {
            let nextSentences = TextAnalysis.sentences(chunks[index + 1].content)
            guard let head = nextSentences.first, nextSentences.count > 1 else { return }
            let rest = nextSentences.dropFirst().joined(separator: " ")
            let candidate = chunks[index].content + "\n" + head
            guard rest.count >= options.minChars / 2, candidate.count <= options.maxChars else { return }
            chunks[index].content = candidate
            chunks[index + 1].content = rest
        }
    }

    /// 从尾到头递归补齐短片：先补上一片，再从上一片句尾搬句子过来。
    private static func ensureMin(_ index: Int, in chunks: inout [ChunkRecord], options: ChunkOptions) {
        guard index > 0,
              chunks[index].partID == chunks[index - 1].partID,
              chunks[index].content.count < options.minChars else { return }
        ensureMin(index - 1, in: &chunks, options: options)
        while chunks[index].content.count < options.minChars {
            let donorSentences = TextAnalysis.sentences(chunks[index - 1].content)
            guard donorSentences.count > 1, let tail = donorSentences.last else { return }
            let remaining = donorSentences.dropLast().joined(separator: " ")
            let candidate = tail + "\n" + chunks[index].content
            guard remaining.count >= options.minChars / 2, candidate.count <= options.maxChars else { return }
            chunks[index - 1].content = remaining
            chunks[index].content = candidate
        }
    }

    private static func rebalance(_ first: ChunkRecord, _ second: ChunkRecord,
                                  options: ChunkOptions) -> [ChunkRecord]? {
        var kept = TextAnalysis.sentences(first.content)
        guard kept.count > 1 else { return nil }
        var moved: [String] = []
        var nextText = second.content
        while nextText.count < options.minChars, kept.count > 1, let tail = kept.last {
            let candidate = tail + "\n" + nextText
            if candidate.count > options.maxChars { break }
            moved.insert(tail, at: 0)
            kept.removeLast()
            nextText = candidate
        }
        let firstText = kept.joined(separator: " ")
        guard !moved.isEmpty, firstText.count >= options.minChars * 4 / 5,
              nextText.count >= options.minChars * 9 / 10,
              firstText.count <= options.maxChars, nextText.count <= options.maxChars else { return nil }
        return [ChunkRecord(title: first.title, content: firstText,
                            sectionID: first.sectionID, partID: first.partID),
                ChunkRecord(title: second.title, content: nextText,
                            sectionID: second.sectionID, partID: second.partID)]
    }
}
