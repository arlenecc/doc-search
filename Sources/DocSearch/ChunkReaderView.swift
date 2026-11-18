import SwiftUI

struct ChunkReaderView: View {
    @EnvironmentObject var model: AppModel
    let state: ReaderState

    private var hit: SearchHit { state.hits[state.index] }

    /// 把查询词在正文中标黄加粗，方便定位命中位置。
    private var highlightedText: Text {
        let terms = TextAnalysis.words(state.query).filter {
            !TokenPruner.shared.isStopword($0) && $0.count >= 2
        }
        guard !terms.isEmpty else { return Text(hit.content) }
        let ns = hit.content as NSString
        var result = Text("")
        var location = 0
        while location < ns.length {
            var best: NSRange? = nil
            for term in terms {
                let found = ns.range(of: term, options: [.caseInsensitive],
                                     range: NSRange(location: location, length: ns.length - location))
                if found.location != NSNotFound, best == nil || found.location < best!.location {
                    best = found
                }
            }
            guard let range = best else { break }
            if range.location > location {
                result = result + Text(ns.substring(with: NSRange(location: location,
                                                                   length: range.location - location)))
            }
            var marked = AttributedString(ns.substring(with: range))
            marked.font = .body.bold()
            marked.backgroundColor = .yellow
            result = result + Text(marked)
            location = range.location + range.length
        }
        if location < ns.length { result = result + Text(ns.substring(from: location)) }
        return result
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button {
                    model.moveReader(-1)
                } label: {
                    Label("上一片", systemImage: "chevron.left")
                }
                .disabled(state.index == 0)
                .keyboardShortcut("[", modifiers: .command)

                Spacer()
                VStack(spacing: 2) {
                    Text(hit.title).font(.headline)
                    Text("分片 \(state.index + 1)/\(state.hits.count) · score \(String(format: "%.4f", hit.score))"
                         + " · vec#\(hit.vectorRank.map(String.init) ?? "-") bm25#\(hit.bm25Rank.map(String.init) ?? "-")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()

                Button {
                    model.moveReader(1)
                } label: {
                    Label("下一片", systemImage: "chevron.right")
                }
                .disabled(state.index >= state.hits.count - 1)
                .keyboardShortcut("]", modifiers: .command)
            }
            .padding()
            Divider()
            ScrollView {
                highlightedText
                    .textSelection(.enabled)
                    .font(.body)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(20)
            }
            Divider()
            HStack {
                Text(hit.docID).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                Button("关闭") { model.reader = nil }
            }
            .padding(12)
        }
    }
}
