import SwiftUI

struct ChatView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(model.messages) { item in
                            bubble(item).id(item.id)
                        }
                    }
                    .padding(20)
                }
                .onChange(of: model.messages.count) { _, _ in
                    if let last = model.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            Divider()
            HStack(spacing: 10) {
                TextField("输入问题，直接检索知识库…", text: $model.input)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await model.send() } }
                Button {
                    Task { await model.send() }
                } label: {
                    Image(systemName: "arrow.up.circle.fill").font(.title2)
                }
                .disabled(model.input.trimmingCharacters(in: .whitespaces).isEmpty || model.isSearching)
                .help("检索")
            }
            .padding(14)
            HStack {
                Text(model.status).font(.caption).foregroundStyle(.secondary)
                Spacer()
                if model.isSearching { ProgressView().scaleEffect(0.7) }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
    }

    @ViewBuilder
    private func bubble(_ item: ChatBubbleItem) -> some View {
        HStack {
            if item.role == .user { Spacer(minLength: 80) }
            VStack(alignment: .leading, spacing: 8) {
                Text(item.text)
                    .textSelection(.enabled)
                    .font(.body)
                if let hits = item.hits, !hits.isEmpty {
                    ResultChipRow(hits: hits, messageID: item.id)
                }
            }
            .padding(12)
            .background(item.role == .user ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .frame(maxWidth: 820, alignment: item.role == .user ? .trailing : .leading)
            if item.role == .assistant { Spacer(minLength: 80) }
        }
    }
}

private struct ResultChipRow: View {
    @EnvironmentObject var model: AppModel
    let hits: [SearchHit]
    let messageID: UUID

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 8)], spacing: 8) {
            ForEach(Array(hits.enumerated()), id: \.element.docID) { index, hit in
                Button {
                    model.openReader(messageID: messageID, index: index)
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "doc.text")
                            Text(hit.title).font(.caption.weight(.semibold)).lineLimit(1)
                            Spacer()
                            Text(String(format: "%.2f", hit.score)).font(.caption2).monospacedDigit()
                        }
                        Text(hit.content.prefix(90))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                    .padding(9)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.accentColor.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(.plain)
                .contentShape(Rectangle())
                .help("打开分片阅读")
            }
        }
    }
}
