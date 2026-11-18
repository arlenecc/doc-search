import SwiftUI
import AppKit

/// 裁剪自主应用 SettingsView：只保留知识同步与检索阅读相关设置。
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var draft: ToolConfig
    let apply: (ToolConfig) -> Void

    init(config: ToolConfig, apply: @escaping (ToolConfig) -> Void) {
        _draft = State(initialValue: config)
        self.apply = apply
    }

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("知识同步") {
                    Toggle("启用向量检索（关闭则只入库文本切片，省空间）", isOn: $draft.useVectorSearch)
                    pathField("源目录", text: $draft.sourceDirectory, chooseDirectory: true)
                    pathField("工作目录", text: $draft.workDirectory, chooseDirectory: true)
                    Text("Markdown：\(draft.outputDirectory)\n数据库：\(draft.databasePath)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    pathField("ColBERT 模型目录", text: $draft.modelDirectory, chooseDirectory: true)
                    if draft.useVectorSearch {
                        Stepper("词向量回填线程 \(draft.vectorBackfillThreads)",
                                value: $draft.vectorBackfillThreads, in: 1...8)
                        Text("同步先入库文本切片，词向量后台异步填充；填充完成前向量检索结果不完整，可直接用决策判定模式。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Toggle("增量转换（源文件未变化时跳过）", isOn: $draft.incrementalConversion)
                    Stepper("并发数 \(draft.syncConcurrency)", value: $draft.syncConcurrency, in: 1...8)
                }

                Section("高级设置") {
                    DisclosureGroup("文档解析 / OCR") {
                        Picker("PDF 引擎", selection: $draft.pdfOCREngine) {
                            Text("Vision（系统 OCR，快）").tag("vision")
                            Text("OvisOCR3（VLM，保表格/公式）").tag("ovis")
                            Text("Hybrid（Vision + 表格页 Ovis）").tag("hybrid")
                        }
                        Picker("EPUB 扫描页引擎", selection: $draft.epubOCREngine) {
                            Text("Vision").tag("vision")
                            Text("OvisOCR3").tag("ovis")
                        }
                        Picker("OCR 模型后端", selection: $draft.ocrBackend) {
                            Text("MLX（OvisOCR3，Apple Silicon）").tag("mlx")
                            Text("GGUF（llama.cpp + mmproj）").tag("gguf")
                            Text("ONNX（合并 VLM / 检测+识别）").tag("onnx")
                        }
                        pathField("OCR 模型路径", text: $draft.ovisModelPath, chooseDirectory: true)
                        if draft.ocrBackend == "gguf" {
                            pathField("mmproj 路径", text: $draft.ocrMMProjPath, chooseDirectory: false)
                        }
                        Stepper("OCR 实例数 \(draft.ovisInstances)",
                                value: $draft.ovisInstances, in: 1...2)
                        Stepper("渲染 DPI \(Int(draft.parseRenderDPI))",
                                value: $draft.parseRenderDPI, in: 100...400, step: 25)
                        Stepper("最大页数（0 = 不限）\(draft.parseMaxPages)",
                                value: $draft.parseMaxPages, in: 0...10000, step: 50)
                        Toggle("OCR 预处理归一化", isOn: $draft.parsePreprocessOCR)
                        Toggle("OCR 二值化", isOn: $draft.parseBinarizeOCR)
                        Toggle("保留页眉页脚", isOn: $draft.parseKeepHeadersFooters)
                        Text("Vision 用系统 OCR；VLM 支持 MLX / GGUF / ONNX 三种本地视觉模型，扫描件的表格与公式更准。模型路径按后端填：MLX 目录、.gguf 文件、ONNX 目录或 .onnx 文件。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    DisclosureGroup("切片 / 词向量 / 残差") {
                        Stepper("最小字数 \(draft.chunkMinChars)", value: $draft.chunkMinChars,
                                in: 2500...2900, step: 100)
                        Stepper("最大字数 \(draft.chunkMaxChars)", value: $draft.chunkMaxChars,
                                in: 3000...12000, step: 100)
                        Slider(value: $draft.chunkOverlapRatio, in: 0...0.05, step: 0.005)
                        Text("overlap：\(Int(draft.chunkOverlapRatio * 1000) / 10)%")
                            .font(.caption2).foregroundStyle(.secondary)
                        Toggle("potion 真语义切片（int4 ONNX 多语言句向量）", isOn: $draft.usePotionSemantic)
                        if draft.usePotionSemantic {
                            pathField("potion 模型目录", text: $draft.potionModelPath, chooseDirectory: true)
                        }
                        Slider(value: $draft.semanticThreshold, in: 0.15...0.95, step: 0.05)
                        Text("语义阈值：\(String(format: "%.2f", draft.semanticThreshold))，下限 \(String(format: "%.2f", draft.semanticFloor))。potion 不可用时自动回退 ColBERT 词向量/词法相似度。")
                            .font(.caption2).foregroundStyle(.secondary)
                        Picker("残差量化", selection: $draft.quantBits) {
                            Text("4 bit（默认）").tag(4)
                            Text("2 bit（更省空间）").tag(2)
                        }
                        .pickerStyle(.segmented)
                    }
                    DisclosureGroup("决策判定") {
                        Picker("检索方式", selection: $draft.searchMode) {
                            Text("向量检索（默认）").tag("vector")
                            Text("决策判定（BM25 + 决策模型）").tag("decision")
                        }
                        pathField("决策模型目录", text: $draft.decisionModelPath, chooseDirectory: true)
                        Stepper("BM25 召回切片数 \(draft.decisionRecallCount)",
                                value: $draft.decisionRecallCount, in: 10...1000, step: 10)
                        Stepper("判定批大小 \(draft.decisionBatchSize)",
                                value: $draft.decisionBatchSize, in: 1...32, step: 1)
                        Stepper("核心句每句最大字数 \(draft.decisionChunkChars)",
                                value: $draft.decisionChunkChars, in: 80...1200, step: 40)
                        Stepper("核心句数量 \(draft.decisionCoreSentences)",
                                value: $draft.decisionCoreSentences, in: 1...10, step: 1)
                        Text("决策模型默认 karpo-decision-0.8B-mlx-int4（需含 readout.safetensors）。切片先经字符级 BM25 提取核心句再判定；句数/字数越大越准确但更慢，批大小越大越快但占用更多内存。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    DisclosureGroup("检索 / 阅读") {
                        Stepper("返回分片数 \(draft.resultTopN)", value: $draft.resultTopN, in: 1...50)
                        Stepper("粗筛数量 \(draft.coarseCount)", value: $draft.coarseCount, in: 5...200, step: 5)
                        Slider(value: Binding(
                            get: { draft.effectiveMinScoreRatio },
                            set: { draft.minScoreRatio = $0 }), in: 0.5...1.0, step: 0.01)
                        Text("最低分比例：\(String(format: "%.2f", draft.effectiveMinScoreRatio)) × 最高分（至少保留 1 片）")
                            .font(.caption2).foregroundStyle(.secondary)
                        Toggle("大库使用 ANN 粗筛", isOn: $draft.useANN)
                        Stepper("ANN 簇数 \(draft.annClusters)", value: $draft.annClusters,
                                in: 32...2048, step: 64)
                        Stepper("nprobe \(draft.nprobe)", value: $draft.nprobe, in: 1...256, step: 4)
                        Stepper("rescore \(draft.rescore)", value: $draft.rescore, in: 16...512, step: 16)
                    }
                }
            }
            .formStyle(.grouped)

            Divider()
            HStack {
                Text("配置保存在 ~/Library/Application Support/DocSearch/config.json")
                    .font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Button("取消") { dismiss() }
                Button("保存并应用") {
                    if draft.chunkMinChars > draft.chunkMaxChars {
                        draft.chunkMinChars = max(2500, draft.chunkMaxChars - 100)
                    }
                    apply(draft)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(14)
        }
    }

    private func pathField(_ title: String, text: Binding<String>, chooseDirectory: Bool) -> some View {
        HStack {
            Text(title).frame(width: 130, alignment: .leading)
            TextField(title, text: text).textFieldStyle(.roundedBorder)
            Button("…") {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = chooseDirectory
                panel.canChooseFiles = !chooseDirectory
                panel.allowsMultipleSelection = false
                if panel.runModal() == .OK, let url = panel.url {
                    text.wrappedValue = url.path
                }
            }
            .frame(width: 30)
        }
    }
}
