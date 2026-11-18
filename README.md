# DocSearch

macOS 上的本地文档检索与阅读工具：把 PDF / EPUB / Office / 纯文本等资料解析成 Markdown，切片后建立 **ColBERT 词向量 + 4bit 残差量化 + BM25 + vec0** 混合索引，用自然语言问句直接召回相关长文切片，并在阅读视图中高亮命中词。

整个链路离线运行，不依赖云端检索服务。

## 功能特性

- **多格式解析**：PDF（Apple Vision / OvisOCR3 / 混合）、EPUB、Office、纯文本；支持增量转换与取消。
- **语义切片**：优先按章节目录层级分节，小节尽量保持完整；结合 potion multilingual 句向量判断语义断点；默认切片 2500–3000 字、1.5% 重叠，表格与代码块保持完整。
- **ColBERT 编码**：查询与文档分别使用 `[QueryMarker]`(250002) / `[DocumentMarker]`(250003)；查询侧按 ColBERT 约定补 `<mask>` 增强；文档侧做词项剪枝后逐词编码并 L2 归一化。
- **异步入库 + 双检索路径**：同步时 Markdown 生成后立即切文本切片入库，**决策判定随即可用**；ColBERT 词向量/4bit 残差由后台异步读取切片计算填充，填完后向量检索才完全生效（GUI 侧栏显示回填进度，可停止）。配置可关闭向量检索模式——关闭后完全不计算词向量、不占向量空间，只保留文本切片供决策判定。
- **双检索路径**：主窗口左上角“工作区”提供二选一开关——**向量检索**（默认：ColBERT + BM25 混合召回 + 4bit 残差 MaxSim 精排）与**决策判定**（BM25 从原文切片表召回前 50 条 → 决策模型逐片判定相关性 → 取分数最高的 5 条）。关闭向量检索模式时自动走决策判定。
- **决策判定**：默认使用 `karpo-decision-0.8B-mlx-int4`（也可在设置里换成 4B）。每个召回切片先用**字符级 BM25**（字符 3-gram 滑动窗口、k1=1.6、b=0.4）在切片内逐句打分，取分数最高的 3 句按原文顺序拼成核心片段；再在 joint-preview 模板下为核心片段放 readout marker，读取 backbone final-norm hidden state 经 fp32 readout 头算出 P(相关)；一次前向批量判定多个切片。可配置召回数（默认 50）、批大小（默认 8）、每句最大字数（默认 160）与核心句数量（默认 3）。
- **检索阅读模式**：结果以气泡展示，点击打开阅读面板，前后翻片并显示查询词高亮。
- **实时日志窗口**：右上角“日志”按钮或 `⌘⇧L` 打开独立窗口，实时查看检索阶段耗时、同步进度与错误。
- **CLI 与 GUI 共用一套内核**：可命令行批处理，也可 `swift run DocSearch` 启动图形界面。

## 检索流程

```
问句 → 轻量剪枝(标点/空白/停用词) → [QueryMarker] + <mask> 增强 → ColBERT ONNX 编码
     → query 中心向量(vec0/ANN 粗排) ┐
     → BM25 倒排召回                 ├→ RRF → shortlist → 4bit 残差 MaxSim 重排
文档 → 解析/Markdown → 语义切片 → 词项剪枝 → ColBERT 词向量 → L2 归一化 → 4bit 残差量化
                                                                  → 最低分比例 → TopN
```

索引表：`document` / `document_fts`(BM25) / `doc_summary_vec`(vec0 cosine) / `doc_vectors` / `doc_residuals` / `ann_blocks` / `ann_cells`。

## 环境要求

- macOS 14+
- Xcode 15+（含 Command Line Tools；构建 MLX / Metal 相关依赖需要完整 Xcode 与 Metal Toolchain）
- Swift 5.9+

依赖通过 SwiftPM 自动下载：`swift-transformers`、`onnxruntime-swift-package-manager`、`mlx-swift-lm`、`ZIPFoundation`、`AEXML`、`SwiftSoup`、`swift-jinja`，以及 llama.cpp XCFramework。

## 构建与运行

```bash
./build.sh          # swift build -c release
./build-app.sh      # 生成 dist/DocSearch.app 与 dist/DocSearch.dmg

swift run -c release DocSearch          # 启动 GUI
```

CLI 支持 `--key` 与 `--key=value` 两种参数形式。每次检索返回前都会自动打印整体耗时及各阶段耗时，无需额外参数。

### 常用命令

```bash
# 从源目录同步（解析 → Markdown → 文本切片入库；词向量后台异步回填）
swift run -c release DocSearch sync --source /path/to/docs [--workdir /path/to/workdir] [--no-vector]

# 建库 / 追加单个文本文件
swift run -c release DocSearch init --db /path/to/doc_search.db
swift run -c release DocSearch add --db /path/to/doc_search.db --file /path/a.md [--title "标题"] [--text-only]

# 检索（向量检索：ColBERT + BM25 混合）
swift run -c release DocSearch search --db /path/to/doc_search.db \
  --query "街道名称如何改变城市记忆" --top 5 --coarse 20

# 决策判定检索（BM25 召回 50 片 → 决策模型判定 → Top5）
swift run -c release DocSearch decision-search --db /path/to/doc_search.db \
  --query "街道名称如何改变城市记忆" --top 5 --recall 50 --batch 8 --chunk-chars 160 --core 3

# 单独运行后台词向量回填（同步中断/重启后补算）
swift run -c release DocSearch vector-fill [--db PATH] [--model DIR] [--threads 2]

# 决策模型自检（对比 Python 参考实现的数值一致性）
swift run -c release DocSearch decision-verify --probe /path/to/probe.json

# 索引维护与诊断
swift run -c release DocSearch db-stats --db PATH
swift run -c release DocSearch ann-status --db PATH
swift run -c release DocSearch ann-build --db PATH --clusters 512
swift run -c release DocSearch ann-bench --rows 50000 --k 20

# 自检与模型检查
swift run -c release DocSearch self-test
swift run -c release DocSearch chunk-test
swift run -c release DocSearch models [--download]
```

## 配置

GUI 设置写入 `~/Library/Application Support/DocSearch/config.json`，CLI 也读取同一份配置（包括模型目录回退）。常用项：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `sourceDirectory` | 空 | 待同步文档目录 |
| `workDirectory` | `~/Library/Application Support/DocSearch` | 唯一工作目录；Markdown 输出到 `<工作目录>/markdown`，数据库为 `<工作目录>/db/doc_search.db` |
| `modelDirectory` | jina-colbert-v2-64-onnx-int8 所在目录 | ColBERT ONNX 模型，可用 `--model` 或 `DOCSEARCH_MODEL` 覆盖 |
| `searchMode` | `vector` | 检索路径：`vector` 向量检索 / `decision` 决策判定（GUI 左上角开关可切换） |
| `decisionModelPath` | karpo-decision-0.8B-mlx-int4 所在目录 | 决策判定模型，目录需含 `readout.safetensors`（4B 更准但慢约 4×） |
| `decisionRecallCount` / `decisionBatchSize` / `decisionChunkChars` / `decisionCoreSentences` | 50 / 8 / 160 / 3 | 决策模式 BM25 召回数 / 单批判定数 / 核心句每句最大字数 / 核心句数量 |
| `chunkMinChars` / `chunkMaxChars` | 2500 / 3000 | 切片字数区间；优先按目录层级切，小节尽量保持完整 |
| `chunkOverlapRatio` | 0.015 | 相邻切片重叠比例 |
| `quantBits` | 4 | 残差量化位数 |
| `resultTopN` / `coarseCount` | 5 / 20 | 返回数与粗筛数 |
| `minScoreRatio` | 0.9 | 只保留分数达到最高分该比例的结果 |
| `usePotionSemantic` | true | 使用 potion 句向量做语义切片 |
| `useVectorSearch` | true | 启用向量检索。关闭后同步只入库文本切片、不算词向量（省空间），检索自动走决策判定 |
| `vectorBackfillThreads` | 2 | 后台词向量回填线程数 |

修改切片或文档侧表示相关配置后，需要**重新同步知识库**才会反映到索引中；查询侧参数即时生效。

## 模型依赖

- **ColBERT**：`jina-colbert-v2-64-onnx-int8`（64 维），目录需含 `model_int8.onnx`、`tokenizer.json`、`config.json`。
- **语义切片**：`potion-multilingual-128M`（int4 ONNX），可在配置中指定目录或自动查找。
- **决策判定（可选）**：`karpo-decision-0.8B-mlx-int4`（默认，Qwen3.5-0.8B backbone + fp32 readout 头），另有 4B 版本可选（更准但 M1 上判定 50 片约 52s vs 0.8B 约 9s）；仅在“决策判定”检索路径用到，输出每个切片的相关概率，非生成式用途。
- **OCR（可选）**：OvisOCR3（MLX/GGUF/ONNX）用于扫描 PDF 与图片；默认 PDF 引擎为 Apple Vision。
- 停用词表位于 `Sources/DocSearch/Resources/stopwords_full.txt`。

## 目录结构

```
Sources/DocSearch/           GUI、CLI、检索内核、向量量化、ANN
Sources/DocSearch/Parsers/   PDF/EPUB/Office/OCR 解析与切片辅助
CPDFShim/ CSQLiteVec/        C 辅助模块（PDF 图片提取、sqlite-vec）
build.sh / build-app.sh      编译与打包脚本
dist/                        打包产物（gitignore）
```

## 自检

`self-test` 覆盖量化往返、vDSP MaxSim 对照、切片边界、SQLite 建表查询与 potion 语义信号；加 `--model DIR` 可跑端到端检索测试。

## 许可证

GPL-3.0，见 [LICENSE](LICENSE)。
