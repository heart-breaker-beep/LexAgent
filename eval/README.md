# RAGAS 自动评测系统 — 实施计划

## Context

项目 RAG + Agent 功能已完成，需要建立自动化评测体系来量化检索质量和回答质量。使用 RAGAS 框架，评测数据集包含 100 条法律场景案例。

评测分两层：
1. **检索评测** — 法条是否检索对了（context precision/recall）
2. **端到端评测** — 最终回答是否正确、是否忠实于检索到的法条（faithfulness/correctness）

## 整体方案

### 1. 测试数据集格式

`eval/dataset.json`，每条数据包含：

```json
{
  "question": "房东不退押金怎么办？",
  "ground_truth": "承租人可以依据民法典要求房东返还押金...",
  "ground_truth_contexts": [
    "民法典_第七百一十四条",
    "民法典_第五百七十七条"
  ],
  "acceptable_contexts": [
    "民法典_第七百一十四条",
    "民法典_第五百七十七条"
  ],
  "corpus_status": "in_corpus"
}
```

字段说明：
- `question`: 用户问题（口语化法律场景）
- `ground_truth`: 标准答案（期望的回答要点）
- `ground_truth_contexts`: 相关法条的 chunk_id 列表（格式：`{law_name}_{article_no}`，与 `LawChunk.chunk_id` 对应）
- `acceptable_contexts`: 检索评测可接受命中的 chunk_id。用于“标准法条不是唯一合理答案”的场景
- `corpus_status`: `in_corpus` 或 `out_of_corpus`。后者会在本地 RAG 汇总指标中跳过，避免把语料缺失算成检索失败

### 2. ChatGPT 生成 Prompt

`eval/generate_prompt.md` 提供精心设计的 prompt，用户复制到 ChatGPT 可生成候选案例。写入 `dataset.json` 前必须用本地 chunk 列表校验所有 `ground_truth_contexts` 和 `acceptable_contexts`。

### 3. 评测脚本

**A. 检索评测（不需要 LLM judge）**

直接调用 `HybridRetriever.retrieve()`，对比返回的 chunk_id 与 ground_truth_contexts：
- **Hit Rate**: 至少命中一条相关法条的比例
- **MRR**: 第一条相关法条的排名倒数
- **Context Recall**: 检索到的相关法条数 / 标准答案中的法条总数
- **Context Precision**: 检索到的相关法条数 / 检索返回的总数

**B. 端到端评测（需要 LLM judge）**

调用完整的 LangGraph Agent，获取最终回答，用 RAGAS LLM-as-judge 评估：
- **Faithfulness**: 回答是否忠实于检索到的上下文
- **Answer Relevancy**: 回答是否切题
- **Answer Correctness**: 回答与标准答案的语义一致性

### 4. 文件结构

```
eval/
├── README.md               # 本文件
├── generate_prompt.md      # 给 ChatGPT 的生成 prompt
├── dataset.json            # 100 条测试数据
├── run_eval.py             # 评测主脚本
├── metrics.py              # 指标计算逻辑
└── results/                # 评测结果输出目录
```

### 5. 运行方式

```bash
# 仅检索评测（快速，不需要 LLM judge）
python eval/run_eval.py --mode retrieval

# 限制条数，适合 smoke test
python eval/run_eval.py --mode retrieval --limit 10

# 端到端评测（需要 MCP Server + RAGAS + 已配置的主 LLM API Key）
python eval/run_eval.py --mode e2e

# 全部评测
python eval/run_eval.py --mode all
```

