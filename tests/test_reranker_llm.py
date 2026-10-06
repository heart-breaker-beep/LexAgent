"""LLM 精排器测试 —— 锁住相关性判别行为与输出解析。

背景：LLM 精排的效果高度依赖 prompt 措辞。实测若把 _RERANK_PROMPT 改成
“直接相关、可用于回答问题”这类严格要求，3B 级模型会过度拒绝，
导致相关 query 召回全空。这里的端到端用例就是防这次回归的。

运行方式：
    pytest tests/test_reranker_llm.py -v
需要本地 Ollama 已启动并拉取 embedding + LLM 模型，否则自动跳过。
"""
from __future__ import annotations

import os

import pytest

from services.retriever.reranker import LLMReranker

# 相关问题：期望精排后仍有结果（召回不能丢）
RELEVANT_QUERIES = [
    "劳动合同解除需要什么条件",
    "试用期最长多久",
]

# 无关问题：期望精排后无结果（否则 low_quality 联网兜底失效）
IRRELEVANT_QUERIES = [
    "番茄炒蛋怎么做才好吃",
    "推荐一款笔记本电脑",
]


def _ollama_available() -> bool:
    """探测本地 Ollama 是否可用。"""
    import httpx

    base = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    root = base.split("/v1")[0]
    try:
        return httpx.get(f"{root}/api/tags", timeout=3, trust_env=False).status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="本地 Ollama 未启动或不可达"
)


class TestParseIndices:
    """输出解析单元测试（不依赖 LLM）。"""

    def test_parses_plain_array(self):
        """标准 JSON 数组应被正确解析。"""
        assert LLMReranker._parse_indices("[0,2]", 5) == {0, 2}

    def test_parses_array_with_markdown_fence(self):
        """被 markdown 代码块包裹时也应能抠出数组。"""
        raw = "```json\n[1, 3]\n```"
        assert LLMReranker._parse_indices(raw, 5) == {1, 3}

    def test_parses_empty_array(self):
        """空数组表示没有相关法条。"""
        assert LLMReranker._parse_indices("[]", 5) == set()

    def test_out_of_range_indices_dropped(self):
        """越界编号应被丢弃，避免索引错位。"""
        assert LLMReranker._parse_indices("[0,9]", 5) == {0}

    def test_unparseable_returns_none(self):
        """无法解析时返回 None，触发 fail-open 分支。"""
        assert LLMReranker._parse_indices("没有相关法条", 5) is None
        assert LLMReranker._parse_indices("", 5) is None


@pytest.fixture(scope="module")
def retriever():
    """构建带 LLM 精排的混合检索器（整个模块复用，避免重复建索引）。"""
    from services.indexer.chunker import chunk_all_laws
    from services.retriever import init_retriever

    chunks = chunk_all_laws("data/laws")
    if not chunks:
        pytest.skip("data/laws 下没有法律文本")
    return init_retriever(chunks)


@requires_ollama
@pytest.mark.slow
class TestLLMRerankerDiscrimination:
    """端到端相关性判别 —— prompt 措辞回归防线。"""

    def test_relevant_queries_keep_recall(self, retriever):
        """相关 query 精排后必须仍有结果，不能被过度拒绝。"""
        empty = [q for q in RELEVANT_QUERIES if not retriever.retrieve(q, top_k=5)]

        assert not empty, (
            f"以下相关 query 精排后召回为空，说明 _RERANK_PROMPT 措辞过严: {empty}"
        )

    def test_irrelevant_queries_are_rejected(self, retriever):
        """无关 query 应被拒绝，保证 low_quality 联网兜底可用。"""
        passed = [q for q in IRRELEVANT_QUERIES if retriever.retrieve(q, top_k=5)]

        assert not passed, (
            f"以下无关 query 未被精排拒绝，相关性门控失效: {passed}"
        )
