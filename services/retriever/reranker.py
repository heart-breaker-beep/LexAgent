"""Rerank 精排器 —— 对粗排候选做相关性判别。

提供两种后端（由 RERANKER_BACKEND 选择）：
- cross_encoder：本地 bge-reranker-base（需 models/ 目录，精度最高）
- llm：调用 Ollama 的 LLM 从候选中挑出相关条目（默认，无需本地模型）

两者接口一致：rerank(query, chunks, top_n) -> list[tuple[LawChunk, float]]，
分数语义也一致（相关≈1.0 / 不相关≈0.0），因此阈值过滤逻辑可共用。
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Protocol

from services.vectorstore.base import LawChunk

log = logging.getLogger("legal.reranker")


class RerankerBackend(Protocol):
    """精排后端接口 —— Cross-Encoder 与 LLM 两种实现共用。"""

    def rerank(
        self, query: str, chunks: list[LawChunk], top_n: Optional[int] = None
    ) -> list[tuple[LawChunk, float]]:
        ...


def is_rerank_enabled() -> bool:
    """
    函数作用：
        读取 RERANKER_ENABLED，判断是否启用精排。
    输入参数：
        - 无
    输出参数：
        - bool
    """
    return os.getenv("RERANKER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


# 模块级单例
_reranker_model = None


def _get_reranker():
    """
    函数作用：
        延迟加载 reranker 模型（单例）。
    输入参数：
        - 无
    输出参数：
        - 未标注
    """
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        model_name = os.getenv("RERANKER_MODEL", "models/bge-reranker-base")
        model_path = Path(model_name)
        if model_path.exists():
            model_name = str(model_path.resolve())
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


class CrossEncoderReranker:
    """Cross-Encoder 精排器。

    对粗排阶段产出的候选法条进行精细化重排序，
    输出最终的 top_n 高质量结果。需要本地 bge-reranker-base 模型。
    """

    def __init__(self, top_n: Optional[int] = None):
        """
        函数作用：
            初始化精排器。
        输入参数：
            - top_n: Optional[int]，默认值 None
        输出参数：
            - 未标注
        """
        self._top_n = top_n or int(os.getenv("RERANKER_TOP_N", "5"))

    def rerank(
        self, query: str, chunks: list[LawChunk], top_n: Optional[int] = None
    ) -> list[tuple[LawChunk, float]]:
        """
        函数作用：
            对候选法条进行精排。
        输入参数：
            - query: str
            - chunks: list[LawChunk]
            - top_n: Optional[int]，默认值 None
        输出参数：
            - list[tuple[LawChunk, float]]
        """
        if not chunks:
            return []

        n = top_n or self._top_n
        reranker = _get_reranker()

        # 构造 (query, document) 对
        pairs = [(query, chunk.content) for chunk in chunks]

        # Cross-Encoder 打分
        scores = reranker.predict(pairs)

        # 按分数降序排列
        scored_chunks = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )

        return [(chunk, float(score)) for chunk, score in scored_chunks[:n]]


# 措辞针对 qwen2.5:7b 调优，实测相关保留 5/5、无关拒绝 5/5。
#
# 两个已知的踩坑方向，改动前务必重跑 tests/test_reranker_llm.py：
# - 措辞过严（如“只选直接相关、可用于回答问题的”）：小模型会过度拒绝，
#   连「试用期最长多久」配劳动法第二十一条都判不相关，相关 query 召回全空；
# - 措辞过松（如“只要主题沾边就算有关”）：7b 会把「番茄炒蛋怎么做」这类
#   生活常识问题也判为相关，low_quality 联网兜底失效。
# 最后一句的“与法律无关则返回 []”就是给模型一个明确的拒绝出口。
_RERANK_PROMPT = (
    "判断下列法条中哪些与用户的问题属于同一法律主题、可用于回答该问题。\n"
    "若用户问题是生活常识、闲聊、或与法律无关，则一律返回 []。\n\n"
    "用户问题：{query}\n\n"
    "候选法条：\n{candidates}\n\n"
    "只输出编号数组，如 [0,2]。没有任何相关法条时输出 []。\n"
    "不要输出解释或 markdown 代码块。\n"
)

# 从 LLM 输出里抠出第一个 JSON 数组
_JSON_ARRAY_RE = re.compile(r"\[[\s\d,]*\]")


class LLMReranker:
    """基于 LLM 的相关性精排器。

    单次调用让 LLM 从候选中挑出相关条目，避免逐条打分带来 N 次调用的开销。
    分数为二值（相关 1.0 / 不相关 0.0），与 Cross-Encoder 后端语义一致，
    因此 RERANKER_SCORE_THRESHOLD 可共用。
    """

    def __init__(self, top_n: Optional[int] = None):
        """
        函数作用：
            初始化 LLM 精排器。
        输入参数：
            - top_n: Optional[int]，默认值 None
        输出参数：
            - 未标注
        """
        self._top_n = top_n or int(os.getenv("RERANKER_TOP_N", "5"))
        self._max_candidates = int(os.getenv("RERANKER_LLM_MAX_CANDIDATES", "20"))

    def _build_prompt(self, query: str, chunks: list[LawChunk]) -> str:
        """
        函数作用：
            构造候选法条列表 prompt。
        输入参数：
            - query: str
            - chunks: list[LawChunk]
        输出参数：
            - str
        """
        lines = [
            f"[{i}] 《{c.law_name}》{c.article_no}：{c.content}"
            for i, c in enumerate(chunks)
        ]
        return _RERANK_PROMPT.format(query=query, candidates="\n".join(lines))

    @staticmethod
    def _parse_indices(raw: str, upper_bound: int) -> Optional[set[int]]:
        """
        函数作用：
            解析 LLM 输出中的相关编号数组，失败返回 None。
        输入参数：
            - raw: str
            - upper_bound: int
        输出参数：
            - Optional[set[int]]
        """
        match = _JSON_ARRAY_RE.search(raw or "")
        if not match:
            return None
        try:
            values = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        if not isinstance(values, list):
            return None
        return {int(v) for v in values if isinstance(v, (int, float)) and 0 <= int(v) < upper_bound}

    def rerank(
        self, query: str, chunks: list[LawChunk], top_n: Optional[int] = None
    ) -> list[tuple[LawChunk, float]]:
        """
        函数作用：
            调用 LLM 判别候选法条相关性，返回相关条目及其分数。
        输入参数：
            - query: str
            - chunks: list[LawChunk]
            - top_n: Optional[int]，默认值 None
        输出参数：
            - list[tuple[LawChunk, float]]
        """
        if not chunks:
            return []

        n = top_n or self._top_n
        candidates = chunks[: self._max_candidates]

        from services.llm import get_llm

        try:
            # 精排模型可独立配置：默认复用 LLM_PROVIDER/LLM_MODEL，
            # 需要时用 RERANKER_LLM_PROVIDER/MODEL 指向更擅长判别的模型
            llm = get_llm(
                os.getenv("RERANKER_LLM_PROVIDER") or None,
                model=os.getenv("RERANKER_LLM_MODEL") or None,
                temperature=0,
                streaming=False,
                timeout=float(os.getenv("RERANKER_LLM_TIMEOUT", "60")),
            )
            response = llm.invoke(self._build_prompt(query, candidates))
            raw = getattr(response, "content", "") or ""
        except Exception as exc:
            log.warning("LLM 精排调用失败，保留 RRF 顺序: %s", exc)
            return [(chunk, 1.0) for chunk in candidates[:n]]

        indices = self._parse_indices(raw, len(candidates))
        if indices is None:
            # 解析失败时 fail-open：保留 RRF 顺序，避免因模型输出格式问题丢掉全部召回
            log.warning("LLM 精排输出无法解析，保留 RRF 顺序: %r", raw[:200])
            return [(chunk, 1.0) for chunk in candidates[:n]]

        return [(candidates[i], 1.0) for i in sorted(indices)][:n]


def build_reranker() -> Optional[RerankerBackend]:
    """
    函数作用：
        按 RERANKER_BACKEND 构造精排器；RERANKER_ENABLED=false 时返回 None。
    输入参数：
        - 无
    输出参数：
        - Optional[RerankerBackend]
    """
    if not is_rerank_enabled():
        return None

    backend = os.getenv("RERANKER_BACKEND", "llm").strip().lower()
    if backend == "cross_encoder":
        return CrossEncoderReranker()
    if backend == "llm":
        return LLMReranker()
    raise ValueError(
        f"unknown RERANKER_BACKEND: {backend!r}, expected one of ['cross_encoder', 'llm']"
    )
