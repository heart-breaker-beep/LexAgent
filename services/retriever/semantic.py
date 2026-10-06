"""语义检索器 —— 基于向量相似度的法条检索。

查询文本通过 services.embeddings 走 Ollama（bge-m3）编码为向量，
然后在向量库中检索最相似的法条分块。
"""
from __future__ import annotations

import os
from typing import Optional

from services.embeddings import embed_query
from services.vectorstore import get_vectorstore
from services.vectorstore.base import LawChunk


def _query_prefix() -> str:
    """
    函数作用：
        获取查询前缀。bge-small-zh 需要中文检索指令前缀，bge-m3 不需要，
        因此默认留空、由 EMBEDDING_QUERY_PREFIX 覆盖。
    输入参数：
        - 无
    输出参数：
        - str
    """
    return os.getenv("EMBEDDING_QUERY_PREFIX", "")


class SemanticRetriever:
    """语义检索器 —— 通过向量相似度检索法条。

    工作流程：
    1. 将用户查询编码为 embedding 向量
    2. 在向量库中执行近似最近邻搜索
    3. 返回相似度最高的 top_k 个法条分块
    """

    def __init__(self, vectorstore=None):
        """
        函数作用：
            初始化语义检索器。
        输入参数：
            - vectorstore: 未标注，默认值 None
        输出参数：
            - 未标注
        """
        self._store = vectorstore or get_vectorstore()

    def retrieve(
        self, query: str, top_k: int = 20
    ) -> list[tuple[LawChunk, float]]:
        """
        函数作用：
            语义检索法条。
        输入参数：
            - query: str
            - top_k: int，默认值 20
        输出参数：
            - list[tuple[LawChunk, float]]
        """
        prefix = _query_prefix()
        query_text = f"{prefix}{query}" if prefix else query
        embedding = embed_query(query_text)

        return self._store.search(embedding, top_k=top_k)
