"""Embedding 客户端工厂 —— 统一走 Ollama 的 OpenAI 兼容接口。

检索（法条向量）、长期记忆、离线建索引三处共用同一份配置，
避免各自加载本地 sentence-transformers 模型。

配置项：
- EMBEDDING_BASE_URL  默认 http://localhost:11434/v1
- EMBEDDING_MODEL     默认 bge-m3:latest（1024 维）
- EMBEDDING_API_KEY   默认 ollama（Ollama 不校验，占位即可）
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from services.net import ensure_localhost_bypass

# 本地 Ollama 不能被系统代理劫持，必须在创建 httpx 客户端前生效
ensure_localhost_bypass()

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "bge-m3:latest"


def _build_embeddings() -> OpenAIEmbeddings:
    """
    函数作用：
        按环境变量构造 OpenAIEmbeddings 客户端。
    输入参数：
        - 无
    输出参数：
        - OpenAIEmbeddings
    """
    base_url = os.getenv("EMBEDDING_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("EMBEDDING_API_KEY") or "ollama"

    return OpenAIEmbeddings(
        base_url=base_url,
        model=model,
        api_key=api_key,
        # 必须为 False：否则 LangChain 会用 tiktoken 把文本编码成 token 数组，
        # 而 Ollama 的 /v1/embeddings 只接受字符串输入。
        check_embedding_ctx_length=False,
        chunk_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAIEmbeddings:
    """
    函数作用：
        获取 embedding 客户端单例。
    输入参数：
        - 无
    输出参数：
        - OpenAIEmbeddings
    """
    return _build_embeddings()


def embed_query(text: str) -> list[float]:
    """
    函数作用：
        将单条查询文本编码为向量。
    输入参数：
        - text: str
    输出参数：
        - list[float]
    """
    return get_embedding_client().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    函数作用：
        批量编码文档文本为向量。
    输入参数：
        - texts: list[str]
    输出参数：
        - list[list[float]]
    """
    if not texts:
        return []
    return get_embedding_client().embed_documents(texts)


def reset_embedding_client() -> None:
    """
    函数作用：
        清空客户端缓存（仅供测试或切换配置后使用）。
    输入参数：
        - 无
    输出参数：
        - 无
    """
    get_embedding_client.cache_clear()
