"""查询增强模块 —— 问题重写 + HyDE 假设文档生成。

两阶段查询增强策略：
1. 问题重写：将用户口语化表达改写为精确的法律检索 query
2. HyDE：生成假设性法条文本，用于语义检索的 embedding

两者先各自生成增强文本，再进入多路召回：
- 重写后的 query → BM25 关键词检索 + 语义向量检索
- HyDE 假设文档 → 语义向量检索 + BM25 关键词检索
- 原始 query → BM25 关键词检索 + 语义向量检索
- 原始 query → Reranker 精排（仅 RERANKER_ENABLED=true 时）
"""
from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

from services.env import require_env
from services.net import ensure_localhost_bypass

# HyDE 可能指向本地 Ollama，同样不能被系统代理劫持
ensure_localhost_bypass()

log = logging.getLogger("legal.query_enhance")

_REWRITE_PROMPT = (
    "你是一个法律检索查询优化器。将用户的口语化法律问题改写为适合检索的精确查询。\n"
    "要求：\n"
    "- 提取核心法律概念和关键词\n"
    "- 补充隐含的法律术语\n"
    "- 去除口语化表达、语气词\n"
    "- 输出 1-2 句精炼的检索 query\n"
    "- 直接输出改写结果，不要解释\n\n"
    "用户问题：{query}"
)

_HYDE_PROMPT = (
    "你是一个中国法律条文生成器。根据用户的法律问题，"
    "生成一段 50-100 字的假设性法律条文原文，"
    "模拟真实法条的语言风格和结构。"
    "不需要真实存在，只需要语义上与问题高度相关。"
    "直接输出法条文本，不要加任何解释。\n\n"
    "用户问题：{query}"
)


def _get_enhance_llm() -> ChatOpenAI:
    """
    函数作用：
        获取查询增强专用的轻量 LLM 实例。
    输入参数：
        - 无
    输出参数：
        - ChatOpenAI
    """
    model = require_env("HYDE_MODEL")
    base_url = require_env("HYDE_LLM_BASE_URL")
    api_key = _get_hyde_api_key(base_url)

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=0.7,
        streaming=False,
        max_tokens=150,
    )


def _get_hyde_api_key(base_url: str) -> str:
    """获取 HyDE / query rewrite 使用的 API key。"""
    explicit = os.getenv("HYDE_API_KEY")
    if explicit:
        return explicit
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "ollama"
    return require_env("ZHIPU_API_KEY")


def is_query_enhance_enabled() -> bool:
    """
    函数作用：
        检查查询增强是否启用。
    输入参数：
        - 无
    输出参数：
        - bool
    """
    return os.getenv("HYDE_ENABLED", "true").lower() in ("true", "1", "yes")


def rewrite_query(query: str) -> str:
    """
    函数作用：
        将用户口语化问题改写为精确的法律检索 query。
    输入参数：
        - query: str
    输出参数：
        - str
    """
    if os.getenv("HYDE_REWRITE_ENABLED", "true").lower() not in ("true", "1", "yes"):
        return query

    try:
        llm = _get_enhance_llm()
        prompt = _REWRITE_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        log.debug("问题重写: %s -> %s", query, rewritten)
        return rewritten
    except Exception as e:
        log.warning("问题重写失败，回退到原始 query: %s", e)
        return query


def generate_hypothetical_doc(query: str) -> str:
    """
    函数作用：
        根据用户问题生成假设性法条文本，用于语义检索。
    输入参数：
        - query: str
    输出参数：
        - str
    """
    try:
        llm = _get_enhance_llm()
        prompt = _HYDE_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        hypothetical = response.content.strip()
        log.debug("HyDE 假设文档: %s -> %s", query, hypothetical[:80])
        return hypothetical
    except Exception as e:
        log.warning("HyDE 生成失败，回退到原始 query: %s", e)
        return query
