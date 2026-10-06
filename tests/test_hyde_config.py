from __future__ import annotations

import pytest

from services.retriever.hyde import _get_enhance_llm


def test_hyde_requires_explicit_model_and_base_url(monkeypatch):
    """未配置时显式失败，而不是静默回退到某个厂商的云端默认值。"""
    monkeypatch.delenv("HYDE_MODEL", raising=False)
    monkeypatch.delenv("HYDE_LLM_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="HYDE_MODEL"):
        _get_enhance_llm()


def test_hyde_requires_api_key_for_non_local_endpoint(monkeypatch):
    """远端 endpoint 缺少 API key 时显式失败，不再使用占位假 key。"""
    monkeypatch.setenv("HYDE_MODEL", "glm-4.6v")
    monkeypatch.setenv("HYDE_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.delenv("HYDE_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ZHIPU_API_KEY"):
        _get_enhance_llm()


def test_hyde_can_still_be_switched_back_to_local_ollama_qwen(monkeypatch):
    monkeypatch.setenv("HYDE_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("HYDE_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("HYDE_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)

    llm = _get_enhance_llm()

    assert llm.model_name == "qwen2.5:1.5b"
    assert llm.openai_api_base == "http://localhost:11434/v1"
    assert llm.openai_api_key.get_secret_value() == "ollama"
