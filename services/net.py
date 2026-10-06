"""本地网络访问工具 —— 绕开系统代理对 localhost 的劫持。

背景：Windows 上若配置了系统代理（如 Clash 的 127.0.0.1:7897），httpx 在
trust_env=True 时会通过 urllib.request.getproxies() 读到它，并把发往
localhost / 127.0.0.1 的请求也代理出去，导致 Ollama 返回 502。

本模块在导入时把 localhost 加入 NO_PROXY，保证所有基于 httpx 的客户端
（openai SDK / langchain ChatOpenAI / OpenAIEmbeddings）直连本地服务。
"""
from __future__ import annotations

import os

# 需要直连的主机名，覆盖 IPv4 / IPv6 回环与常见别名
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

# 标记是否已执行过，避免重复拼接
_applied = False


def ensure_localhost_bypass() -> None:
    """
    函数作用：
        确保 NO_PROXY 包含本地回环地址，使 httpx 不把本地请求交给系统代理。
        保留用户已有的 NO_PROXY 配置，仅追加缺失项。
    输入参数：
        - 无
    输出参数：
        - 无
    """
    global _applied
    if _applied:
        return

    for var in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.getenv(var, "").split(",") if item.strip()]
        lowered = {item.lower() for item in existing}
        for host in _LOCAL_HOSTS:
            if host.lower() not in lowered:
                existing.append(host)
        os.environ[var] = ",".join(existing)

    _applied = True
