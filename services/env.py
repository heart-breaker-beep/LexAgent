"""环境变量读取工具 —— 缺失即显式失败。

背景：模型/provider 这类配置一旦用了硬编码默认值，变量名写错、漏配或复制
.env.example 时漏行，都会静默回退到另一个厂商的模型，最终表现为难以定位的
401 / 404，而不是「配置缺失」。本模块让缺失的配置在调用点立即抛出带指引的
错误。
"""
from __future__ import annotations

import os


def require_env(*names: str) -> str:
    """
    函数作用：
        按顺序返回第一个已设置且非空的环境变量；全部缺失时抛出带指引的
        RuntimeError。传入多个名称即构成回退链。
    输入参数：
        - names: str，一个或多个环境变量名，按优先级排列
    输出参数：
        - str
    """
    if not names:
        raise ValueError("require_env() 至少需要一个环境变量名")

    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()

    joined = " 或 ".join(names)
    raise RuntimeError(
        f"缺少必需的环境变量 {joined}。请在项目根目录的 .env 中配置后再启动服务"
        f"（可参考 .env.example）。"
    )
