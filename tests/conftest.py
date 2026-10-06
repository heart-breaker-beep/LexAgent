"""pytest 全局配置 —— 让测试与应用使用同一份 .env。

应用侧在 main.py / mcp_server/startup.py 里各自调用 load_dotenv()，
但 pytest 进程不走这两个入口。若不加载，HYDE_LLM_BASE_URL、LLM_PROVIDER
等必需配置会缺失，require_env() 直接抛错，测试无法运行。
"""
from __future__ import annotations

from dotenv import load_dotenv

# 项目根目录下的 .env；override=False 保证真实环境变量优先
load_dotenv(override=False)
