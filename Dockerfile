# 法智法律咨询 Agent —— 运行镜像
#
# 设计要点：
#  1. Ollama 不在容器内，运行在宿主机上（模型动辄几 GB，打进镜像不现实）。
#     容器通过 host.docker.internal 访问它，URL 由 compose 的 environment 注入。
#  2. 先装 CPU 版 torch：requirements 里的 sentence-transformers 会拖入 torch，
#     默认是 CUDA 构建（2.5GB+）。本项目 embedding 走 Ollama，CPU 版足够。
#     若构建时下载 download.pytorch.org 太慢，可删掉那一行，改用完整版（镜像会大 ~1.5GB）。
#  3. 法条索引在首次启动时自动构建（mcp_server/startup.py 的 load_or_build_index），
#     结果写入 /app/data/chroma_db，该目录通过卷持久化，重建容器不会丢。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# curl 供 HEALTHCHECK 使用；libgomp1 是 chromadb/onnxruntime 的运行时依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU 版 torch 先装，后续 pip 见到已满足的依赖会跳过
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

# 非 root 运行
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .

# 数据目录：首次创建命名卷时，Docker 会把镜像里的 data/ 复制进去，
# 因此 data/laws（法条语料）会自动带到卷里，索引得以构建。
VOLUME ["/app/data"]

USER appuser

EXPOSE 8000

# 首次启动要构建法条索引（需调用 Ollama 做 embedding），给足冷启动时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
