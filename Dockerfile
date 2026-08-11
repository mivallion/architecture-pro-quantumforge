FROM python:3.14-slim AS runtime

ARG EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY app/requirements.txt /tmp/requirements.txt
RUN python -m pip install --requirement /tmp/requirements.txt

# Bake the public embedding model into the image so runtime does not depend on
# Hugging Face availability and EMBEDDING_LOCAL_FILES_ONLY can remain enabled.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', truncate_dim=512)"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

COPY --chown=app:app app /workspace/app
COPY --chown=app:app Task3/index /workspace/Task3/index

RUN mkdir -p /workspace/Task7 \
    && chown -R app:app /workspace /opt/huggingface

ENV HF_HUB_OFFLINE=1 \
    EMBEDDING_LOCAL_FILES_ONLY=true \
    RAG_REPOSITORY_ROOT=/workspace \
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M \
    OLLAMA_KEEP_ALIVE=5m

USER app

CMD ["python", "-m", "app.bot"]
