FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HOME=/app/.hf

# 1. Dependencies first (cached as long as pyproject/uv.lock don't change)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

# 2. The project itself
COPY src ./src
COPY config ./config
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

# 3. Bake the models into the image, so a cold start doesn't download them
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "paper_tutor.api:app", "--host", "0.0.0.0", "--port", "8000"]