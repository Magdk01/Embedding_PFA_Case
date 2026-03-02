FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface

COPY uv.lock pyproject.toml README.md LICENSE ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src src/

RUN uv sync --frozen --no-dev

RUN uv run python -c "\
from transformers import AutoModel, AutoTokenizer; \
AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large'); \
AutoModel.from_pretrained('intfloat/multilingual-e5-large'); \
"

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn", "embedding_pfa_case.app:app", "--host", "0.0.0.0", "--port", "8000"]
