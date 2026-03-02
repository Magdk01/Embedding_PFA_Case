FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY uv.lock pyproject.toml README.md LICENSE ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src src/

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "uvicorn", "embedding_pfa_case.app:app", "--host", "0.0.0.0", "--port", "8000"]
