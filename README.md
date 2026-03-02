# Embedding API — multilingual-e5-large

A production-ready API for generating text embeddings using Microsoft's
[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) model.
Built with FastAPI, managed with [uv](https://docs.astral.sh/uv/).

## Quick start

**Prerequisites:** Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
# 1. Install dependencies
uv sync --dev

# 2. Start the API
uv run invoke serve
```

The API will be available at `http://127.0.0.1:8001`.
Swagger docs are served at `http://127.0.0.1:8001/docs`.

> **First run:** The model (~2.2 GB) is downloaded from HuggingFace on the first startup.
> This is a one-time download — subsequent starts use the cached model.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check — confirms the model is loaded |
| `POST` | `/embed` | Generate embeddings for a list of texts |
| `POST` | `/similarity` | Compute query–passage similarity scores |

### Example: embed texts

```bash
curl -X POST http://127.0.0.1:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["how much protein should a female eat"], "input_type": "query"}'
```

### Example: compute similarity

```bash
curl -X POST http://127.0.0.1:8001/similarity \
  -H "Content-Type: application/json" \
  -d '{"queries": ["best protein sources"], "passages": ["Chicken breast contains 31g of protein per 100g."]}'
```

## Development

All common tasks are available via `invoke`:

```bash
uv run invoke --list       # Show all available tasks
uv run invoke test         # Run tests
uv run invoke check        # Run formatting, linting, type checks, and tests
uv run invoke serve        # Start the API server
```

## Docker
Project supports Docker, for reproducibility as well as containerization if needed for cloud deployment.

### Quick start
First ensure Docker Engine is running.
```bash
uv run invoke docker_build
uv run invoke docker_run
```
API is then served at **http://localhost:8000** (Swagger UI: http://localhost:8000/docs).

### Image name and tags
- **Default image name:** `embedding-api`. The invoke tasks build and run this tag by default.
- **List the image:** `docker images embedding-api`
- **Custom tag (e.g. for a version):**
  ```bash
  uv run invoke docker_build --tag embedding-api:v0.1.0
  uv run invoke docker_run --tag embedding-api:v0.1.0
  ```
- **Custom port:** `uv run invoke docker_run --port 9000` (maps host 9000 → container 8000)
- **Run without invoke:** `docker run --rm -p 8000:8000 embedding-api`

### Docker Hub
Successful release images are available and named according to their tags at:
[Docker Hub — magdk01/embedding-api](https://hub.docker.com/repository/docker/magdk01/embedding-api/general)


## Project structure

```
├── .github/                      # CI workflows and Dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── release.yaml
├── dockerfiles/
│   └── api.dockerfile
├── src/embedding_pfa_case/       # Source code
│   ├── api.py                    #   FastAPI application
│   └── model.py                  #   EmbeddingModel wrapper
├── tests/
│   ├── api/
│   │   └── test_api.py           #   API endpoint tests
│   └── model/
│       └── test_model.py         #   Model unit tests
├── pyproject.toml
├── tasks.py                      # Invoke task definitions
└── uv.lock
```
