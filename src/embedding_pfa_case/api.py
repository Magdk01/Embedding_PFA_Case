import sys
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from embedding_pfa_case.model import EmbeddingModel

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
logger.add("logs/api.log", level="DEBUG", rotation="100 MB", retention="30 days")

ENGLISH_DANISH_PATTERN = r"^[A-Za-zÆØÅæøå0-9\s.,;!?:'\-]+$"

# We assign max length based on the token truncation
EmbedText = Annotated[
    str,
    Field(min_length=1, max_length=512, pattern=ENGLISH_DANISH_PATTERN),
]

embedding_model: EmbeddingModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup, clean up on shutdown."""
    global embedding_model
    logger.info("Starting up — loading embedding model...")
    embedding_model = EmbeddingModel()
    logger.info("Model ready, accepting requests")
    yield
    logger.info("Shutting down")


class EmbedRequest(BaseModel):
    """Request body for the /embed endpoint."""

    texts: list[EmbedText] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Texts to embed (1–64 items, each 1–512 chars, English/Danish alphanumeric).",
        examples=[["how much protein should a female eat", "hvad er vejret i dag"]],
    )
    input_type: Literal["query", "passage"] = Field(
        default="query",
        description=(
            "Prefix type per the e5 convention. "
            "Use 'query' for questions, symmetric similarity, or feature extraction. "
            "Use 'passage' for documents in asymmetric retrieval."
        ),
    )


class EmbedResponse(BaseModel):
    """Response body from the /embed endpoint."""

    embeddings: list[list[float]] = Field(description="Normalized embedding vectors (1024-dim each).")
    num_texts: int = Field(description="Number of texts that were embedded.")
    dim: int = Field(description="Dimensionality of each embedding vector.")


app = FastAPI(
    title="Embedding API — multilingual-e5-large",
    description=(
        "API for generating text embeddings using "
        "[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large). "
        "Supports English and Danish text with automatic query/passage prefixing."
    ),
    version="0.0.1",
    lifespan=lifespan,
)


@app.get("/health", summary="Health check")
async def health() -> dict:
    """Check whether the API is up and the model is loaded."""
    return {"status": "ok", "model_loaded": embedding_model is not None}


@app.post("/embed", response_model=EmbedResponse, summary="Generate text embeddings")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate normalized embeddings for the provided texts.

    The model automatically prepends the appropriate prefix based on `input_type`.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    logger.info(f"Embed request: {len(request.texts)} text(s), input_type={request.input_type}")

    try:
        embeddings = embedding_model.embed(request.texts, prefix=request.input_type)
    except Exception as e:
        logger.error(f"Embedding inference failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding inference failed") from e

    logger.info(f"Embed response: {len(embeddings)} embedding(s), dim={len(embeddings[0])}")

    return EmbedResponse(
        embeddings=embeddings,
        num_texts=len(embeddings),
        dim=len(embeddings[0]),
    )
