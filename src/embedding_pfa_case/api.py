"""API route handlers and request/response schemas."""

from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

ENGLISH_DANISH_PATTERN = r"^[A-Za-zÆØÅæøå0-9\s.,;!?:'\-]+$"

EmbedText = Annotated[
    str,
    Field(min_length=1, max_length=512, pattern=ENGLISH_DANISH_PATTERN),
]


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


class SimilarityRequest(BaseModel):
    """Request body for the /similarity endpoint."""

    queries: list[EmbedText] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Query texts (1–64 items). Automatically prefixed with 'query: '.",
        examples=[["how much protein should a female eat", "What is the capital of denmark?"]],
    )
    passages: list[EmbedText] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Passage texts (1–64 items). Automatically prefixed with 'passage: '.",
        examples=[["The CDC recommends 46 grams of protein per day for women.", "Copenhagen", "Stockholm"]],
    )


class SimilarityResponse(BaseModel):
    """Response body from the /similarity endpoint."""

    scores: list[list[float]] = Field(
        description="Similarity matrix: scores[i][j] = cosine_similarity(query_i, passage_j) * 100."
    )
    num_queries: int = Field(description="Number of queries.")
    num_passages: int = Field(description="Number of passages.")


def register_routes(app: FastAPI) -> None:
    """Register all API route handlers on the given app."""

    @app.get("/health", summary="Health check")
    async def health() -> dict:
        """Check whether the API is up and the model is loaded."""
        from embedding_pfa_case.app import embedding_model

        return {"status": "ok", "model_loaded": embedding_model is not None}

    @app.post("/embed", response_model=EmbedResponse, summary="Generate text embeddings")
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """Generate normalized embeddings for the provided texts.

        The model automatically prepends the appropriate prefix based on `input_type`.
        """
        from embedding_pfa_case.app import embedding_model

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

    @app.post("/similarity", response_model=SimilarityResponse, summary="Compute query-passage similarity")
    async def similarity(request: SimilarityRequest) -> SimilarityResponse:
        """Compute cosine similarity between queries and passages.

        Returns a score matrix where scores[i][j] is the similarity between
        query i and passage j, scaled by 100.
        """
        from embedding_pfa_case.app import embedding_model

        if embedding_model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

        logger.info(f"Similarity request: {len(request.queries)} query(s) x {len(request.passages)} passage(s)")

        try:
            scores = embedding_model.similarity(request.queries, request.passages)
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise HTTPException(status_code=500, detail="Similarity computation failed") from e

        logger.info(f"Similarity response: {len(scores)}x{len(scores[0])} score matrix")

        return SimilarityResponse(
            scores=scores,
            num_queries=len(request.queries),
            num_passages=len(request.passages),
        )
