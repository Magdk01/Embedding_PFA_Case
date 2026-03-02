"""FastAPI application factory — creates, configures, and wires up the app."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from embedding_pfa_case.logging import setup_logging, validation_exception_handler
from embedding_pfa_case.model import EmbeddingModel

setup_logging()

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


# Create the FastAPI app
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

# Register exception handlers before route handlers
app.exception_handler(RequestValidationError)(validation_exception_handler)

# Register route handlers after app is created
from embedding_pfa_case.api import register_routes  # noqa: E402

register_routes(app)
