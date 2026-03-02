"""Centralized logging configuration.

Configures loguru as the single logging backend for both application code
and uvicorn, with consistent formatting and file rotation.
"""
import sys

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"


def setup_logging() -> None:
    """Configure loguru"""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    logger.add("logs/api.log", level="DEBUG", rotation="100 MB", retention="30 days", format=LOG_FORMAT)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log validation errors before returning the standard 422 response."""
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error["loc"])
        logger.warning(f"Validation error at [{location}]: {error['msg']} (input: {error.get('input', 'N/A')!r})")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
