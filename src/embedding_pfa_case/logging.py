"""Centralized logging configuration.

Configures loguru as the single logging backend for both application code
and uvicorn, with consistent formatting and file rotation.
"""
import logging
import sys

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"


class InterceptHandler(logging.Handler):
    """Send standard library logging (e.g. uvicorn) through loguru with the same format."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        message = f"{record.name} | {record.getMessage()}"
        logger.opt(depth=6, exception=record.exc_info).log(level, message)


def setup_logging() -> None:
    """Configure loguru and intercept uvicorn loggers so all output uses LOG_FORMAT."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    logger.add("logs/api.log", level="DEBUG", rotation="100 MB", retention="30 days", format=LOG_FORMAT)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log validation errors before returning the standard 422 response."""
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        input_val = error.get("input", "N/A")
        logger.warning(
            "Validation error | location={} | msg={} | input={!r}",
            location,
            msg,
            input_val,
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
