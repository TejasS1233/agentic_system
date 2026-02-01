"""Logging configuration using Loguru."""

import os
import sys

from loguru import logger

logger.remove()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)

logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
)

if LOG_TO_FILE:
    logger.add(
        "logs/ToolForger_{time:YYYYMMDD}.log",
        format=LOG_FORMAT,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


def get_logger(name: str):
    """Get a named logger instance."""
    return logger.bind(name=name)
