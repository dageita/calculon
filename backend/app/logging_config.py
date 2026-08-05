import contextlib
import logging
import logging.config
import os
import sys
from typing import Dict, Optional

LOG_LEVELS = ("critical", "error", "warning", "info", "debug")
ENV_LOG_LEVEL = "CALCULON_LOG_LEVEL"


def normalize_level(level: str) -> str:
    level = (level or "info").lower()
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level {level!r}; choose from {', '.join(LOG_LEVELS)}")
    return level


def level_to_logging(level: str) -> int:
    return getattr(logging, normalize_level(level).upper())


def should_quiet_native(level: str) -> bool:
    """Only debug keeps LLMFlowSimulator std::cout; info and above stay quiet."""
    return level_to_logging(level) > logging.DEBUG


def build_log_config(level: str) -> Dict:
    level_name = normalize_level(level).upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s:%(name)s:%(message)s"},
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level_name, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": level_name, "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": level_name, "propagate": False},
            # asyncio emits DEBUG at import time unless capped explicitly.
            "asyncio": {"handlers": ["default"], "level": level_name, "propagate": False},
        },
        "root": {"handlers": ["default"], "level": level_name},
    }


def configure_logging(level: str) -> None:
    normalized = normalize_level(level)
    os.environ[ENV_LOG_LEVEL] = normalized
    logging.config.dictConfig(build_log_config(normalized))


@contextlib.contextmanager
def native_output_guard(level: Optional[str] = None):
    """Suppress native simulator stdout when log level is warning or quieter."""
    active_level = normalize_level(level or os.environ.get(ENV_LOG_LEVEL, "info"))
    if not should_quiet_native(active_level):
        yield
        return

    # Let LLMFlowSimulator mute its own cout when timeline collection is off.
    os.environ.pop("SIM_VERBOSE", None)

    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)
        os.close(devnull_fd)
