"""Shared logging setup for pipeline modules — logs to both stdout and a file."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.pipeline import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger writing to ``backend/logs/<name>.log`` + stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)
    config.LOGS_PATH.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    fh = logging.FileHandler(config.LOGS_PATH / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger
