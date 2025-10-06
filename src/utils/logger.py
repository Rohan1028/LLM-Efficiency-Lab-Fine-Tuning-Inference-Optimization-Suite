from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "llm_efficiency_lab",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure a Rich-enabled logger with optional file output.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    console = Console(stderr=True)
    rich_handler = RichHandler(console=console, show_time=True, show_path=False)
    rich_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    rich_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    logger.debug("Logger %s initialized.", name)
    return logger


def get_logger(name: str = "llm_efficiency_lab") -> logging.Logger:
    """
    Convenience helper to fetch a child logger.
    """

    parent = logging.getLogger("llm_efficiency_lab")
    if not parent.handlers:
        setup_logger()
    return parent.getChild(name)
