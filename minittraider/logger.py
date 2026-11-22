"""Logging configuration for minittraider.

Usage:
    from logger import get_logger
    log = get_logger()
    log.info("Something happened")
    log.error("Something failed", exc_info=True)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

logger_instance: logging.Logger | None = None


def setup_logger(
    name: str = "minittraider",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    global logger_instance
    if logger_instance:
        return logger_instance

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(
            log_file, mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger_instance = logger
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    global logger_instance
    if logger_instance is None:
        logger_instance = setup_logger()
    if name:
        return logger_instance.getChild(name)
    return logger_instance


def get_daily_log_file() -> str:
    """Path to today's log file in the log directory."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(log_dir, f"trader_{today}.log")
