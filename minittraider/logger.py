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


class DailyFileHandler(logging.FileHandler):
    """Writes to logs/trader_<YYYY-MM-DD>.log and switches file at midnight, so a
    bot that runs for days doesn't pile everything into its start day's file."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.day = datetime.now().strftime("%Y-%m-%d")
        super().__init__(self._path(), mode="a", encoding="utf-8")

    def _path(self) -> str:
        return os.path.join(self.log_dir, f"trader_{self.day}.log")

    def emit(self, record: logging.LogRecord) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.day:
            self.acquire()
            try:
                self.close()
                self.day = today
                self.baseFilename = os.path.abspath(self._path())
                self.stream = self._open()
            finally:
                self.release()
        super().emit(record)


def setup_logger(
    name: str = "minittraider",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Create the logger once, but ALWAYS honour a later request for a file
    handler / level (modules call get_logger() at import time, before trader.py
    asks for file logging - that request used to be silently ignored)."""
    global logger_instance
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    logger = logger_instance or logging.getLogger(name)
    if logger_instance is None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger_instance = logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = DailyFileHandler(os.path.dirname(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
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
