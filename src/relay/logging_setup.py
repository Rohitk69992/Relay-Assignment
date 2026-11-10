"""Structured logging setup for The Relay pipeline."""

import logging
import sys
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured text with context."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        request_id = getattr(record, "request_id", "-")
        stage = getattr(record, "stage", "-")
        duration = getattr(record, "duration_ms", None)

        msg = f"[{timestamp}] [{record.levelname:<5}] [{request_id}] [{stage}]"
        if duration is not None:
            msg += f" ({duration:.1f}ms)"
        msg += f" {record.getMessage()}"
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


def get_logger(name: str = "relay", log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger


class MonotonicTimer:
    """Measures precise elapsed monotonic duration in milliseconds."""

    def __init__(self):
        self._start_time: float = time.perf_counter()
        self._first_token_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def mark_first_token(self) -> float:
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
        return (self._first_token_time - self._start_time) * 1000.0

    def mark_complete(self) -> float:
        self._end_time = time.perf_counter()
        return (self._end_time - self._start_time) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        current = self._end_time or time.perf_counter()
        return (current - self._start_time) * 1000.0

    @property
    def first_token_ms(self) -> Optional[float]:
        if self._first_token_time is None:
            return None
        return (self._first_token_time - self._start_time) * 1000.0
