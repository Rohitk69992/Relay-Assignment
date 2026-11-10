"""Retry policy with error classification, exponential backoff, jitter, and deadline control."""

import asyncio
import random
import time
from typing import Dict, Optional, Tuple
from src.relay.config import settings
from src.relay.logging_setup import get_logger

logger = get_logger("relay.retry")


class ErrorClassification:
    """Classifies provider error codes and messages into retryable vs permanent categories."""

    PERMANENT_STATUSES = {400, 401, 403, 404, 422}
    RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

    @classmethod
    def is_retryable(cls, http_status: int, error_type: Optional[str] = None, message: str = "") -> bool:
        if http_status in cls.PERMANENT_STATUSES:
            return False

        # Daily budget exhaustion is non-retryable within the current day
        msg_lower = (message or "").lower()
        if "daily" in msg_lower or "budget" in msg_lower or "tokens per day" in msg_lower or "tpd" in msg_lower:
            return False

        if http_status in cls.RETRYABLE_STATUSES:
            return True

        if error_type in {"service_unavailable", "request_timeout", "rate_limit_exceeded"}:
            return True

        return False


class RetryManager:
    """Manages backoff calculation, retry limits, and request deadline enforcement."""

    def __init__(
        self,
        max_retries: Optional[int] = None,
        base_delay_s: Optional[float] = None,
        max_delay_s: Optional[float] = None,
        jitter_s: Optional[float] = None,
        request_deadline_s: Optional[float] = None
    ):
        self.max_retries = max_retries or settings.MAX_RETRIES
        self.base_delay_s = base_delay_s or settings.RETRY_BASE_DELAY_SECONDS
        self.max_delay_s = max_delay_s or settings.RETRY_MAX_DELAY_SECONDS
        self.jitter_s = jitter_s or settings.RETRY_JITTER
        self.request_deadline_s = request_deadline_s or settings.REQUEST_TIMEOUT_SECONDS

    def should_retry(
        self,
        attempt: int,
        http_status: int,
        error_type: Optional[str],
        error_message: str,
        start_time_monotonic: float,
        headers: Dict[str, str]
    ) -> Tuple[bool, float, str]:
        """Evaluates whether to retry.

        Returns:
            (can_retry: bool, delay_s: float, reason: str)
        """
        if attempt >= self.max_retries:
            return False, 0.0, f"Max retries ({self.max_retries}) exhausted."

        if not ErrorClassification.is_retryable(http_status, error_type, error_message):
            return False, 0.0, f"Error {http_status} ({error_type}) is permanent and non-retryable."

        elapsed = time.perf_counter() - start_time_monotonic
        remaining_budget = self.request_deadline_s - elapsed
        if remaining_budget <= 0.2:
            return False, 0.0, f"Request deadline exceeded (elapsed {elapsed:.2f}s >= {self.request_deadline_s}s)."

        # Determine delay
        delay = self.base_delay_s * (2 ** attempt) + random.uniform(0, self.jitter_s)
        delay = min(delay, self.max_delay_s)

        # Check if provider returned retry-after
        retry_after_str = headers.get("retry-after")
        if retry_after_str:
            try:
                provider_delay = float(retry_after_str)
                delay = max(delay, provider_delay)
            except ValueError:
                pass

        if delay >= remaining_budget:
            return False, 0.0, f"Suggested delay ({delay:.2f}s) exceeds remaining request deadline ({remaining_budget:.2f}s)."

        return True, delay, f"Retry attempt {attempt + 1}/{self.max_retries} scheduled after {delay:.2f}s backoff."
