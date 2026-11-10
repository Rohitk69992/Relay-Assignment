"""Bounded admission queue and concurrency management."""

import asyncio
import time
from typing import Optional
from src.relay.config import settings
from src.relay.logging_setup import get_logger

logger = get_logger("relay.queue")


class QueueFullError(Exception):
    """Raised when admission queue reaches maximum capacity."""
    pass


class QueueTimeoutError(Exception):
    """Raised when request wait time in admission queue exceeds deadline."""
    pass


class AdmissionQueueManager:
    """Manages request admission, queue buffering, queue deadlines, and concurrent execution slots."""

    def __init__(
        self,
        max_queue_size: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        queue_timeout_seconds: Optional[float] = None
    ):
        self.max_queue_size = max_queue_size or settings.MAX_QUEUE_SIZE
        self.max_concurrency = max_concurrency or settings.MAX_CONCURRENT_REQUESTS
        self.queue_timeout_s = queue_timeout_seconds or settings.QUEUE_TIMEOUT_SECONDS

        self.concurrency_semaphore = asyncio.Semaphore(self.max_concurrency)
        self.current_queue_depth: int = 0
        self.peak_queue_depth: int = 0

        self.total_admitted: int = 0
        self.total_rejected_queue_full: int = 0
        self.total_dropped_queue_timeout: int = 0

        self._lock = asyncio.Lock()

    async def enter_queue(self, request_id: str) -> float:
        """Admits a request into the buffer or raises QueueFullError.

        Returns:
            enqueue_timestamp (float)
        """
        async with self._lock:
            if self.current_queue_depth >= self.max_queue_size:
                self.total_rejected_queue_full += 1
                logger.warning(
                    f"Queue full ({self.current_queue_depth}/{self.max_queue_size}). Rejecting request {request_id}",
                    extra={"request_id": request_id, "stage": "admission"}
                )
                raise QueueFullError(f"Admission queue full ({self.current_queue_depth} requests buffered). Please retry later.")

            self.current_queue_depth += 1
            if self.current_queue_depth > self.peak_queue_depth:
                self.peak_queue_depth = self.current_queue_depth
            self.total_admitted += 1
            return time.perf_counter()

    async def acquire_execution_slot(self, request_id: str, enqueue_time: float) -> float:
        """Waits for an available execution concurrency slot subject to queue timeout.

        Returns:
            queue_wait_duration_ms (float)
        """
        try:
            # Wait for concurrency slot with timeout
            remaining_time = max(0.01, self.queue_timeout_s - (time.perf_counter() - enqueue_time))
            await asyncio.wait_for(
                self.concurrency_semaphore.acquire(),
                timeout=remaining_time
            )
            queue_wait_ms = (time.perf_counter() - enqueue_time) * 1000.0
            return queue_wait_ms
        except asyncio.TimeoutError:
            async with self._lock:
                self.total_dropped_queue_timeout += 1
            logger.warning(
                f"Request {request_id} exceeded queue timeout ({self.queue_timeout_s}s)",
                extra={"request_id": request_id, "stage": "admission"}
            )
            raise QueueTimeoutError(f"Request spent {self.queue_timeout_s}s waiting in admission queue and timed out.")
        finally:
            async with self._lock:
                self.current_queue_depth = max(0, self.current_queue_depth - 1)

    def release_execution_slot(self):
        """Releases the concurrency slot after model execution finishes."""
        self.concurrency_semaphore.release()
