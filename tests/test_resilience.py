"""Unit tests for admission queues, concurrency semaphores, and retry policies."""

import asyncio
import time
import pytest
from src.relay.resilience.queue_manager import AdmissionQueueManager, QueueFullError, QueueTimeoutError
from src.relay.resilience.rate_limiter import ClientRateLimiter
from src.relay.resilience.retry_policy import RetryManager, ErrorClassification


@pytest.mark.asyncio
async def test_queue_full_rejection():
    # Queue with capacity 2
    qm = AdmissionQueueManager(max_queue_size=2, max_concurrency=1)

    t1 = await qm.enter_queue("req-1")
    t2 = await qm.enter_queue("req-2")
    assert qm.current_queue_depth == 2

    # 3rd request must be rejected immediately
    with pytest.raises(QueueFullError):
        await qm.enter_queue("req-3")


@pytest.mark.asyncio
async def test_queue_timeout():
    # Concurrency 1, timeout 0.1s
    qm = AdmissionQueueManager(max_queue_size=10, max_concurrency=1, queue_timeout_seconds=0.1)

    # Acquire the single slot with req-1
    t1 = await qm.enter_queue("req-1")
    slot1_wait = await qm.acquire_execution_slot("req-1", t1)

    # req-2 enters queue while slot is busy
    t2 = await qm.enter_queue("req-2")
    with pytest.raises(QueueTimeoutError):
        await qm.acquire_execution_slot("req-2", t2)

    qm.release_execution_slot()


@pytest.mark.asyncio
async def test_client_rate_limiter_rpm():
    limiter = ClientRateLimiter(rpm=2, tpm=10000, rpd=100, tpd=10000)

    p1, _, _ = await limiter.acquire_permit(50)
    assert p1 is True

    p2, _, _ = await limiter.acquire_permit(50)
    assert p2 is True

    # 3rd fails
    p3, wait_sec, reason = await limiter.acquire_permit(50)
    assert p3 is False
    assert wait_sec is not None
    assert wait_sec > 0.0


def test_retry_policy_classification():
    # 401 is permanent
    assert ErrorClassification.is_retryable(401) is False
    # 400 is permanent
    assert ErrorClassification.is_retryable(400) is False
    # 429 is retryable
    assert ErrorClassification.is_retryable(429) is True
    # 503 is retryable
    assert ErrorClassification.is_retryable(503) is True
    # Daily quota in error message is permanent
    assert ErrorClassification.is_retryable(429, message="Daily budget tokens per day exhausted") is False


def test_retry_manager_backoff():
    rm = RetryManager(max_retries=3, base_delay_s=0.2, max_delay_s=2.0, jitter_s=0.05, request_deadline_s=10.0)

    start_mono = time.perf_counter()

    # Attempt 0
    can_retry, delay, _ = rm.should_retry(0, 429, "rate_limit_exceeded", "Rate limited", start_mono, {})
    assert can_retry is True
    assert 0.2 <= delay <= 0.3

    # Attempt 3 (exhausted)
    can_retry_max, _, _ = rm.should_retry(3, 429, "rate_limit_exceeded", "Rate limited", start_mono, {})
    assert can_retry_max is False
