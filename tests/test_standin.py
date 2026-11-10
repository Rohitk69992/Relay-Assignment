"""Unit tests for the controllable Stand-In Model Service."""

import asyncio
import pytest
from src.relay.provider.standin_client import StandInProvider, StandInRateTracker


@pytest.mark.asyncio
async def test_standin_latency_sampling():
    provider = StandInProvider(
        p50_latency_ms=100.0,
        p95_latency_ms=250.0,
        tokens_per_sec=1000.0,
        failure_rate=0.0
    )

    resp = await provider.generate("Test prompt for stand-in latency")
    assert resp.http_status == 200
    assert resp.first_token_ms is not None
    assert resp.first_token_ms > 20.0
    assert resp.total_ms >= resp.first_token_ms
    assert resp.is_simulation is True


@pytest.mark.asyncio
async def test_standin_load_slowdown():
    provider = StandInProvider(
        p50_latency_ms=50.0,
        p95_latency_ms=100.0,
        load_slowdown_factor=0.5
    )

    provider.in_flight_count = 1
    base_ttft = provider.compute_load_inflated_ttft(100.0)
    assert base_ttft == 100.0

    # 5 concurrent requests in flight
    provider.in_flight_count = 5
    inflated_ttft = provider.compute_load_inflated_ttft(100.0)
    assert inflated_ttft == 100.0 * (1.0 + 0.5 * 4)  # 300.0ms


@pytest.mark.asyncio
async def test_standin_quota_rejection():
    # Configure tiny RPM limit of 2 requests
    tracker = StandInRateTracker(rpm=2, tpm=10000, rpd=100, tpd=100000)

    # First two should succeed
    v1 = await tracker.check_and_consume(50)
    assert v1 is None

    v2 = await tracker.check_and_consume(50)
    assert v2 is None

    # Third should fail with RPM violation
    v3 = await tracker.check_and_consume(50)
    assert v3 is not None
    assert v3["type"] == "requests_per_minute"
    assert v3["retry_after"] > 0


@pytest.mark.asyncio
async def test_standin_daily_budget_exhaustion():
    tracker = StandInRateTracker(rpm=100, tpm=100000, rpd=2, tpd=500000)

    # Use up daily limit of 2
    await tracker.check_and_consume(10)
    await tracker.check_and_consume(10)

    # Third should hit daily budget
    v = await tracker.check_and_consume(10)
    assert v is not None
    assert v["type"] == "requests_per_day"


@pytest.mark.asyncio
async def test_standin_streaming():
    provider = StandInProvider(
        p50_latency_ms=30.0,
        p95_latency_ms=60.0,
        tokens_per_sec=2000.0
    )

    chunks = []
    async for chunk in provider.stream_generate("Summarize attention mechanism"):
        chunks.append(chunk)

    assert len(chunks) > 3
    full_text = "".join(chunks)
    assert "[STAND-IN SYNTHESIS" in full_text
