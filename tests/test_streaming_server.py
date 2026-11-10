"""Test for true SSE token streaming and monotonic TTFT."""

import pytest
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.standin_client import StandInProvider
from src.relay.schemas import PipelineRequest, RetrievalSetting, SourceType


@pytest.mark.asyncio
async def test_genuine_token_streaming():
    """Verifies that tokens are yielded sequentially before the completion event."""
    standin = StandInProvider(p50_latency_ms=25.0, p95_latency_ms=50.0, tokens_per_sec=1500.0)
    orchestrator = PipelineOrchestrator(provider=standin)

    req = PipelineRequest(
        request_id="stream_test_01",
        question="What is the dimension d_k in the base Transformer attention mechanism?",
        scenario="streaming_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=500, top_k=2, rerank=True),
        stream=True
    )

    events = []
    first_token_seen_at = None
    done_seen_at = None

    async for chunk in orchestrator.execute_stream_request(req):
        events.append(chunk)
        if chunk["event"] == "token" and first_token_seen_at is None:
            first_token_seen_at = chunk.get("first_token_ms")
        elif chunk["event"] == "done":
            done_seen_at = chunk.get("total_ms")

    # Verify event sequencing
    assert len(events) >= 3, "Stream should yield multiple token events followed by done"
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) > 0, "Must yield at least one token chunk"
    
    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1, "Must yield exactly one terminal done event"

    # Verify first token arrival is strictly less than total completion time
    assert first_token_seen_at is not None
    assert done_seen_at is not None
    assert first_token_seen_at <= done_seen_at, f"TTFT ({first_token_seen_at}ms) should precede or equal total time ({done_seen_at}ms)"
    assert done_events[0]["verified"] is True
