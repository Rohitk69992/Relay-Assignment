"""Comprehensive regression tests verifying research answer generation, streaming, and citation validation.

Covers the 10 core functional tests:
1. Transformer conceptual question (scaled dot-product attention mechanics).
2. Moshi question (Mimi audio codec & Helium LM).
3. Dynamic Chunking question (boundary predictor).
4. Cross-paper comparison (Transformer vs Dynamic Chunking).
5. Unsupported question refusal (Groq pricing).
6. Scope mismatch refusal (Moshi question under Attention paper scope).
7. Non-streaming execution consistency.
8. Streaming SSE consistency (chunk accumulation equals complete answer, request ID verified).
9. Stand-in honesty (labeled extractive simulation, real evidence passages, zero generic filler).
10. No silent fallback (Groq mode without API key yields clean configuration error).
"""

import pytest
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.groq_client import GroqProvider
from src.relay.provider.standin_client import StandInProvider
from src.relay.schemas import (
    PipelineRequest,
    RequestStatus,
    RetrievalSetting,
    SourceType
)


@pytest.fixture
def fast_standin_orchestrator():
    """Provides a fast, deterministic stand-in orchestrator for unit testing."""
    standin = StandInProvider(p50_latency_ms=10.0, p95_latency_ms=25.0, tokens_per_sec=4000.0)
    return PipelineOrchestrator(provider=standin)


# -----------------------------------------------------------------------------
# TEST 1: Transformer Conceptual Question
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transformer_conceptual_question(fast_standin_orchestrator):
    q = (
        "Why does the Transformer use scaled dot-product attention instead of regular dot-product attention? "
        "Explain the purpose of dividing by the square root of dk and what problem it solves."
    )
    req = PipelineRequest(
        request_id="test-trans-01",
        question=q,
        scenario="functional_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True),
        target_papers=["1706.03762v7"]
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert len(resp.citations) > 0

    # Citations must be exclusively from 1706.03762v7
    for c in resp.citations:
        assert c.paper_id == "1706.03762v7"
        assert c.page in {3, 4, 5}

    # Verify substantive extracted evidence
    ans_lower = resp.answer.lower()
    assert any(term in ans_lower for term in ["softmax", "gradient", "magnitude", "scale", "counteract"])
    # Verify no generic boilerplate filler
    assert "architectural parameters and experimental findings align" not in resp.answer
    assert "structural components and algorithmic parameters verified" not in resp.answer


# -----------------------------------------------------------------------------
# TEST 2: Moshi Architecture Question
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_moshi_question(fast_standin_orchestrator):
    q = "Explain the role of the Mimi audio codec and the Helium language model in Moshi's architecture."
    req = PipelineRequest(
        request_id="test-moshi-02",
        question=q,
        scenario="functional_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True),
        target_papers=["2410.00037v1"]
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert len(resp.citations) > 0
    for c in resp.citations:
        assert c.paper_id == "2410.00037v1"

    ans_lower = resp.answer.lower()
    assert "mimi" in ans_lower
    assert "helium" in ans_lower
    assert "architectural parameters and experimental findings align" not in resp.answer


# -----------------------------------------------------------------------------
# TEST 3: Dynamic Chunking Boundary Predictor Question
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dynamic_chunking_question(fast_standin_orchestrator):
    q = (
        "What is the role of the boundary predictor in Dynamic Chunking, and how does it determine "
        "when a sequence should be divided into chunks?"
    )
    req = PipelineRequest(
        request_id="test-dc-03",
        question=q,
        scenario="functional_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True),
        target_papers=["2507.07955v2"]
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert len(resp.citations) > 0
    for c in resp.citations:
        assert c.paper_id == "2507.07955v2"

    ans_lower = resp.answer.lower()
    assert any(term in ans_lower for term in ["boundary", "chunk", "downsampler", "cosine similarity"])
    assert "architectural parameters and experimental findings align" not in resp.answer


# -----------------------------------------------------------------------------
# TEST 4: Cross-Paper Comparative Question
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cross_paper_comparison(fast_standin_orchestrator):
    q = "Compare how the Transformer and Dynamic Chunking approach sequence processing."
    req = PipelineRequest(
        request_id="test-cross-04",
        question=q,
        scenario="functional_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True),
        target_papers=None  # Cross-paper mode
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert len(resp.citations) > 0

    cited_paper_ids = {c.paper_id for c in resp.citations}
    assert "1706.03762v7" in cited_paper_ids
    assert "2507.07955v2" in cited_paper_ids


# -----------------------------------------------------------------------------
# TEST 5: Unsupported Question (Refusal)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unsupported_pricing_question(fast_standin_orchestrator):
    q = "What is the current Groq API pricing?"
    req = PipelineRequest(
        request_id="test-refusal-05",
        question=q,
        scenario="functional_test",
        source=SourceType.STANDIN,
        target_papers=["1706.03762v7"]
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert "REFUSAL" in resp.answer.upper() or "DOES NOT ESTABLISH" in resp.answer.upper()
    assert len(resp.citations) == 0


# -----------------------------------------------------------------------------
# TEST 6: Scope Mismatch Refusal
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scope_mismatch(fast_standin_orchestrator):
    q = "What is the framerate of the Mimi audio codec in Moshi?"
    req = PipelineRequest(
        request_id="test-scope-06",
        question=q,
        scenario="functional_test",
        source=SourceType.STANDIN,
        target_papers=["1706.03762v7"]  # Locked to Attention paper
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert "REFUSAL" in resp.answer.upper()
    # Must explain that the question targets a different paper than selected scope
    assert "Moshi" in resp.answer or "scope" in resp.answer.lower()
    assert len(resp.citations) == 0


# -----------------------------------------------------------------------------
# TEST 7 & 8: Streaming and Non-Streaming Equivalence & Invariants
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_streaming_and_non_streaming_consistency(fast_standin_orchestrator):
    q = "What is the dimension d_k in the base Transformer attention mechanism?"
    req = PipelineRequest(
        request_id="test-equiv-07-08",
        question=q,
        scenario="consistency_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True),
        target_papers=["1706.03762v7"]
    )

    # 1. Non-streaming execution
    non_stream_resp = await fast_standin_orchestrator.execute_request(req)
    assert non_stream_resp.status == RequestStatus.OK
    assert non_stream_resp.answer != ""
    assert len(non_stream_resp.citations) > 0
    assert non_stream_resp.request_id == "test-equiv-07-08"

    # 2. Streaming execution
    stream_tokens = []
    events = []
    done_event = None

    async for event in fast_standin_orchestrator.execute_stream_request(req):
        events.append(event)
        if event["event"] == "token":
            stream_tokens.append(event["content"])
        elif event["event"] == "done":
            done_event = event

    assert len(stream_tokens) > 3, "Stream must deliver multiple incremental tokens"
    assert done_event is not None, "Stream must emit terminal done event"
    assert done_event["request_id"] == "test-equiv-07-08", "Terminal done event must preserve request_id"
    assert len(done_event["citations"]) > 0, "Terminal done event must return valid citations"
    assert done_event["citation_validation"] == "passed"

    # Verify that stream accumulation perfectly reproduces the non-streaming answer
    accumulated_answer = "".join(stream_tokens)
    assert accumulated_answer == non_stream_resp.answer, "Streaming content must match completed answer"


# -----------------------------------------------------------------------------
# TEST 9: Stand-in Honesty & Labelling
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_standin_honesty_and_labelling(fast_standin_orchestrator):
    q = "Why does the Transformer use scaled dot-product attention?"
    req = PipelineRequest(
        request_id="test-honesty-09",
        question=q,
        scenario="honesty_test",
        source=SourceType.STANDIN,
        target_papers=["1706.03762v7"]
    )
    resp = await fast_standin_orchestrator.execute_request(req)

    # Must be explicitly labeled as extractive stand-in simulation
    assert "[Stand-in evidence preview" in resp.answer
    assert "extractive simulation, not LLM-generated" in resp.answer
    # Must preserve provider source as standin
    assert resp.source == SourceType.STANDIN
    # Must not contain generic synthesis filler
    assert "structural components and algorithmic parameters verified" not in resp.answer


# -----------------------------------------------------------------------------
# TEST 10: No Silent Fallback When Groq Key Missing
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_silent_fallback_when_groq_key_missing():
    # Instantiate GroqProvider with explicitly missing API key
    groq_provider = GroqProvider(api_key="")
    orchestrator = PipelineOrchestrator(provider=groq_provider)

    req = PipelineRequest(
        request_id="test-no-fallback-10",
        question="What is self-attention?",
        scenario="fallback_test",
        source=SourceType.REAL,
        target_papers=["1706.03762v7"]
    )

    # 1. Non-streaming call must fail with configuration error
    resp = await orchestrator.execute_request(req)
    assert resp.status == RequestStatus.ERROR
    assert "GROQ_API_KEY" in resp.answer
    assert "Configuration Error" in resp.answer
    assert resp.source == SourceType.REAL  # Must not silently flip to standin

    # 2. Streaming call must emit error event with 503 status
    error_event = None
    async for event in orchestrator.execute_stream_request(req):
        if event.get("event") == "error":
            error_event = event

    assert error_event is not None
    assert error_event["status"] == 503
    assert "GROQ_API_KEY" in error_event["error"]
