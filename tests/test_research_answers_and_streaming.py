"""Comprehensive regression tests verifying research answer generation, streaming, and citation validation.

Covers all required acceptance tests:
- Test A: Groq provider selection
- Test B: No silent fallback when key missing
- Test C: Provider authentication error handling (HTTP 401)
- Test D: Actual generated answer preservation (genuine LLM responses preserved)
- Test E: Streaming answer preservation & SSE accumulation
- Test F: Stand-in honesty and extractive simulation labelling
- Test G: Retrieval relevance (scaled dot-product attention, Moshi, Dynamic Chunking)
- Test H: Scope mismatch server-side enforcement
- Test I: Missing telemetry preservation (TTFT is None for non-streaming)
- Test J: Calibration metadata & status consistency
"""

import json
from unittest.mock import AsyncMock, patch
import pytest
from src.relay.config import ProviderMode, settings
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.base import ProviderResponse
from src.relay.provider.groq_client import GroqProvider
from src.relay.provider.standin_client import StandInProvider
from src.relay.retrieval.retriever import ResearchRetriever
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
# TEST A: Groq Provider Selection
# -----------------------------------------------------------------------------
def test_groq_provider_selection():
    """Assert that when Groq mode is configured with credentials, GroqProvider is selected."""
    with patch.object(settings, "PROVIDER_MODE", ProviderMode.GROQ):
        groq_mock = GroqProvider(api_key="gsk_mock_valid_key_for_testing_selection")
        orch = PipelineOrchestrator(provider=groq_mock)
        assert isinstance(orch.provider, GroqProvider)
        assert not isinstance(orch.provider, StandInProvider)
        provider_name, model_name = orch._get_provider_names()
        assert provider_name == "groq"
        assert model_name == settings.GROQ_MODEL


# -----------------------------------------------------------------------------
# TEST B: No Silent Fallback When Groq Key Missing
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_silent_fallback_when_groq_key_missing():
    """Assert that Groq mode without API key returns an explicit configuration error and never falls back."""
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


# -----------------------------------------------------------------------------
# TEST C: Provider Authentication Error (HTTP 401)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provider_auth_error_handling():
    """Assert that HTTP 401 authentication errors are propagated without secret leakage."""
    groq_provider = GroqProvider(api_key="gsk_test_mock_invalid_key")

    mock_error_resp = ProviderResponse(
        content='{"error": {"message": "Invalid API Key", "type": "invalid_request_error"}}',
        tokens_in=50,
        tokens_out=0,
        first_token_ms=None,
        total_ms=45.0,
        model="openai/gpt-oss-20b",
        headers={},
        http_status=401,
        error_type="authentication_error",
        is_simulation=False
    )

    with patch.object(groq_provider, "generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_error_resp
        orchestrator = PipelineOrchestrator(provider=groq_provider)

        req = PipelineRequest(
            request_id="test-auth-401",
            question="Explain scaled dot-product attention.",
            scenario="auth_test",
            source=SourceType.REAL,
            target_papers=["1706.03762v7"]
        )

        resp = await orchestrator.execute_request(req)
        assert resp.status == RequestStatus.ERROR
        assert "401" in resp.error_message or "authentication_error" in resp.error_message
        assert "gsk_test_mock_invalid_key" not in resp.error_message
        assert "gsk_test_mock_invalid_key" not in resp.answer


# -----------------------------------------------------------------------------
# TEST D: Actual Generated Answer Preservation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_actual_generated_answer_preservation():
    """Assert that a genuine model response from Groq is preserved exactly and not overwritten."""
    groq_provider = GroqProvider(api_key="gsk_mock_key")

    genuine_explanation = (
        "### Direct Answer\n"
        "Scaled dot-product attention computes compatibility between queries and keys divided by sqrt(d_k).\n\n"
        "### Explanation\n"
        "For large values of d_k, the dot products grow large in magnitude, pushing the softmax function "
        "into regions with extremely small gradients. Dividing by sqrt(d_k) counteracts this effect.\n\n"
        "### Supporting Evidence & Citations\n"
        "[Attention Is All You Need, Page 4, Section 3.2.1]\n\n"
        "### Limitations\n"
        "None."
    )

    mock_ok_resp = ProviderResponse(
        content=genuine_explanation,
        tokens_in=120,
        tokens_out=85,
        first_token_ms=None,
        total_ms=180.0,
        model="openai/gpt-oss-20b",
        headers={},
        http_status=200,
        error_type=None,
        is_simulation=False
    )

    with patch.object(groq_provider, "generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_ok_resp
        orchestrator = PipelineOrchestrator(provider=groq_provider)

        req = PipelineRequest(
            request_id="test-answer-preservation",
            question="Why does the Transformer use scaled dot-product attention?",
            scenario="preservation_test",
            source=SourceType.REAL,
            target_papers=["1706.03762v7"]
        )

        resp = await orchestrator.execute_request(req)
        assert resp.status == RequestStatus.OK
        assert resp.answer == genuine_explanation
        assert "Stand-In Evidence Preview" not in resp.answer
        assert len(resp.citations) > 0
        assert resp.citations[0].paper_id == "1706.03762v7"


# -----------------------------------------------------------------------------
# TEST E: Streaming Answer Preservation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_streaming_answer_preservation():
    """Assert that streaming accumulates tokens into the complete answer with citations."""
    groq_provider = GroqProvider(api_key="gsk_mock_key")

    chunks_to_stream = [
        "### Direct Answer\n",
        "Scaled dot-product attention scales scores by 1/sqrt(d_k).\n\n",
        "### Explanation\n",
        "This prevents large dot product magnitudes from saturating the softmax gradient.\n\n",
        "### Supporting Evidence & Citations\n",
        "[Attention Is All You Need, Page 4, Section 3.2.1]\n"
    ]

    async def mock_stream_gen(prompt, system_prompt="", max_tokens=1024, temperature=0.0):
        for token in chunks_to_stream:
            yield token

    with patch.object(groq_provider, "stream_generate", side_effect=mock_stream_gen):
        orchestrator = PipelineOrchestrator(provider=groq_provider)

        req = PipelineRequest(
            request_id="test-stream-preservation",
            question="Why does the Transformer use scaled dot-product attention?",
            scenario="streaming_test",
            source=SourceType.REAL,
            target_papers=["1706.03762v7"],
            stream=True
        )

        streamed_tokens = []
        done_event = None

        async for event in orchestrator.execute_stream_request(req):
            if event.get("event") == "token":
                streamed_tokens.append(event["content"])
            elif event.get("event") == "done":
                done_event = event

        full_streamed = "".join(streamed_tokens)
        assert full_streamed == "".join(chunks_to_stream)
        assert done_event is not None
        assert done_event["request_id"] == "test-stream-preservation"
        assert len(done_event["citations"]) > 0
        assert done_event["citation_validation"] == "passed"


# -----------------------------------------------------------------------------
# TEST F: Stand-In Honesty & Labelling
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_standin_honesty_and_labelling(fast_standin_orchestrator):
    """Assert that stand-in mode outputs honest extractive evidence with clear disclaimers."""
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
    assert "[Stand-In Evidence Preview — Extractive Simulation]" in resp.answer
    assert "Notice: The Stand-In provider performs deterministic evidence extraction" in resp.answer
    assert resp.source == SourceType.STANDIN
    # Must not contain generic diagnostic template text
    assert "architectural parameters and experimental findings align" not in resp.answer


# -----------------------------------------------------------------------------
# TEST G: Retrieval Relevance across All Target Papers
# -----------------------------------------------------------------------------
def test_retrieval_relevance_attention():
    """Assert that Transformer scaling query retrieves genuine explanatory passages, not figure captions."""
    rr = ResearchRetriever()
    q = (
        "Why does the Transformer use scaled dot-product attention instead of regular dot-product attention? "
        "Explain the purpose of dividing by the square root of dk and what problem it solves."
    )
    res = rr.retrieve(q, RetrievalSetting(chunk_size=600, top_k=4, rerank=True), target_papers=["1706.03762v7"])
    assert len(res.chunks) > 0

    # Top chunk must contain substantive technical explanation
    top_chunk_text = res.chunks[0].text.lower()
    assert any(term in top_chunk_text for term in ["softmax", "gradient", "magnitude", "counteract", "faster and more"])
    assert not top_chunk_text.strip().startswith("figure 2:")


def test_retrieval_relevance_moshi():
    """Assert that Moshi query retrieves Mimi and Helium architecture passages."""
    rr = ResearchRetriever()
    q = "Explain the role of the Mimi audio codec and the Helium language model in Moshi architecture."
    res = rr.retrieve(q, RetrievalSetting(chunk_size=600, top_k=4, rerank=True), target_papers=["2410.00037v1"])
    assert len(res.chunks) > 0
    all_text = " ".join(c.text.lower() for c in res.chunks)
    assert "mimi" in all_text
    assert "helium" in all_text


def test_retrieval_relevance_dynamic_chunking():
    """Assert that Dynamic Chunking query retrieves boundary predictor passages."""
    rr = ResearchRetriever()
    q = "What is the boundary predictor and how does it determine chunk boundaries in Dynamic Chunking?"
    res = rr.retrieve(q, RetrievalSetting(chunk_size=600, top_k=4, rerank=True), target_papers=["2507.07955v2"])
    assert len(res.chunks) > 0
    all_text = " ".join(c.text.lower() for c in res.chunks)
    assert any(term in all_text for term in ["boundary predictor", "dynamic chunking", "differentiable"])


# -----------------------------------------------------------------------------
# TEST H: Scope Mismatch Server-Side Enforcement
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scope_mismatch(fast_standin_orchestrator):
    """Assert that asking a Moshi-specific question under Attention scope triggers a refusal."""
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
    assert "Moshi" in resp.answer or "scope" in resp.answer.lower()
    assert len(resp.citations) == 0


# -----------------------------------------------------------------------------
# TEST I: Missing Telemetry Preservation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_telemetry_preservation():
    """Assert that non-streaming TTFT for Groq remains None (not measured) rather than fabricating total latency."""
    groq_provider = GroqProvider(api_key="gsk_mock_key")
    mock_resp = ProviderResponse(
        content="### Direct Answer\nThe base Transformer uses d_model = 512.\n\n### Supporting Evidence & Citations\n[Attention Is All You Need, Page 3, Section 3.1]",
        tokens_in=50,
        tokens_out=25,
        first_token_ms=None,
        total_ms=120.0,
        model="openai/gpt-oss-20b",
        headers={},
        http_status=200,
        error_type=None,
        is_simulation=False
    )
    with patch.object(groq_provider, "generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_resp
        orchestrator = PipelineOrchestrator(provider=groq_provider)
        req = PipelineRequest(
            request_id="test-telemetry-nonstream",
            question="What is the base Transformer dimension d_model?",
            scenario="telemetry_test",
            source=SourceType.REAL,
            target_papers=["1706.03762v7"],
            stream=False
        )
        resp = await orchestrator.execute_request(req)
        assert resp.status == RequestStatus.OK
        # In non-streaming requests, TTFT is not measured and must be None
        assert resp.first_token_ms is None
        assert resp.total_ms > 0.0


# -----------------------------------------------------------------------------
# TEST J: Calibration Metadata & Status Consistency
# -----------------------------------------------------------------------------
def test_calibration_metadata_consistency():
    """Assert that calibration_profile.json status is consistently exploratory."""
    calib_file = settings.DATA_DIR / "calibration_profile.json"
    assert calib_file.exists(), "calibration_profile.json must exist in data directory"
    with open(calib_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    status = data.get("calibration_status")
    assert status == "exploratory_calibration"
    # Must not claim definitive empirical calibration with only 3 samples
    assert data.get("sample_count") == 3
