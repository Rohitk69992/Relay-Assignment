"""Integration tests for the complete multi-stage pipeline."""

import pytest
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.standin_client import StandInProvider
from src.relay.schemas import (
    AnswerGrade,
    PipelineRequest,
    RequestStatus,
    RetrievalSetting,
    SourceType
)


@pytest.mark.asyncio
async def test_pipeline_answerable_query():
    # Use fast stand-in provider
    standin = StandInProvider(p50_latency_ms=20.0, p95_latency_ms=40.0, tokens_per_sec=3000.0)
    orchestrator = PipelineOrchestrator(provider=standin)

    req = PipelineRequest(
        request_id="test_req_01",
        question="What is the dimension d_k in the base Transformer attention mechanism?",
        scenario="unit_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=500, top_k=2, rerank=True)
    )

    resp = await orchestrator.execute_request(
        req,
        expected_answerable=True,
        expected_paper_id="1706.03762v7",
        expected_page=4
    )

    assert resp.status == RequestStatus.OK
    assert resp.answer_grade == AnswerGrade.CORRECT
    assert len(resp.citations) > 0
    assert resp.first_token_ms is not None
    assert resp.total_ms > 0


@pytest.mark.asyncio
async def test_pipeline_unanswerable_refusal():
    standin = StandInProvider(p50_latency_ms=20.0, p95_latency_ms=40.0, tokens_per_sec=3000.0)
    orchestrator = PipelineOrchestrator(provider=standin)

    req = PipelineRequest(
        request_id="test_req_02",
        question="What quantum circuit was tested in Attention Is All You Need?",
        scenario="unit_test",
        source=SourceType.STANDIN
    )

    resp = await orchestrator.execute_request(
        req,
        expected_answerable=False
    )

    assert resp.status == RequestStatus.OK
    assert resp.answer_grade == AnswerGrade.REFUSED_CORRECTLY
    assert "REFUSAL" in resp.answer.upper() or "DOES NOT ESTABLISH" in resp.answer.upper()


@pytest.mark.asyncio
async def test_pipeline_off_topic_query():
    standin = StandInProvider(p50_latency_ms=10.0, p95_latency_ms=20.0)
    orchestrator = PipelineOrchestrator(provider=standin)

    req = PipelineRequest(
        request_id="test_req_03",
        question="What is the weather forecast for Paris tomorrow?",
        scenario="unit_test",
        source=SourceType.STANDIN
    )

    resp = await orchestrator.execute_request(req, expected_answerable=False)
    assert resp.status == RequestStatus.OK
    assert resp.answer_grade == AnswerGrade.REFUSED_CORRECTLY
    assert "REFUSAL" in resp.answer.upper()
