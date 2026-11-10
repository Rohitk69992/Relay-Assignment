"""Integration tests for the complete multi-stage pipeline and evaluation harness."""

import pytest
from src.relay.evaluation.quality_evaluator import QualityEvaluator
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.standin_client import StandInProvider
from src.relay.schemas import (
    AnswerGrade,
    EvaluationItem,
    PipelineRequest,
    RequestStatus,
    RetrievalSetting,
    SourceType
)


@pytest.mark.asyncio
async def test_pipeline_answerable_query():
    standin = StandInProvider(p50_latency_ms=20.0, p95_latency_ms=40.0, tokens_per_sec=3000.0)
    orchestrator = PipelineOrchestrator(provider=standin)
    evaluator = QualityEvaluator()

    req = PipelineRequest(
        request_id="test_req_01",
        question="What is the dimension d_k in the base Transformer attention mechanism?",
        scenario="unit_test",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=500, top_k=2, rerank=True)
    )

    # Note: No ground-truth parameters are passed into the runtime pipeline
    resp = await orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert len(resp.citations) > 0
    assert resp.first_token_ms is not None
    assert resp.total_ms > 0

    # Grade response using independent quality evaluator
    eval_item = EvaluationItem(
        question_id="Q_TEST_01",
        category="factual",
        question=req.question,
        expected_answer="d_k = 64",
        expected_paper="1706.03762v7",
        expected_page=4,
        expected_section="3.2.2 Multi-Head Attention",
        is_answerable=True
    )
    grade, rationale = evaluator.grade(resp, eval_item)
    assert grade == AnswerGrade.CORRECT
    assert "matched" in rationale.lower()


@pytest.mark.asyncio
async def test_pipeline_unanswerable_refusal():
    standin = StandInProvider(p50_latency_ms=20.0, p95_latency_ms=40.0, tokens_per_sec=3000.0)
    orchestrator = PipelineOrchestrator(provider=standin)
    evaluator = QualityEvaluator()

    req = PipelineRequest(
        request_id="test_req_02",
        question="What quantum circuit was tested in Attention Is All You Need?",
        scenario="unit_test",
        source=SourceType.STANDIN
    )

    resp = await orchestrator.execute_request(req)

    assert resp.status == RequestStatus.OK
    assert "REFUSAL" in resp.answer.upper() or "DOES NOT ESTABLISH" in resp.answer.upper()

    eval_item = EvaluationItem(
        question_id="Q_TEST_02",
        category="unanswerable",
        question=req.question,
        expected_answer="REFUSAL",
        expected_paper="1706.03762v7",
        expected_page=0,
        expected_section="None",
        is_answerable=False
    )
    grade, rationale = evaluator.grade(resp, eval_item)
    assert grade == AnswerGrade.REFUSED_CORRECTLY


@pytest.mark.asyncio
async def test_pipeline_off_topic_query():
    standin = StandInProvider(p50_latency_ms=10.0, p95_latency_ms=20.0)
    orchestrator = PipelineOrchestrator(provider=standin)
    evaluator = QualityEvaluator()

    req = PipelineRequest(
        request_id="test_req_03",
        question="What is the weather forecast for Paris tomorrow?",
        scenario="unit_test",
        source=SourceType.STANDIN
    )

    resp = await orchestrator.execute_request(req)
    assert resp.status == RequestStatus.OK
    assert "REFUSAL" in resp.answer.upper()

    eval_item = EvaluationItem(
        question_id="Q_TEST_03",
        category="off_topic",
        question=req.question,
        expected_answer="REFUSAL",
        expected_paper="none",
        expected_page=0,
        expected_section="None",
        is_answerable=False
    )
    grade, rationale = evaluator.grade(resp, eval_item)
    assert grade == AnswerGrade.REFUSED_CORRECTLY
