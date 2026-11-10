"""Regression test proving zero evaluation leakage in The Relay pipeline."""

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
async def test_zero_evaluation_leakage_guarantee():
    """Verifies that the orchestrator receives only the user request and has zero access to ground truth."""
    standin = StandInProvider(p50_latency_ms=15.0, p95_latency_ms=30.0, tokens_per_sec=4000.0)
    orchestrator = PipelineOrchestrator(provider=standin)
    evaluator = QualityEvaluator()

    question = "What is the number of heads (h) and dimension of keys (d_k) in the base Transformer model?"
    req = PipelineRequest(
        request_id="leakage_test_01",
        question=question,
        scenario="leakage_audit",
        source=SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
    )

    # Runtime call takes only request
    response = await orchestrator.execute_request(req)

    assert response.status == RequestStatus.OK
    # PipelineResponse answer_grade is None by design at runtime
    assert response.answer_grade is None
    assert len(response.citations) > 0
    assert response.citations[0].paper_id == "1706.03762v7"
    assert response.citations[0].page in {4, 5}

    # External evaluation is performed strictly by QualityEvaluator
    ground_truth = EvaluationItem(
        question_id="Q02_FACT_ATTN_HEADS",
        category="Direct factual question",
        question=question,
        expected_answer="The base model uses h = 8 parallel attention heads, with d_k = d_v = 64.",
        expected_paper="1706.03762v7",
        expected_page=4,
        expected_section="3.2.2 Multi-Head Attention",
        is_answerable=True
    )

    grade, rationale = evaluator.grade(response, ground_truth)
    assert grade == AnswerGrade.CORRECT
    assert "matched" in rationale.lower()


@pytest.mark.asyncio
async def test_operational_failure_not_graded_as_wrong():
    """Verifies that rate-limited or timed-out requests are not misclassified as factual model hallucinations."""
    evaluator = QualityEvaluator()
    from src.relay.schemas import PipelineResponse

    # Create a simulated dropped response
    dropped_resp = PipelineResponse(
        request_id="drop_test_01",
        scenario="load_test",
        source=SourceType.STANDIN,
        start_time="2026-11-10T12:00:00Z",
        answer="",
        status=RequestStatus.RATE_LIMITED,
        answer_grade=None,
        first_token_ms=None,
        total_ms=5.0,
        tokens_in=0,
        tokens_out=0,
        retrieval_setting="cs600_k4_rerank",
        error_message="Admission queue depth exceeded capacity"
    )

    ground_truth = EvaluationItem(
        question_id="Q01_TEST",
        category="factual",
        question="Some question",
        expected_answer="Some answer",
        expected_paper="1706.03762v7",
        expected_page=4,
        expected_section="Section 3",
        is_answerable=True
    )

    grade, rationale = evaluator.grade(dropped_resp, ground_truth)
    # Must NOT be AnswerGrade.WRONG
    assert grade is None
    assert "operational drop" in rationale.lower()
