"""Unit tests for metric calculation and CSV validation."""

from pathlib import Path
import pytest
from src.relay.evaluation.metrics import (
    compute_scenario_metrics,
    validate_results_csv,
    write_results_csv
)
from src.relay.schemas import AnswerGrade, RequestStatus, ResultsCsvRow


def test_write_and_validate_results_csv(tmp_path: Path):
    test_csv = tmp_path / "test_results.csv"

    rows = [
        ResultsCsvRow(
            request_id="req-1",
            scenario="steady",
            source="stand-in",
            start_time="2026-11-10T10:00:00Z",
            first_token_ms=120.5,
            total_ms=450.2,
            status=RequestStatus.OK.value,
            retries=0,
            tokens_in=250,
            tokens_out=85,
            retrieval_setting="cs600_k4_rerank",
            answer_grade=AnswerGrade.CORRECT.value
        ),
        ResultsCsvRow(
            request_id="req-2",
            scenario="steady",
            source="stand-in",
            start_time="2026-11-10T10:00:01Z",
            first_token_ms=None,
            total_ms=105.0,
            status=RequestStatus.RATE_LIMITED.value,
            retries=1,
            tokens_in=200,
            tokens_out=0,
            retrieval_setting="cs600_k4_rerank",
            answer_grade=AnswerGrade.WRONG.value
        )
    ]

    write_results_csv(rows, test_csv)
    is_valid, errors = validate_results_csv(test_csv)
    assert is_valid is True, f"Validation errors: {errors}"
    assert len(errors) == 0


def test_compute_scenario_metrics():
    rows = [
        ResultsCsvRow(
            request_id=f"req-{i}",
            scenario="burst",
            source="stand-in",
            start_time="2026-11-10T10:00:00Z",
            first_token_ms=100.0 + i * 10,
            total_ms=300.0 + i * 20,
            status=RequestStatus.OK.value if i < 8 else RequestStatus.TIMEOUT.value,
            retries=0 if i < 8 else 2,
            tokens_in=300,
            tokens_out=100,
            retrieval_setting="cs600_k4_rerank",
            answer_grade=AnswerGrade.CORRECT.value if i < 8 else AnswerGrade.WRONG.value
        )
        for i in range(10)
    ]

    metrics = compute_scenario_metrics(rows, "burst")
    assert metrics["requests_sent"] == 10
    assert metrics["succeeded_pct"] == 80.0
    assert metrics["timed_out_pct"] == 20.0
    assert metrics["ft_p50"] is not None
    assert metrics["tot_p50"] > 0
