"""Metric calculation, CSV serialization, and summary table generator."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.relay.config import settings
from src.relay.schemas import AnswerGrade, RequestStatus, ResultsCsvRow


REQUIRED_CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

EXTENDED_CSV_COLUMNS = REQUIRED_CSV_COLUMNS + [
    "queue_wait_ms",
    "processing_ms",
    "retry_wait_ms",
    "model",
    "error_type"
]


def write_results_csv(rows: List[ResultsCsvRow], filepath: Path) -> None:
    """Writes results rows to CSV ensuring exact schema."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTENDED_CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            data = r.model_dump()
            # Format floats to 1 decimal place or empty
            for k in ["first_token_ms", "total_ms", "queue_wait_ms", "processing_ms", "retry_wait_ms"]:
                if data.get(k) is not None:
                    data[k] = round(float(data[k]), 1)
                else:
                    data[k] = ""
            writer.writerow(data)


def validate_results_csv(filepath: Path) -> Tuple[bool, List[str]]:
    """Validates structural and semantic integrity of results.csv."""
    errors = []
    if not filepath.exists():
        return False, [f"File {filepath} does not exist."]

    seen_ids = set()
    valid_statuses = {s.value for s in RequestStatus}
    valid_grades = {g.value for g in AnswerGrade}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for req_col in REQUIRED_CSV_COLUMNS:
            if req_col not in reader.fieldnames:
                errors.append(f"Missing mandatory column: {req_col}")

        for idx, row in enumerate(reader, start=2):
            req_id = row.get("request_id", "").strip()
            if not req_id:
                errors.append(f"Row {idx}: missing request_id")
            elif req_id in seen_ids:
                errors.append(f"Row {idx}: duplicate request_id '{req_id}'")
            else:
                seen_ids.add(req_id)

            status = row.get("status", "").strip()
            if status not in valid_statuses:
                errors.append(f"Row {idx}: invalid status '{status}'")

            grade = row.get("answer_grade", "").strip()
            if grade not in valid_grades:
                errors.append(f"Row {idx}: invalid answer_grade '{grade}'")

            source = row.get("source", "").strip()
            if source not in {"real", "stand-in"}:
                errors.append(f"Row {idx}: invalid source '{source}' (must be 'real' or 'stand-in')")

            try:
                tot_ms = float(row.get("total_ms", 0))
                if tot_ms < 0:
                    errors.append(f"Row {idx}: negative total_ms {tot_ms}")
            except ValueError:
                errors.append(f"Row {idx}: invalid total_ms value")

            ft_str = row.get("first_token_ms", "").strip()
            if ft_str:
                try:
                    ft_ms = float(ft_str)
                    if ft_ms < 0:
                        errors.append(f"Row {idx}: negative first_token_ms {ft_ms}")
                except ValueError:
                    errors.append(f"Row {idx}: invalid first_token_ms value")

    return len(errors) == 0, errors


def compute_scenario_metrics(rows: List[ResultsCsvRow], scenario_name: str) -> Dict[str, Any]:
    """Computes summary latency and outcome percentiles for a given scenario."""
    sc_rows = [r for r in rows if r.scenario == scenario_name]
    if not sc_rows:
        return {}

    source = sc_rows[0].source
    requests_sent = len(sc_rows)

    # Compute duration in seconds between earliest start and latest end
    total_latencies = [r.total_ms for r in sc_rows]
    first_token_latencies = [r.first_token_ms for r in sc_rows if r.first_token_ms is not None]

    succeeded = sum(1 for r in sc_rows if r.status == "ok")
    rate_limited = sum(1 for r in sc_rows if r.status == "rate_limited")
    timed_out = sum(1 for r in sc_rows if r.status == "timeout")
    errors = sum(1 for r in sc_rows if r.status == "error")

    retries_list = [r.retries for r in sc_rows]
    avg_retries = sum(retries_list) / max(1, requests_sent)

    successful_rows = [r for r in sc_rows if r.status == "ok"]
    total_tokens_out = sum(r.tokens_out for r in successful_rows)
    tokens_per_answer = total_tokens_out / max(1, len(successful_rows))

    # Pricing estimation
    total_tokens_in = sum(r.tokens_in for r in successful_rows)
    cost_per_req = (
        (total_tokens_in / 1_000_000.0) * settings.INPUT_PRICE_PER_MILLION_USD +
        (total_tokens_out / 1_000_000.0) * settings.OUTPUT_PRICE_PER_MILLION_USD
    ) / max(1, len(successful_rows))
    cost_per_1000 = cost_per_req * 1000.0

    # Percentiles
    ft_p50 = np.percentile(first_token_latencies, 50) if first_token_latencies else None
    ft_p95 = np.percentile(first_token_latencies, 95) if first_token_latencies else None
    ft_p99 = np.percentile(first_token_latencies, 99) if first_token_latencies else None

    tot_p50 = np.percentile(total_latencies, 50) if total_latencies else 0.0
    tot_p95 = np.percentile(total_latencies, 95) if total_latencies else 0.0
    tot_p99 = np.percentile(total_latencies, 99) if total_latencies else 0.0

    return {
        "scenario": scenario_name,
        "source": source,
        "requests_sent": requests_sent,
        "succeeded_pct": (succeeded / requests_sent) * 100.0,
        "rate_limited_pct": (rate_limited / requests_sent) * 100.0,
        "timed_out_pct": (timed_out / requests_sent) * 100.0,
        "retries_per_request": avg_retries,
        "tokens_per_answer": round(tokens_per_answer, 1),
        "cost_per_1000": round(cost_per_1000, 4),
        "ft_p50": round(ft_p50, 1) if ft_p50 is not None else None,
        "ft_p95": round(ft_p95, 1) if ft_p95 is not None else None,
        "ft_p99": round(ft_p99, 1) if ft_p99 is not None else None,
        "tot_p50": round(tot_p50, 1),
        "tot_p95": round(tot_p95, 1),
        "tot_p99": round(tot_p99, 1)
    }
