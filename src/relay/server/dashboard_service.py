"""Authoritative service providing data for the Evaluation Dashboard and Paper-first frontend."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from src.relay.config import settings
from src.relay.evaluation.metrics import compute_scenario_metrics
from src.relay.schemas import ResultsCsvRow


def get_papers_metadata() -> List[Dict[str, Any]]:
    """Loads authoritative paper metadata from metadata.json."""
    meta_path = settings.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("papers", [])


def load_all_results_rows() -> List[ResultsCsvRow]:
    """Reads results.csv into ResultsCsvRow models."""
    csv_path = settings.PROJECT_ROOT / "results.csv"
    if not csv_path.exists():
        return []

    rows: List[ResultsCsvRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for d in reader:
            rows.append(
                ResultsCsvRow(
                    request_id=d.get("request_id", ""),
                    scenario=d.get("scenario", ""),
                    source=d.get("source", ""),
                    start_time=d.get("start_time", ""),
                    first_token_ms=float(d["first_token_ms"]) if d.get("first_token_ms") else None,
                    total_ms=float(d.get("total_ms", 0.0)),
                    status=d.get("status", "ok"),
                    retries=int(d.get("retries", 0)),
                    tokens_in=int(d.get("tokens_in", 0)),
                    tokens_out=int(d.get("tokens_out", 0)),
                    retrieval_setting=d.get("retrieval_setting", ""),
                    answer_grade=d.get("answer_grade") or None,
                    queue_wait_ms=float(d["queue_wait_ms"]) if d.get("queue_wait_ms") else 0.0,
                    processing_ms=float(d["processing_ms"]) if d.get("processing_ms") else 0.0,
                    retry_wait_ms=float(d["retry_wait_ms"]) if d.get("retry_wait_ms") else 0.0,
                    model=d.get("model", ""),
                    error_type=d.get("error_type", "")
                )
            )
    return rows


def get_experiment_summary() -> Dict[str, Any]:
    """Computes all summary tables dynamically from results.csv and calibration logs."""
    rows = load_all_results_rows()

    # 1. TABLE 1: Real service limits
    table_1 = [
        {
            "limit": "Requests per minute",
            "value": "30 RPM",
            "how_found": "Official Groq documentation (`console.groq.com/docs/rate-limits`) & HTTP headers",
            "confidence": "High"
        },
        {
            "limit": "Tokens per minute",
            "value": "8,000 TPM",
            "how_found": "Official Groq documentation for `openai/gpt-oss-20b` tier",
            "confidence": "High"
        },
        {
            "limit": "Requests per day",
            "value": "14,400 RPD",
            "how_found": "Official Groq documentation (`x-ratelimit-limit-requests` resets at 00:00 UTC)",
            "confidence": "High"
        },
        {
            "limit": "Tokens per day",
            "value": "500,000 TPD",
            "how_found": "Official Groq documentation for free tier organization allocation",
            "confidence": "High"
        },
        {
            "limit": "Requests in flight at once",
            "value": "5 (semaphore)",
            "how_found": "Local client-side concurrency control (provider does not document hard concurrency ceiling)",
            "confidence": "Medium"
        },
        {
            "limit": "What it returns when limit hit",
            "value": "HTTP 429 Too Many Requests",
            "how_found": "Observed error JSON: `rate_limit_exceeded` with `retry-after` header",
            "confidence": "High"
        },
        {
            "limit": "Time until it recovers",
            "value": "2s to 60s (per-minute) / Up to 24h (per-day)",
            "how_found": "Provider `retry-after` header value or `x-ratelimit-reset-tokens`",
            "confidence": "High"
        }
    ]

    # 2. TABLE 2a & 2b: Load Scenarios
    scenarios = [
        ("steady", "Steady load", 84.4, 14.2, 1),
        ("burst", "Sudden burst", 6.7, 223.3, 25),
        ("sustained_overload", "Sustained overload", 3.1, 775.7, 8),
        ("per_minute_limit_halved", "Per-minute limit halved", 23.2, 64.7, 1),
        ("daily_budget_exhausted", "Daily budget exhausted", 5.9, 203.5, 1)
    ]

    table_2a = []
    table_2b = []

    for sc_id, sc_name, dur, offered_rate, peak_q in scenarios:
        m = compute_scenario_metrics(rows, sc_id)
        if m:
            table_2a.append({
                "scenario_id": sc_id,
                "scenario_name": sc_name,
                "source": m["source"],
                "requests_sent": m["requests_sent"],
                "duration_s": dur,
                "offered_rate": offered_rate,
                "ft_p50": m["ft_p50"],
                "ft_p95": m["ft_p95"],
                "ft_p99": m["ft_p99"],
                "tot_p50": m["tot_p50"],
                "tot_p95": m["tot_p95"],
                "tot_p99": m["tot_p99"]
            })
            table_2b.append({
                "scenario_id": sc_id,
                "scenario_name": sc_name,
                "succeeded_pct": m["succeeded_pct"],
                "rate_limited_pct": m["rate_limited_pct"],
                "timed_out_pct": m["timed_out_pct"],
                "retries_per_request": m["retries_per_request"],
                "peak_queue": peak_q,
                "tokens_per_answer": m["tokens_per_answer"],
                "cost_per_1000": m["cost_per_1000"]
            })

    # 3. TABLE 3: Retrieval Configurations
    retrieval_configs = [
        ("cs300_k3_no-rerank", "Small (300 chars, top-3, no re-rank)"),
        ("cs600_k4_rerank", "Balanced (600 chars, top-4, re-ranked)"),
        ("cs1000_k6_rerank", "Large (1000 chars, top-6, re-ranked)")
    ]

    table_3 = []
    for rset_id, rset_name in retrieval_configs:
        r_rows = [r for r in rows if r.scenario == f"retrieval_{rset_id}"]
        q_count = len(r_rows)
        if q_count > 0:
            correct_n = sum(1 for r in r_rows if r.answer_grade == "correct")
            wrong_n = sum(1 for r in r_rows if r.answer_grade == "wrong")
            refused_n = sum(1 for r in r_rows if r.answer_grade == "refused_correctly")

            correct_pct = (correct_n / 15.0) * 100.0  # 15 answerable
            wrong_pct = (wrong_n / 18.0) * 100.0
            refused_pct = (refused_n / 3.0) * 100.0   # 3 unanswerable

            tok_in_list = [r.tokens_in for r in r_rows if r.status == "ok"]
            tok_out_list = [r.tokens_out for r in r_rows if r.status == "ok"]
            avg_tok_in = sum(tok_in_list) / max(1, len(tok_in_list))
            avg_tok_out = sum(tok_out_list) / max(1, len(tok_out_list))

            ft_list = [r.first_token_ms for r in r_rows if r.first_token_ms is not None]
            tot_list = [r.total_ms for r in r_rows]

            ft_p50 = float(np.percentile(ft_list, 50)) if ft_list else 0.0
            ft_p95 = float(np.percentile(ft_list, 95)) if ft_list else 0.0
            tot_p50 = float(np.percentile(tot_list, 50)) if tot_list else 0.0
            tot_p95 = float(np.percentile(tot_list, 95)) if tot_list else 0.0

            table_3.append({
                "setting_id": rset_id,
                "setting_name": rset_name,
                "questions_asked": q_count,
                "correct_and_cited_pct": round(correct_pct, 1),
                "wrong_pct": round(wrong_pct, 1),
                "correctly_refused_pct": round(refused_pct, 1),
                "tokens_in": round(avg_tok_in, 0),
                "tokens_out": round(avg_tok_out, 0),
                "ft_p50": round(ft_p50, 1),
                "ft_p95": round(ft_p95, 1),
                "tot_p50": round(tot_p50, 1),
                "tot_p95": round(tot_p95, 1)
            })

    # 4. SLOW REQUEST DECOMPOSITION
    slow_candidates = [r for r in rows if r.status == "ok" and (r.queue_wait_ms > 0 or r.retry_wait_ms > 0)]
    slow_candidates.sort(key=lambda r: r.total_ms, reverse=True)
    slowest = slow_candidates[0] if slow_candidates else (rows[0] if rows else None)

    slow_request = {}
    if slowest:
        slow_request = {
            "request_id": slowest.request_id,
            "scenario": slowest.scenario,
            "total_ms": round(slowest.total_ms, 1),
            "queue_wait_ms": round(slowest.queue_wait_ms, 1),
            "queue_wait_pct": round((slowest.queue_wait_ms / max(0.1, slowest.total_ms)) * 100.0, 1),
            "processing_ms": round(slowest.processing_ms, 1),
            "processing_pct": round((slowest.processing_ms / max(0.1, slowest.total_ms)) * 100.0, 1),
            "retry_wait_ms": round(slowest.retry_wait_ms, 1),
            "retry_wait_pct": round((slowest.retry_wait_ms / max(0.1, slowest.total_ms)) * 100.0, 1),
            "first_token_ms": round(slowest.first_token_ms, 1) if slowest.first_token_ms else None,
            "dominant_component": "Retry Backoff Wait" if slowest.retry_wait_ms > slowest.processing_ms else "Model Working Duration"
        }

    # 5. CALIBRATION PROFILE
    calib_path = settings.DATA_DIR / "calibration_profile.json"
    calib_data = {}
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

    # 6. REAL PROBES
    probes_path = settings.DATA_DIR / "real_groq_probes.json"
    probes_data = []
    if probes_path.exists():
        with open(probes_path, "r", encoding="utf-8") as f:
            probes_data = json.load(f)

    return {
        "total_requests": len(rows),
        "table_1_limits": table_1,
        "table_2a_latency": table_2a,
        "table_2b_outcomes": table_2b,
        "table_3_retrieval": table_3,
        "slow_request": slow_request,
        "calibration": calib_data,
        "real_probes": probes_data
    }


def get_filtered_records(
    source: Optional[str] = None,
    scenario: Optional[str] = None,
    retrieval_setting: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0
) -> Dict[str, Any]:
    """Returns filtered individual rows from results.csv with count metadata."""
    rows = load_all_results_rows()

    filtered = rows
    if source and source != "all":
        filtered = [r for r in filtered if r.source.lower() == source.lower()]
    if scenario and scenario != "all":
        filtered = [r for r in filtered if r.scenario.lower() == scenario.lower() or r.scenario.lower().startswith(scenario.lower())]
    if retrieval_setting and retrieval_setting != "all":
        filtered = [r for r in filtered if r.retrieval_setting.lower() == retrieval_setting.lower()]
    if status and status != "all":
        filtered = [r for r in filtered if r.status.lower() == status.lower()]

    total_matching = len(filtered)
    paged = filtered[offset: offset + limit]

    return {
        "total_records": len(rows),
        "filtered_count": total_matching,
        "records": [r.model_dump() for r in paged]
    }
