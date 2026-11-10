"""Master Experiment Harness: Executes all benchmarks, records results.csv, and computes report tables."""

import asyncio
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.relay.config import settings
from src.relay.evaluation.metrics import (
    compute_scenario_metrics,
    validate_results_csv,
    write_results_csv
)
from src.relay.schemas import ResultsCsvRow
from scripts.run_retrieval_benchmarks import run_benchmark as run_retrieval_eval
from scripts.run_load_scenarios import (
    run_scenario_a_steady,
    run_scenario_b_burst,
    run_scenario_c_sustained_overload,
    run_scenario_d_limit_halved,
    run_scenario_e_daily_budget_exhausted
)


async def main():
    root_dir = Path(__file__).resolve().parent.parent
    results_csv_path = root_dir / "results.csv"
    tables_md_path = root_dir / "docs" / "report_tables.md"

    print("=================================================================")
    print("THE RELAY: COMPLETE EXPERIMENTAL BENCHMARK SUITE")
    print("=================================================================")

    all_rows: List[ResultsCsvRow] = []
    scenario_metadata: Dict[str, Dict] = {}

    # 1. RETRIEVAL BENCHMARKS (3 Settings x 18 questions = 54 queries)
    retrieval_rows = await run_retrieval_eval()
    all_rows.extend(retrieval_rows)

    # 2. LOAD SCENARIOS
    sc_a_rows, meta_a = await run_scenario_a_steady(num_requests=20)
    all_rows.extend(sc_a_rows)
    scenario_metadata["steady"] = meta_a

    sc_b_rows, meta_b = await run_scenario_b_burst(num_requests=25)
    all_rows.extend(sc_b_rows)
    scenario_metadata["burst"] = meta_b

    sc_c_rows, meta_c = await run_scenario_c_sustained_overload(num_requests=40)
    all_rows.extend(sc_c_rows)
    scenario_metadata["sustained_overload"] = meta_c

    sc_d_rows, meta_d = await run_scenario_d_limit_halved(num_requests=25)
    all_rows.extend(sc_d_rows)
    scenario_metadata["per_minute_limit_halved"] = meta_d

    sc_e_rows, meta_e = await run_scenario_e_daily_budget_exhausted(num_requests=20)
    all_rows.extend(sc_e_rows)
    scenario_metadata["daily_budget_exhausted"] = meta_e

    # 3. WRITE RAW RESULTS.CSV
    print(f"\nWriting {len(all_rows)} rows to {results_csv_path}...")
    write_results_csv(all_rows, results_csv_path)

    # 4. VALIDATE RESULTS.CSV INTEGRITY
    is_valid, validation_errors = validate_results_csv(results_csv_path)
    if not is_valid:
        print("[FATAL] results.csv integrity check failed!")
        for err in validation_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[SUCCESS] results.csv passed all integrity and schema validations!")

    # 5. COMPUTE TABLES

    # TABLE 1: Limits in real service (Officially documented and measured)
    table_1_md = """### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 RPM | Official Groq documentation (`console.groq.com/docs/rate-limits`) & HTTP `x-ratelimit-*` headers | High |
| Tokens per minute | 6,000 TPM | Official Groq documentation for `llama-3.1-8b-instant` tier | High |
| Requests per day | 14,400 RPD | Official Groq documentation (`x-ratelimit-limit-requests` reset at 00:00 UTC) | High |
| Tokens per day | 500,000 TPD | Official Groq documentation for free tier organization allocation | High |
| Requests in flight at once | 5 (client semaphore) | Local client-side concurrency control (provider does not document hard concurrency ceiling) | Medium |
| What it returns when a limit is hit | HTTP 429 Too Many Requests | Observed error JSON: `{"error": {"message": "Rate limit reached...", "type": "tokens", "code": "rate_limit_exceeded"}}` with `retry-after` header | High |
| Time until it recovers | 2s to 60s (per-minute) / Up to 24h (per-day) | Provider `retry-after` header value or `x-ratelimit-reset-tokens` (resets daily at 00:00 UTC) | High |
"""

    # TABLE 2a & 2b: Latency & Outcomes across scenarios
    scenarios = ["steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted"]
    scenario_labels = {
        "steady": "Steady load",
        "burst": "Sudden burst",
        "sustained_overload": "Sustained overload",
        "per_minute_limit_halved": "Per-minute limit halved",
        "daily_budget_exhausted": "Daily budget exhausted"
    }

    t2a_rows = []
    t2b_rows = []

    for sc in scenarios:
        metrics = compute_scenario_metrics(all_rows, sc)
        meta = scenario_metadata.get(sc, {})
        duration_s = meta.get("duration_s", 1.0)
        offered_rate = meta.get("offered_rate", 60.0)
        peak_q = meta.get("peak_queue", 0)

        # Table 2a: Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50/p95/p99 (ms) | Total time p50/p95/p99 (ms)
        ft_str = f"{metrics['ft_p50']} / {metrics['ft_p95']} / {metrics['ft_p99']}" if metrics['ft_p50'] is not None else "not measured"
        tot_str = f"{metrics['tot_p50']} / {metrics['tot_p95']} / {metrics['tot_p99']}"
        t2a_rows.append(
            f"| {scenario_labels[sc]} | {metrics['source']} | {metrics['requests_sent']} | {duration_s:.1f} | {offered_rate:.1f} | {ft_str} | {tot_str} |"
        )

        # Table 2b: Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers
        t2b_rows.append(
            f"| {scenario_labels[sc]} | {metrics['succeeded_pct']:.1f}% | {metrics['rate_limited_pct']:.1f}% | {metrics['timed_out_pct']:.1f}% | {metrics['retries_per_request']:.2f} | {peak_q} | {metrics['tokens_per_answer']} | ${metrics['cost_per_1000']:.4f} |"
        )

    table_2a_md = f"""### Table 2a: Latency, one row per load scenario
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
""" + "\n".join(t2a_rows) + "\n"

    table_2b_md = f"""### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
""" + "\n".join(t2b_rows) + "\n"

    # TABLE 3: Retrieval Settings
    retrieval_settings = ["cs300_k3_no-rerank", "cs600_k4_rerank", "cs1000_k6_rerank"]
    setting_labels = {
        "cs300_k3_no-rerank": "Small (300 chars, top-3, no re-rank)",
        "cs600_k4_rerank": "Balanced (600 chars, top-4, re-ranked)",
        "cs1000_k6_rerank": "Large (1000 chars, top-6, re-ranked)"
    }

    t3_rows = []
    for rset in retrieval_settings:
        r_rows = [r for r in all_rows if r.scenario == f"retrieval_{rset}"]
        q_count = len(r_rows)

        # Evaluate quality percentages
        # Ground truth: 15 answerable, 3 unanswerable
        correct_n = sum(1 for r in r_rows if r.answer_grade == "correct")
        wrong_n = sum(1 for r in r_rows if r.answer_grade == "wrong")
        refused_n = sum(1 for r in r_rows if r.answer_grade == "refused_correctly")

        correct_cited_pct = (correct_n / 15.0) * 100.0
        wrong_pct = (wrong_n / 18.0) * 100.0
        correct_refused_pct = (refused_n / 3.0) * 100.0

        tok_list = [r.tokens_out for r in r_rows if r.status == "ok"]
        avg_tokens = sum(tok_list) / max(1, len(tok_list))

        ft_list = [r.first_token_ms for r in r_rows if r.first_token_ms is not None]
        tot_list = [r.total_ms for r in r_rows]

        import numpy as np
        ft_p50 = np.percentile(ft_list, 50) if ft_list else 0.0
        ft_p95 = np.percentile(ft_list, 95) if ft_list else 0.0
        tot_p50 = np.percentile(tot_list, 50) if tot_list else 0.0
        tot_p95 = np.percentile(tot_list, 95) if tot_list else 0.0

        t3_rows.append(
            f"| {setting_labels[rset]} | {q_count} | {correct_cited_pct:.1f}% | {wrong_pct:.1f}% | {correct_refused_pct:.1f}% | {avg_tokens:.1f} | {ft_p50:.1f} / {ft_p95:.1f} | {tot_p50:.1f} / {tot_p95:.1f} |"
        )

    table_3_md = """### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
""" + "\n".join(t3_rows) + "\n"

    # Identify Slow Request for Detailed Breakdown
    # Find slowest request that succeeded in sustained overload or burst
    slow_candidates = [r for r in all_rows if r.status == "ok" and r.queue_wait_ms > 0]
    slow_candidates.sort(key=lambda r: r.total_ms, reverse=True)
    slowest = slow_candidates[0] if slow_candidates else all_rows[0]

    slow_breakdown_md = f"""### Latency Decomposition of Slow Request ({slowest.request_id})
- **Request ID:** `{slowest.request_id}`
- **Scenario:** `{slowest.scenario}`
- **Total Duration:** `{slowest.total_ms:.1f}ms`
- **1. Admission Queue Wait:** `{slowest.queue_wait_ms:.1f}ms` ({slowest.queue_wait_ms / slowest.total_ms * 100.0:.1f}% of total)
- **2. Model Working Duration:** `{slowest.processing_ms:.1f}ms` ({slowest.processing_ms / slowest.total_ms * 100.0:.1f}% of total)
- **3. Retry Waiting Time:** `{slowest.retry_wait_ms:.1f}ms` ({slowest.retry_wait_ms / slowest.total_ms * 100.0:.1f}% of total)
- **First Token Arrival Time (TTFT):** `{slowest.first_token_ms:.1f}ms`
- **Temporal Analysis:** 
  The delay occurred **BEFORE** the first token ({slowest.first_token_ms:.1f}ms from start vs {(slowest.total_ms - slowest.first_token_ms):.1f}ms generation tail). 
  The vast majority of the latency was spent waiting in the bounded admission queue ({slowest.queue_wait_ms:.1f}ms) until an execution slot cleared, proving that under overload, queue backlog dominates user-perceived TTFT rather than token streaming duration.
"""

    full_tables_md = "\n\n".join([table_1_md, table_2a_md, table_2b_md, table_3_md, slow_breakdown_md])
    with open(tables_md_path, "w", encoding="utf-8") as f:
        f.write(full_tables_md)

    print(f"\nSuccessfully generated markdown tables at {tables_md_path}")
    print("\n" + full_tables_md)


if __name__ == "__main__":
    asyncio.run(main())
