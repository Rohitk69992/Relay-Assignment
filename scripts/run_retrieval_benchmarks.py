"""Executes comparative retrieval benchmarks across 3 configurations on the 18 evaluation questions."""

import asyncio
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.relay.config import settings
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.standin_client import StandInProvider
from src.relay.schemas import (
    AnswerGrade,
    EvaluationItem,
    PipelineRequest,
    PipelineResponse,
    ResultsCsvRow,
    RetrievalSetting,
    SourceType
)

RETRIEVAL_CONFIGS = [
    RetrievalSetting(chunk_size=300, chunk_overlap=50, top_k=3, rerank=False),
    RetrievalSetting(chunk_size=600, chunk_overlap=100, top_k=4, rerank=True),
    RetrievalSetting(chunk_size=1000, chunk_overlap=150, top_k=6, rerank=True)
]


async def run_benchmark():
    data_dir = settings.DATA_DIR
    eval_file = data_dir / "evaluation_set.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_items = [EvaluationItem(**item) for item in json.load(f)]

    print(f"Loaded {len(eval_items)} evaluation questions.")

    # Use calibrated stand-in provider with high throughput for retrieval quality evaluation
    provider = StandInProvider(
        rpm=1000,
        tpm=1000000,
        p50_latency_ms=120.0,
        p95_latency_ms=280.0,
        tokens_per_sec=800.0,
        random_seed=42
    )
    from src.relay.resilience.rate_limiter import ClientRateLimiter
    rate_limiter = ClientRateLimiter(rpm=1000, tpm=1000000)
    orchestrator = PipelineOrchestrator(provider=provider, rate_limiter=rate_limiter)

    all_csv_rows: List[ResultsCsvRow] = []

    for cfg in RETRIEVAL_CONFIGS:
        print(f"\n==================================================")
        print(f"Running Retrieval Setting: {cfg.label} (Chunk={cfg.chunk_size}, TopK={cfg.top_k}, Rerank={cfg.rerank})")
        print(f"==================================================")

        correct_count = 0
        refused_correctly_count = 0
        wrong_count = 0
        answerable_total = sum(1 for item in eval_items if item.is_answerable)
        unanswerable_total = sum(1 for item in eval_items if not item.is_answerable)

        for idx, item in enumerate(eval_items, 1):
            req_id = f"retrieval_{cfg.label}_q{idx:02d}"
            req = PipelineRequest(
                request_id=req_id,
                question=item.question,
                scenario=f"retrieval_{cfg.label}",
                source=SourceType.STANDIN,
                retrieval_setting=cfg
            )

            resp: PipelineResponse = await orchestrator.execute_request(
                req,
                expected_answerable=item.is_answerable,
                expected_paper_id=item.expected_paper if item.is_answerable else None,
                expected_page=item.expected_page if item.is_answerable else None
            )

            if resp.answer_grade == AnswerGrade.CORRECT:
                correct_count += 1
            elif resp.answer_grade == AnswerGrade.REFUSED_CORRECTLY:
                refused_correctly_count += 1
            else:
                wrong_count += 1

            # Convert to ResultsCsvRow
            csv_row = ResultsCsvRow(
                request_id=resp.request_id,
                scenario=resp.scenario,
                source=resp.source.value,
                start_time=resp.start_time,
                first_token_ms=resp.first_token_ms,
                total_ms=resp.total_ms,
                status=resp.status.value,
                retries=resp.retries,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                retrieval_setting=cfg.label,
                answer_grade=resp.answer_grade.value,
                queue_wait_ms=resp.queue_wait_ms,
                processing_ms=resp.processing_ms,
                retry_wait_ms=resp.retry_wait_ms,
                model="stand-in-calibrated"
            )
            all_csv_rows.append(csv_row)

            grade_symbol = "[PASS]" if resp.answer_grade in {AnswerGrade.CORRECT, AnswerGrade.REFUSED_CORRECTLY} else "[FAIL]"
            print(f"  [{idx:02d}/{len(eval_items)}] {grade_symbol} Grade: {resp.answer_grade.value:<17} | {resp.total_ms:6.1f}ms | {item.question[:55]}...")

        # Calculate percentages
        correct_pct = (correct_count / max(1, answerable_total)) * 100.0
        refused_pct = (refused_correctly_count / max(1, unanswerable_total)) * 100.0
        wrong_pct = (wrong_count / len(eval_items)) * 100.0

        print(f"\n--- Setting Summary: {cfg.label} ---")
        print(f"  Answerable Correct: {correct_count}/{answerable_total} ({correct_pct:.1f}%)")
        print(f"  Correctly Refused:  {refused_correctly_count}/{unanswerable_total} ({refused_pct:.1f}%)")
        print(f"  Wrong / Hallucinated: {wrong_count}/{len(eval_items)} ({wrong_pct:.1f}%)")

    return all_csv_rows


if __name__ == "__main__":
    rows = asyncio.run(run_benchmark())
    print(f"\nCompleted {len(rows)} retrieval evaluation queries.")
