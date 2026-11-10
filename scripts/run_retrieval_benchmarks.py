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
from src.relay.evaluation.quality_evaluator import QualityEvaluator
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.standin_client import StandInProvider
from src.relay.resilience.rate_limiter import ClientRateLimiter
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
        random_seed=42
    )
    rate_limiter = ClientRateLimiter(rpm=1000, tpm=1000000)
    orchestrator = PipelineOrchestrator(provider=provider, rate_limiter=rate_limiter)
    evaluator = QualityEvaluator()

    all_csv_rows: List[ResultsCsvRow] = []

    for cfg in RETRIEVAL_CONFIGS:
        print(f"\n==================================================")
        print(f"Running Retrieval Setting: {cfg.label} (Chunk={cfg.chunk_size}, TopK={cfg.top_k}, Rerank={cfg.rerank})")
        print(f"==================================================")

        correct_count = 0
        refused_correctly_count = 0
        refused_wrongly_count = 0
        wrong_count = 0
        answerable_total = sum(1 for item in eval_items if item.is_answerable)
        unanswerable_total = sum(1 for item in eval_items if not item.is_answerable)
        prompt_tokens_list = []
        gen_tokens_list = []

        for idx, item in enumerate(eval_items, 1):
            req_id = f"retrieval_{cfg.label}_q{idx:02d}"
            req = PipelineRequest(
                request_id=req_id,
                question=item.question,
                scenario=f"retrieval_{cfg.label}",
                source=SourceType.STANDIN,
                retrieval_setting=cfg
            )

            # Runtime pipeline invocation: zero evaluation leakage
            resp: PipelineResponse = await orchestrator.execute_request(req)

            # Independent grading
            grade, rationale = evaluator.grade(resp, item)
            resp.answer_grade = grade

            prompt_tokens_list.append(resp.tokens_in)
            gen_tokens_list.append(resp.tokens_out)

            if grade == AnswerGrade.CORRECT:
                correct_count += 1
            elif grade == AnswerGrade.REFUSED_CORRECTLY:
                refused_correctly_count += 1
            elif grade == AnswerGrade.REFUSED_WRONGLY:
                refused_wrongly_count += 1
            else:
                wrong_count += 1

            grade_str = grade.value if grade else ""

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
                answer_grade=grade_str,
                queue_wait_ms=resp.queue_wait_ms,
                processing_ms=resp.processing_ms,
                retry_wait_ms=resp.retry_wait_ms,
                model="stand-in-calibrated"
            )
            all_csv_rows.append(csv_row)

            grade_symbol = "[PASS]" if grade in {AnswerGrade.CORRECT, AnswerGrade.REFUSED_CORRECTLY} else "[FAIL]"
            print(f"  [{idx:02d}/{len(eval_items)}] {grade_symbol} Grade: {grade_str:<17} | {resp.total_ms:6.1f}ms | In={resp.tokens_in:<4} Out={resp.tokens_out:<3} | {item.question[:50]}...")

        # Calculate percentages
        correct_pct = (correct_count / max(1, answerable_total)) * 100.0
        refused_pct = (refused_correctly_count / max(1, unanswerable_total)) * 100.0
        wrong_pct = (wrong_count / len(eval_items)) * 100.0
        avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list)
        avg_gen_tokens = sum(gen_tokens_list) / len(gen_tokens_list)

        print(f"\n--- Setting Summary: {cfg.label} ---")
        print(f"  Answerable Correct: {correct_count}/{answerable_total} ({correct_pct:.1f}%)")
        print(f"  Correctly Refused:  {refused_correctly_count}/{unanswerable_total} ({refused_pct:.1f}%)")
        print(f"  Refused Wrongly:    {refused_wrongly_count}/{answerable_total}")
        print(f"  Wrong / Missed:     {wrong_count}/{len(eval_items)} ({wrong_pct:.1f}%)")
        print(f"  Tokens per request: Prompt={avg_prompt_tokens:.1f}, Completion={avg_gen_tokens:.1f}, Total={avg_prompt_tokens + avg_gen_tokens:.1f}")

    return all_csv_rows


if __name__ == "__main__":
    rows = asyncio.run(run_benchmark())
    print(f"\nCompleted {len(rows)} retrieval evaluation queries.")
