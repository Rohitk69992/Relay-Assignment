"""Executes the 5 required load scenarios (Steady, Burst, Sustained Overload, Limit Halved, Daily Budget Exhausted)."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.relay.config import settings
from src.relay.logging_setup import get_logger
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.provider.standin_client import StandInProvider
from src.relay.resilience.queue_manager import AdmissionQueueManager
from src.relay.resilience.rate_limiter import ClientRateLimiter
from src.relay.schemas import (
    AnswerGrade,
    PipelineRequest,
    PipelineResponse,
    ResultsCsvRow,
    RetrievalSetting,
    SourceType
)

logger = get_logger("relay.load_test")


async def run_scenario_a_steady(num_requests: int = 15) -> Tuple[List[ResultsCsvRow], Dict]:
    """Scenario A: Steady Load - Regular request arrival within provider capacity."""
    print("\n" + "=" * 60)
    print("SCENARIO A: STEADY LOAD (Offered load within capacity)")
    print("=" * 60)

    # RPM = 30; requests spaced at 2.1s (offered rate ~28.5 req/min < 30)
    provider = StandInProvider(rpm=30, tpm=10000, p50_latency_ms=120.0, p95_latency_ms=280.0, tokens_per_sec=800.0, random_seed=101)
    rate_limiter = ClientRateLimiter(rpm=30, tpm=10000)
    orchestrator = PipelineOrchestrator(provider=provider, rate_limiter=rate_limiter)

    test_queries = [
        "In Attention Is All You Need, why is dot product scaled by 1/sqrt(d_k)?",
        "What neural audio codec does Moshi use and what is its framerate?",
        "What boundary predictor is used in Dynamic Chunking?",
        "Explain the learning rate warmup schedule in the Transformer paper."
    ]

    rows: List[ResultsCsvRow] = []
    t0 = time.perf_counter()

    for i in range(num_requests):
        req_id = f"steady_req_{i+1:02d}"
        q = test_queries[i % len(test_queries)]
        req = PipelineRequest(
            request_id=req_id,
            question=q,
            scenario="steady",
            source=SourceType.STANDIN,
            retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
        )
        resp = await orchestrator.execute_request(req, expected_answerable=True)
        rows.append(
            ResultsCsvRow(
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
                retrieval_setting=resp.retrieval_setting,
                answer_grade=resp.answer_grade.value,
                queue_wait_ms=resp.queue_wait_ms,
                processing_ms=resp.processing_ms,
                retry_wait_ms=resp.retry_wait_ms,
                model="stand-in-calibrated"
            )
        )
        print(f"  [{i+1:02d}/{num_requests}] Status: {resp.status.value:<12} | Total: {resp.total_ms:5.1f}ms | TTFT: {resp.first_token_ms or 0.0:5.1f}ms")
        if i < num_requests - 1:
            await asyncio.sleep(2.1)

    duration_s = time.perf_counter() - t0
    offered_rate = (num_requests / duration_s) * 60.0
    print(f"Scenario A Complete: {num_requests} requests in {duration_s:.1f}s (Offered: {offered_rate:.1f} req/min)")
    return rows, {"duration_s": duration_s, "offered_rate": offered_rate, "peak_queue": orchestrator.queue_mgr.peak_queue_depth}


async def run_scenario_b_burst(num_requests: int = 20) -> Tuple[List[ResultsCsvRow], Dict]:
    """Scenario B: Sudden Burst - Sharp spike of concurrent demand at t=0."""
    print("\n" + "=" * 60)
    print("SCENARIO B: SUDDEN BURST (20 concurrent requests at t=0)")
    print("=" * 60)

    # Concurrency limit = 4, Queue limit = 40
    provider = StandInProvider(rpm=60, tpm=20000, p50_latency_ms=130.0, p95_latency_ms=320.0, load_slowdown_factor=0.08, random_seed=202)
    queue_mgr = AdmissionQueueManager(max_queue_size=40, max_concurrency=4, queue_timeout_seconds=15.0)
    rate_limiter = ClientRateLimiter(rpm=60, tpm=20000)
    orchestrator = PipelineOrchestrator(provider=provider, queue_mgr=queue_mgr, rate_limiter=rate_limiter)

    rows: List[ResultsCsvRow] = []
    t0 = time.perf_counter()

    async def send_burst_req(idx: int):
        req_id = f"burst_req_{idx+1:02d}"
        req = PipelineRequest(
            request_id=req_id,
            question="What is the parameter size of Helium and codec in Moshi?",
            scenario="burst",
            source=SourceType.STANDIN,
            retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
        )
        resp = await orchestrator.execute_request(req, expected_answerable=True)
        return ResultsCsvRow(
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
            retrieval_setting=resp.retrieval_setting,
            answer_grade=resp.answer_grade.value,
            queue_wait_ms=resp.queue_wait_ms,
            processing_ms=resp.processing_ms,
            retry_wait_ms=resp.retry_wait_ms,
            model="stand-in-calibrated"
        )

    tasks = [send_burst_req(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    rows.extend(results)

    duration_s = time.perf_counter() - t0
    offered_rate = (num_requests / duration_s) * 60.0
    print(f"Scenario B Complete: {num_requests} requests burst in {duration_s:.1f}s. Peak queue depth: {queue_mgr.peak_queue_depth}")
    return rows, {"duration_s": duration_s, "offered_rate": offered_rate, "peak_queue": queue_mgr.peak_queue_depth}


async def run_scenario_c_sustained_overload(num_requests: int = 35) -> Tuple[List[ResultsCsvRow], Dict]:
    """Scenario C: Sustained Overload - Offered traffic continuously exceeds processing capacity."""
    print("\n" + "=" * 60)
    print("SCENARIO C: SUSTAINED OVERLOAD (Rapid arrival exceeding queue & concurrency)")
    print("=" * 60)

    # Tight queue (8 max) and queue wait timeout (1.0s), concurrency 2
    provider = StandInProvider(rpm=100, tpm=50000, p50_latency_ms=180.0, p95_latency_ms=450.0, load_slowdown_factor=0.15, failure_rate=0.05, random_seed=303)
    queue_mgr = AdmissionQueueManager(max_queue_size=8, max_concurrency=2, queue_timeout_seconds=1.0)
    rate_limiter = ClientRateLimiter(rpm=100, tpm=50000)
    orchestrator = PipelineOrchestrator(provider=provider, queue_mgr=queue_mgr, rate_limiter=rate_limiter)

    rows: List[ResultsCsvRow] = []
    t0 = time.perf_counter()

    async def send_overload_req(idx: int):
        await asyncio.sleep(idx * 0.04)  # Arriving rapidly every 40ms
        req_id = f"overload_req_{idx+1:02d}"
        req = PipelineRequest(
            request_id=req_id,
            question="Compare quadratic self-attention with dynamic hierarchical chunking.",
            scenario="sustained_overload",
            source=SourceType.STANDIN,
            retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
        )
        resp = await orchestrator.execute_request(req, expected_answerable=True)
        return ResultsCsvRow(
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
            retrieval_setting=resp.retrieval_setting,
            answer_grade=resp.answer_grade.value,
            queue_wait_ms=resp.queue_wait_ms,
            processing_ms=resp.processing_ms,
            retry_wait_ms=resp.retry_wait_ms,
            model="stand-in-calibrated",
            error_type=resp.error_message
        )

    tasks = [send_overload_req(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    rows.extend(results)

    duration_s = time.perf_counter() - t0
    offered_rate = (num_requests / duration_s) * 60.0
    print(f"Scenario C Complete: {num_requests} requests in {duration_s:.1f}s. Peak queue: {queue_mgr.peak_queue_depth}, Queue full drops: {queue_mgr.total_rejected_queue_full}, Timeouts: {queue_mgr.total_dropped_queue_timeout}")
    return rows, {"duration_s": duration_s, "offered_rate": offered_rate, "peak_queue": queue_mgr.peak_queue_depth}


async def run_scenario_d_limit_halved(num_requests: int = 20) -> Tuple[List[ResultsCsvRow], Dict]:
    """Scenario D: Per-Minute Limit Halved - Stand-in RPM reduced by 50% (from 30 to 15)."""
    print("\n" + "=" * 60)
    print("SCENARIO D: PER-MINUTE LIMIT HALVED (RPM = 15)")
    print("=" * 60)

    # RPM = 15 instead of standard 30; 20 requests sent at interval of 0.8s (offered rate ~75 req/min)
    provider = StandInProvider(rpm=15, tpm=5000, p50_latency_ms=130.0, p95_latency_ms=300.0, random_seed=404)
    rate_limiter = ClientRateLimiter(rpm=15, tpm=5000)
    orchestrator = PipelineOrchestrator(provider=provider, rate_limiter=rate_limiter)

    rows: List[ResultsCsvRow] = []
    t0 = time.perf_counter()

    for i in range(num_requests):
        req_id = f"halved_req_{i+1:02d}"
        req = PipelineRequest(
            request_id=req_id,
            question="What is the BLEU score achieved by base and big Transformer models on English to German?",
            scenario="per_minute_limit_halved",
            source=SourceType.STANDIN,
            retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
        )
        resp = await orchestrator.execute_request(req, expected_answerable=True)
        rows.append(
            ResultsCsvRow(
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
                retrieval_setting=resp.retrieval_setting,
                answer_grade=resp.answer_grade.value,
                queue_wait_ms=resp.queue_wait_ms,
                processing_ms=resp.processing_ms,
                retry_wait_ms=resp.retry_wait_ms,
                model="stand-in-calibrated",
                error_type=resp.error_message
            )
        )
        if i < num_requests - 1:
            await asyncio.sleep(0.8)

    duration_s = time.perf_counter() - t0
    offered_rate = (num_requests / duration_s) * 60.0
    print(f"Scenario D Complete: {num_requests} requests in {duration_s:.1f}s.")
    return rows, {"duration_s": duration_s, "offered_rate": offered_rate, "peak_queue": orchestrator.queue_mgr.peak_queue_depth}


async def run_scenario_e_daily_budget_exhausted(num_requests: int = 20) -> Tuple[List[ResultsCsvRow], Dict]:
    """Scenario E: Daily Budget Exhausted - Simulates 3 PM cutoff when RPD quota is spent."""
    print("\n" + "=" * 60)
    print("SCENARIO E: DAILY BUDGET EXHAUSTED (Simulating 3 PM Cutoff)")
    print("=" * 60)

    # RPD = 5: first 5 succeed, remaining 15 hit daily quota immediately
    provider = StandInProvider(rpd=5, p50_latency_ms=100.0, random_seed=505)
    rate_limiter = ClientRateLimiter(rpd=5)
    orchestrator = PipelineOrchestrator(provider=provider, rate_limiter=rate_limiter)

    rows: List[ResultsCsvRow] = []
    t0 = time.perf_counter()

    for i in range(num_requests):
        req_id = f"daily_exhaust_req_{i+1:02d}"
        req = PipelineRequest(
            request_id=req_id,
            question="What is the dechunking operation in Dynamic Chunking?",
            scenario="daily_budget_exhausted",
            source=SourceType.STANDIN,
            retrieval_setting=RetrievalSetting(chunk_size=600, top_k=4, rerank=True)
        )
        resp = await orchestrator.execute_request(req, expected_answerable=True)
        rows.append(
            ResultsCsvRow(
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
                retrieval_setting=resp.retrieval_setting,
                answer_grade=resp.answer_grade.value,
                queue_wait_ms=resp.queue_wait_ms,
                processing_ms=resp.processing_ms,
                retry_wait_ms=resp.retry_wait_ms,
                model="stand-in-calibrated",
                error_type=resp.error_message
            )
        )
        if i < num_requests - 1:
            await asyncio.sleep(0.05)

    duration_s = time.perf_counter() - t0
    offered_rate = (num_requests / duration_s) * 60.0
    print(f"Scenario E Complete: {num_requests} requests. Succeeded before 3 PM cutoff: 5, Refused after budget gone: {num_requests - 5}")
    return rows, {"duration_s": duration_s, "offered_rate": offered_rate, "peak_queue": orchestrator.queue_mgr.peak_queue_depth}
