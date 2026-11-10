"""End-to-end multi-stage pipeline orchestrator for The Relay."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from src.relay.config import ProviderMode, settings
from src.relay.logging_setup import MonotonicTimer, get_logger
from src.relay.pipeline.generator import AnswerGenerator, SYSTEM_PROMPT
from src.relay.pipeline.intent import IntentPlanner
from src.relay.pipeline.sufficiency import SufficiencyEvaluator
from src.relay.pipeline.verifier import ResponseVerifier
from src.relay.provider.base import ModelProviderBase, ProviderResponse
from src.relay.provider.groq_client import GroqProvider
from src.relay.provider.standin_client import StandInProvider
from src.relay.resilience.queue_manager import AdmissionQueueManager, QueueFullError, QueueTimeoutError
from src.relay.resilience.rate_limiter import ClientRateLimiter
from src.relay.resilience.retry_policy import RetryManager
from src.relay.retrieval.retriever import ResearchRetriever
from src.relay.schemas import (
    AnswerGrade,
    PipelineRequest,
    PipelineResponse,
    RequestStatus,
    RetrievalSetting,
    SourceType
)

logger = get_logger("relay.pipeline")


class PipelineOrchestrator:
    """Coordinates admission, query planning, retrieval, sufficiency, provider execution, and verification."""

    def __init__(
        self,
        retriever: Optional[ResearchRetriever] = None,
        provider: Optional[ModelProviderBase] = None,
        queue_mgr: Optional[AdmissionQueueManager] = None,
        rate_limiter: Optional[ClientRateLimiter] = None,
        retry_mgr: Optional[RetryManager] = None
    ):
        self.retriever = retriever or ResearchRetriever()
        self.provider = provider or (
            GroqProvider() if settings.PROVIDER_MODE == ProviderMode.GROQ and settings.is_groq_available()
            else StandInProvider()
        )
        self.queue_mgr = queue_mgr or AdmissionQueueManager()
        self.rate_limiter = rate_limiter or ClientRateLimiter()
        self.retry_mgr = retry_mgr or RetryManager()

        self.intent_planner = IntentPlanner()
        self.sufficiency_evaluator = SufficiencyEvaluator()
        self.verifier = ResponseVerifier()

    async def execute_request(
        self,
        request: PipelineRequest
    ) -> PipelineResponse:
        """Executes full pipeline for a user request without receiving ground-truth evaluation labels."""
        req_start_wall = datetime.now(timezone.utc).isoformat()
        total_timer = MonotonicTimer()
        queue_wait_ms = 0.0
        retry_wait_ms = 0.0
        retries = 0

        # STAGE 1: Admission Control & Queue Buffering
        try:
            enqueue_time = await self.queue_mgr.enter_queue(request.request_id)
            queue_wait_ms = await self.queue_mgr.acquire_execution_slot(request.request_id, enqueue_time)
        except QueueFullError as e:
            total_ms = total_timer.mark_complete()
            return PipelineResponse(
                request_id=request.request_id,
                scenario=request.scenario,
                source=request.source,
                start_time=req_start_wall,
                answer="",
                status=RequestStatus.RATE_LIMITED,
                answer_grade=None,
                first_token_ms=None,
                total_ms=total_ms,
                queue_wait_ms=0.0,
                processing_ms=0.0,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=request.retrieval_setting.label,
                error_message=str(e)
            )
        except QueueTimeoutError as e:
            total_ms = total_timer.mark_complete()
            return PipelineResponse(
                request_id=request.request_id,
                scenario=request.scenario,
                source=request.source,
                start_time=req_start_wall,
                answer="",
                status=RequestStatus.TIMEOUT,
                answer_grade=None,
                first_token_ms=None,
                total_ms=total_ms,
                queue_wait_ms=queue_wait_ms,
                processing_ms=0.0,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=request.retrieval_setting.label,
                error_message=str(e)
            )

        try:
            # Check client-side rate limits
            permit_granted, wait_sec, reason = await self.rate_limiter.acquire_permit(estimated_tokens=200)
            if not permit_granted:
                if wait_sec is None:  # Daily quota exhausted
                    total_ms = total_timer.mark_complete()
                    return PipelineResponse(
                        request_id=request.request_id,
                        scenario=request.scenario,
                        source=request.source,
                        start_time=req_start_wall,
                        answer="System notice: Daily API request/token budget is completely exhausted. Resets at 00:00 UTC.",
                        status=RequestStatus.RATE_LIMITED,
                        answer_grade=None,
                        first_token_ms=None,
                        total_ms=total_ms,
                        queue_wait_ms=queue_wait_ms,
                        processing_ms=0.0,
                        retries=0,
                        tokens_in=0,
                        tokens_out=0,
                        retrieval_setting=request.retrieval_setting.label,
                        error_message=reason
                    )
                else:
                    # Wait if backoff is small and within deadline
                    if wait_sec < 2.0:
                        await asyncio.sleep(wait_sec)
                        retry_wait_ms += wait_sec * 1000.0

            # STAGE 2: Intent Understanding & Query Planning
            query_plan = self.intent_planner.plan_query(request.question)

            # If question is clearly off-topic, short circuit to explicit refusal
            if query_plan.is_off_topic:
                refusal_msg = (
                    "REFUSAL: The supplied research library does not establish the answer to this question. "
                    "The topic is outside the scope of the ingested research papers."
                )
                total_ms = total_timer.mark_complete()
                return PipelineResponse(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=req_start_wall,
                    answer=refusal_msg,
                    status=RequestStatus.OK,
                    answer_grade=None,
                    first_token_ms=queue_wait_ms + 10.0,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    processing_ms=total_ms - queue_wait_ms,
                    retries=0,
                    tokens_in=15,
                    tokens_out=25,
                    retrieval_setting=request.retrieval_setting.label
                )

            # STAGE 3: Research-Paper Retrieval Tool Invocation
            retrieval_result = self.retriever.retrieve(
                query=query_plan.original_query,
                setting=request.retrieval_setting
            )
            chunks = retrieval_result.chunks

            # STAGE 4 & 5: Evidence Sufficiency Evaluation & Bounded Refinement Branch
            is_sufficient, refined_query, reason = self.sufficiency_evaluator.evaluate(
                query_plan=query_plan,
                chunks=chunks,
                refinement_count=0
            )

            if not is_sufficient and refined_query != query_plan.original_query:
                # Perform bounded refinement loop (at most 1 iteration)
                refined_result = self.retriever.retrieve(
                    query=refined_query,
                    setting=request.retrieval_setting
                )
                chunks = refined_result.chunks
                retrieval_result.refinement_performed = True
                is_sufficient, _, _ = self.sufficiency_evaluator.evaluate(
                    query_plan=query_plan,
                    chunks=chunks,
                    refinement_count=1
                )

            # Check if evidence is still insufficient (e.g. key terms completely absent)
            if not is_sufficient or not chunks:
                refusal_msg = (
                    "REFUSAL: The supplied research library does not establish the answer to this question. "
                    "The ingested papers do not contain sufficient evidence on this topic."
                )
                total_ms = total_timer.mark_complete()
                return PipelineResponse(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=req_start_wall,
                    answer=refusal_msg,
                    status=RequestStatus.OK,
                    answer_grade=None,
                    first_token_ms=queue_wait_ms + 15.0,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    processing_ms=total_ms - queue_wait_ms,
                    retries=0,
                    tokens_in=max(10, len(request.question) // 4),
                    tokens_out=25,
                    retrieval_setting=request.retrieval_setting.label
                )

            # STAGE 6: Answer Generation via Model Provider with Retry Management
            user_prompt = AnswerGenerator.build_user_prompt(request.question, chunks)
            attempt = 0
            provider_resp: Optional[ProviderResponse] = None
            first_token_ms: Optional[float] = None
            model_timer = MonotonicTimer()

            while True:
                # Dispatch to provider
                if isinstance(self.provider, StandInProvider):
                    standin_text = AnswerGenerator.build_standin_answer(request.question, chunks, is_refusal=False)
                    provider_resp = await self.provider.generate(
                        prompt=user_prompt,
                        system_prompt=SYSTEM_PROMPT
                    )
                    if provider_resp.http_status == 200:
                        provider_resp.content = standin_text
                else:
                    provider_resp = await self.provider.generate(
                        prompt=user_prompt,
                        system_prompt=SYSTEM_PROMPT
                    )

                # Evaluate provider outcome
                if provider_resp.http_status == 200:
                    first_token_ms = queue_wait_ms + (provider_resp.first_token_ms or model_timer.elapsed_ms)
                    break

                # Handle failure / rate-limit with retry policy
                can_retry, delay_s, retry_reason = self.retry_mgr.should_retry(
                    attempt=attempt,
                    http_status=provider_resp.http_status,
                    error_type=provider_resp.error_type,
                    error_message=provider_resp.content,
                    start_time_monotonic=total_timer._start_time,
                    headers=provider_resp.headers
                )

                if can_retry:
                    retries += 1
                    attempt += 1
                    retry_wait_ms += delay_s * 1000.0
                    await asyncio.sleep(delay_s)
                    continue
                else:
                    status = RequestStatus.RATE_LIMITED if provider_resp.http_status == 429 else RequestStatus.ERROR
                    total_ms = total_timer.mark_complete()
                    return PipelineResponse(
                        request_id=request.request_id,
                        scenario=request.scenario,
                        source=request.source,
                        start_time=req_start_wall,
                        answer="",
                        status=status,
                        answer_grade=None,
                        first_token_ms=None,
                        total_ms=total_ms,
                        queue_wait_ms=queue_wait_ms,
                        processing_ms=total_ms - queue_wait_ms - retry_wait_ms,
                        retry_wait_ms=retry_wait_ms,
                        retries=retries,
                        tokens_in=provider_resp.tokens_in,
                        tokens_out=0,
                        retrieval_setting=request.retrieval_setting.label,
                        error_message=f"Provider call failed: HTTP {provider_resp.http_status} ({provider_resp.error_type})"
                    )

            # STAGE 7: Response Verification & Citation Auditing
            generated_answer = provider_resp.content
            is_grounded, citations, is_refusal, audit_note = self.verifier.verify_grounding(
                answer=generated_answer,
                retrieved_chunks=chunks
            )

            total_ms = total_timer.mark_complete()
            processing_ms = max(0.0, total_ms - queue_wait_ms - retry_wait_ms)

            return PipelineResponse(
                request_id=request.request_id,
                scenario=request.scenario,
                source=request.source,
                start_time=req_start_wall,
                answer=generated_answer,
                citations=citations,
                status=RequestStatus.OK,
                answer_grade=None,
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                queue_wait_ms=queue_wait_ms,
                processing_ms=processing_ms,
                retry_wait_ms=retry_wait_ms,
                retries=retries,
                tokens_in=provider_resp.tokens_in,
                tokens_out=provider_resp.tokens_out,
                retrieval_setting=request.retrieval_setting.label,
                error_message=None
            )

        finally:
            self.queue_mgr.release_execution_slot()

    async def execute_stream_request(
        self,
        request: PipelineRequest
    ) -> AsyncIterator[Dict[str, Any]]:
        """Executes full pipeline streaming tokens in real time via Server-Sent Events."""
        total_timer = MonotonicTimer()
        queue_wait_ms = 0.0

        try:
            enqueue_time = await self.queue_mgr.enter_queue(request.request_id)
            queue_wait_ms = await self.queue_mgr.acquire_execution_slot(request.request_id, enqueue_time)
        except QueueFullError as e:
            yield {
                "event": "error",
                "status": 429,
                "error": str(e),
                "total_ms": total_timer.mark_complete()
            }
            return
        except QueueTimeoutError as e:
            yield {
                "event": "error",
                "status": 504,
                "error": str(e),
                "total_ms": total_timer.mark_complete()
            }
            return

        try:
            # Query planning & off-topic check
            query_plan = self.intent_planner.plan_query(request.question)
            if query_plan.is_off_topic:
                refusal_msg = (
                    "REFUSAL: The supplied research library does not establish the answer to this question. "
                    "The topic is outside the scope of the ingested research papers."
                )
                yield {"event": "token", "content": refusal_msg, "first_token_ms": queue_wait_ms + 10.0}
                yield {"event": "done", "total_ms": total_timer.mark_complete(), "citations": [], "verified": True}
                return

            # Retrieval tool
            retrieval_result = self.retriever.retrieve(query_plan.original_query, request.retrieval_setting)
            chunks = retrieval_result.chunks

            # Sufficiency check
            is_sufficient, refined_query, _ = self.sufficiency_evaluator.evaluate(query_plan, chunks, refinement_count=0)
            if not is_sufficient and refined_query != query_plan.original_query:
                refined_result = self.retriever.retrieve(refined_query, request.retrieval_setting)
                chunks = refined_result.chunks
                is_sufficient, _, _ = self.sufficiency_evaluator.evaluate(query_plan, chunks, refinement_count=1)

            if not is_sufficient or not chunks:
                refusal_msg = (
                    "REFUSAL: The supplied research library does not establish the answer to this question. "
                    "The ingested papers do not contain sufficient evidence on this topic."
                )
                yield {"event": "token", "content": refusal_msg, "first_token_ms": queue_wait_ms + 15.0}
                yield {"event": "done", "total_ms": total_timer.mark_complete(), "citations": [], "verified": True}
                return

            # Grounded generation streaming
            user_prompt = AnswerGenerator.build_user_prompt(request.question, chunks)
            first_token_ms = None
            accumulated_tokens = []

            # Stream from provider
            async for token_chunk in self.provider.stream_generate(user_prompt, system_prompt=SYSTEM_PROMPT):
                if first_token_ms is None:
                    first_token_ms = queue_wait_ms + total_timer.elapsed_ms
                accumulated_tokens.append(token_chunk)
                yield {
                    "event": "token",
                    "content": token_chunk,
                    "first_token_ms": first_token_ms
                }

            full_answer = "".join(accumulated_tokens)
            is_grounded, citations, _, _ = self.verifier.verify_grounding(full_answer, chunks)

            yield {
                "event": "done",
                "first_token_ms": first_token_ms,
                "total_ms": total_timer.mark_complete(),
                "citations": [c.model_dump() for c in citations],
                "verified": is_grounded
            }

        finally:
            self.queue_mgr.release_execution_slot()
