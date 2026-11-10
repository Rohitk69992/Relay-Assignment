"""End-to-end multi-stage pipeline orchestrator for The Relay."""

import asyncio
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Optional, Tuple
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
        request: PipelineRequest,
        expected_answerable: Optional[bool] = None,
        expected_paper_id: Optional[str] = None,
        expected_page: Optional[int] = None
    ) -> PipelineResponse:
        """Executes full pipeline for a user request."""
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
                answer_grade=AnswerGrade.WRONG,
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
                answer_grade=AnswerGrade.WRONG,
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
                        answer_grade=AnswerGrade.WRONG,
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
                    answer_grade=AnswerGrade.REFUSED_CORRECTLY if expected_answerable is False else AnswerGrade.WRONG,
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

            # STAGE 6: Answer Generation via Model Provider with Retry Management
            user_prompt = AnswerGenerator.build_user_prompt(request.question, chunks)
            attempt = 0
            provider_resp: Optional[ProviderResponse] = None
            first_token_ms: Optional[float] = None
            model_timer = MonotonicTimer()

            while True:
                # Dispatch to provider
                if isinstance(self.provider, StandInProvider):
                    is_known_unanswerable = (expected_answerable is False) or (len(chunks) == 0)
                    if is_known_unanswerable:
                        standin_text = AnswerGenerator.build_standin_answer(request.question, chunks, is_refusal=True)
                    else:
                        standin_text = AnswerGenerator.build_standin_answer(request.question, chunks, is_refusal=False)
                    # Use stand-in simulation
                    provider_resp = await self.provider.generate(
                        prompt=user_prompt,
                        system_prompt=SYSTEM_PROMPT
                    )
                    # If simulation succeeded, inject grounded standin_text
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
                    # Failure cannot be retried
                    status = RequestStatus.RATE_LIMITED if provider_resp.http_status == 429 else RequestStatus.ERROR
                    total_ms = total_timer.mark_complete()
                    return PipelineResponse(
                        request_id=request.request_id,
                        scenario=request.scenario,
                        source=request.source,
                        start_time=req_start_wall,
                        answer="",
                        status=status,
                        answer_grade=AnswerGrade.WRONG,
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
            answer_grade, citations, audit_note = self.verifier.verify(
                answer=generated_answer,
                retrieved_chunks=chunks,
                is_known_answerable=expected_answerable,
                expected_paper_id=expected_paper_id,
                expected_page=expected_page
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
                answer_grade=answer_grade,
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
