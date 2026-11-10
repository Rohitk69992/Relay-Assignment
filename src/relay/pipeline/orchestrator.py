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
        if provider:
            self.provider = provider
        elif settings.PROVIDER_MODE == ProviderMode.GROQ:
            self.provider = GroqProvider()
        else:
            self.provider = StandInProvider()

        self.queue_mgr = queue_mgr or AdmissionQueueManager()
        self.rate_limiter = rate_limiter or ClientRateLimiter()
        self.retry_mgr = retry_mgr or RetryManager()

        self.intent_planner = IntentPlanner()
        self.sufficiency_evaluator = SufficiencyEvaluator()
        self.verifier = ResponseVerifier()

    def _get_provider_names(self) -> Tuple[str, str]:
        """Returns (provider_name, model_name)."""
        if isinstance(self.provider, GroqProvider):
            return "groq", self.provider.model
        return "standin", getattr(self.provider, "model", "stand-in-calibrated")

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
        provider_name, model_name = self._get_provider_names()

        # Check configuration: No silent fallback
        if isinstance(self.provider, GroqProvider) and not self.provider.api_key:
            total_ms = total_timer.mark_complete()
            return PipelineResponse(
                request_id=request.request_id,
                scenario=request.scenario,
                source=SourceType.REAL,
                start_time=req_start_wall,
                answer=(
                    "Configuration Error: PROVIDER_MODE is configured as 'groq', but GROQ_API_KEY is missing or empty. "
                    "Please configure GROQ_API_KEY in your local .env file and restart the server, "
                    "or set PROVIDER_MODE=standin for offline extractive simulation."
                ),
                status=RequestStatus.ERROR,
                first_token_ms=None,
                total_ms=total_ms,
                queue_wait_ms=0.0,
                processing_ms=0.0,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=request.retrieval_setting.label,
                model=self.provider.model,
                error_message="GROQ_API_KEY is not configured in .env"
            )

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
                model=model_name,
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
                model=model_name,
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
                        model=model_name,
                        error_message=reason
                    )
                else:
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
                    first_token_ms=None,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    processing_ms=total_ms - queue_wait_ms,
                    retries=0,
                    tokens_in=15,
                    tokens_out=25,
                    retrieval_setting=request.retrieval_setting.label,
                    model=model_name
                )

            # Check Scope Mismatch: query explicitly targets a different paper than selected scope
            if request.target_papers and "all" not in request.target_papers:
                if query_plan.target_papers and not any(p in request.target_papers for p in query_plan.target_papers):
                    paper_titles = {
                        "1706.03762v7": "Attention Is All You Need",
                        "2410.00037v1": "Moshi",
                        "2507.07955v2": "Dynamic Chunking"
                    }
                    detected = [paper_titles.get(p, p) for p in query_plan.target_papers]
                    locked = [paper_titles.get(p, p) for p in request.target_papers]
                    refusal_msg = (
                        f"REFUSAL: The question targets topics from {', '.join(detected)}, "
                        f"but the retrieval scope is strictly locked to {', '.join(locked)}. "
                        f"The selected paper does not establish the answer."
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
                        first_token_ms=None,
                        total_ms=total_ms,
                        queue_wait_ms=queue_wait_ms,
                        processing_ms=total_ms - queue_wait_ms,
                        retries=0,
                        tokens_in=max(10, len(request.question) // 4),
                        tokens_out=25,
                        retrieval_setting=request.retrieval_setting.label,
                        target_papers=request.target_papers,
                        model=model_name
                    )

            # STAGE 3: Research-Paper Retrieval Tool Invocation
            retrieval_result = self.retriever.retrieve(
                query=query_plan.original_query,
                setting=request.retrieval_setting,
                target_papers=request.target_papers
            )
            chunks = retrieval_result.chunks

            # STAGE 4 & 5: Evidence Sufficiency Evaluation & Bounded Refinement Branch
            is_sufficient, refined_query, reason = self.sufficiency_evaluator.evaluate(
                query_plan=query_plan,
                chunks=chunks,
                refinement_count=0
            )

            if not is_sufficient and refined_query != query_plan.original_query:
                refined_result = self.retriever.retrieve(
                    query=refined_query,
                    setting=request.retrieval_setting,
                    target_papers=request.target_papers
                )
                chunks = refined_result.chunks
                retrieval_result.refinement_performed = True
                is_sufficient, _, _ = self.sufficiency_evaluator.evaluate(
                    query_plan=query_plan,
                    chunks=chunks,
                    refinement_count=1
                )

            # Check if evidence is still insufficient
            if not is_sufficient or not chunks:
                scope_desc = (
                    f"in the selected paper scope ({', '.join(request.target_papers)})"
                    if (request.target_papers and "all" not in request.target_papers)
                    else "in the supplied research library"
                )
                refusal_msg = (
                    f"REFUSAL: The evidence {scope_desc} does not establish the answer to this question. "
                    f"{reason or 'The paper does not contain sufficient evidence on this topic.'}"
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
                    first_token_ms=None,
                    total_ms=total_ms,
                    queue_wait_ms=queue_wait_ms,
                    processing_ms=total_ms - queue_wait_ms,
                    retries=0,
                    tokens_in=max(10, len(request.question) // 4),
                    tokens_out=25,
                    retrieval_setting=request.retrieval_setting.label,
                    target_papers=request.target_papers,
                    model=model_name
                )

            # STAGE 6: Answer Generation via Model Provider with Retry Management
            user_prompt = AnswerGenerator.build_user_prompt(request.question, chunks)
            attempt = 0
            provider_resp: Optional[ProviderResponse] = None
            first_token_ms: Optional[float] = None
            model_timer = MonotonicTimer()

            while True:
                if isinstance(self.provider, StandInProvider):
                    standin_text = AnswerGenerator.build_standin_answer(request.question, chunks, is_refusal=False)
                    provider_resp = await self.provider.generate(
                        prompt=user_prompt,
                        system_prompt=SYSTEM_PROMPT,
                        precomputed_text=standin_text
                    )
                else:
                    provider_resp = await self.provider.generate(
                        prompt=user_prompt,
                        system_prompt=SYSTEM_PROMPT
                    )

                # Evaluate provider outcome
                if provider_resp.http_status == 200:
                    first_token_ms = provider_resp.first_token_ms
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
                        answer=provider_resp.content or f"Provider call failed with HTTP {provider_resp.http_status}",
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
                        model=model_name,
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
                target_papers=request.target_papers,
                model=provider_resp.model or model_name,
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
        provider_name, model_name = self._get_provider_names()

        # Emit initial start event with request_id and provider metadata
        yield {
            "event": "start",
            "request_id": request.request_id,
            "model": model_name,
            "provider": provider_name,
            "source": request.source.value
        }

        # Check configuration: No silent fallback
        if isinstance(self.provider, GroqProvider) and not self.provider.api_key:
            yield {
                "event": "error",
                "request_id": request.request_id,
                "status": 503,
                "error": (
                    "Configuration Error: PROVIDER_MODE is configured as 'groq', but GROQ_API_KEY is missing or empty. "
                    "Please configure GROQ_API_KEY in your local .env file and restart the server, "
                    "or set PROVIDER_MODE=standin for offline extractive simulation."
                ),
                "total_ms": total_timer.mark_complete()
            }
            return

        try:
            enqueue_time = await self.queue_mgr.enter_queue(request.request_id)
            queue_wait_ms = await self.queue_mgr.acquire_execution_slot(request.request_id, enqueue_time)
        except QueueFullError as e:
            yield {
                "event": "error",
                "request_id": request.request_id,
                "status": 429,
                "error": str(e),
                "total_ms": total_timer.mark_complete()
            }
            return
        except QueueTimeoutError as e:
            yield {
                "event": "error",
                "request_id": request.request_id,
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
                yield {
                    "event": "done",
                    "request_id": request.request_id,
                    "total_ms": total_timer.mark_complete(),
                    "queue_wait_ms": queue_wait_ms,
                    "citations": [],
                    "citation_validation": "refusal",
                    "verified": True,
                    "answer_quality": "not_evaluated",
                    "model": model_name,
                    "provider": provider_name,
                    "source": request.source.value,
                    "tokens_in": 15,
                    "tokens_out": 25,
                    "retries": 0,
                    "status": "refusal"
                }
                return

            # Scope Mismatch Check
            if request.target_papers and "all" not in request.target_papers:
                if query_plan.target_papers and not any(p in request.target_papers for p in query_plan.target_papers):
                    paper_titles = {
                        "1706.03762v7": "Attention Is All You Need",
                        "2410.00037v1": "Moshi",
                        "2507.07955v2": "Dynamic Chunking"
                    }
                    detected = [paper_titles.get(p, p) for p in query_plan.target_papers]
                    locked = [paper_titles.get(p, p) for p in request.target_papers]
                    refusal_msg = (
                        f"REFUSAL: The question targets topics from {', '.join(detected)}, "
                        f"but the retrieval scope is strictly locked to {', '.join(locked)}. "
                        f"The selected paper does not establish the answer."
                    )
                    yield {"event": "token", "content": refusal_msg, "first_token_ms": queue_wait_ms + 10.0}
                    yield {
                        "event": "done",
                        "request_id": request.request_id,
                        "total_ms": total_timer.mark_complete(),
                        "queue_wait_ms": queue_wait_ms,
                        "citations": [],
                        "citation_validation": "refusal",
                        "verified": True,
                        "answer_quality": "not_evaluated",
                        "model": model_name,
                        "provider": provider_name,
                        "source": request.source.value,
                        "tokens_in": max(10, len(request.question) // 4),
                        "tokens_out": 25,
                        "retries": 0,
                        "status": "refusal"
                    }
                    return

            # Retrieval tool
            retrieval_result = self.retriever.retrieve(
                query=query_plan.original_query,
                setting=request.retrieval_setting,
                target_papers=request.target_papers
            )
            chunks = retrieval_result.chunks

            # Sufficiency check
            is_sufficient, refined_query, reason = self.sufficiency_evaluator.evaluate(query_plan, chunks, refinement_count=0)
            if not is_sufficient and refined_query != query_plan.original_query:
                refined_result = self.retriever.retrieve(
                    query=refined_query,
                    setting=request.retrieval_setting,
                    target_papers=request.target_papers
                )
                chunks = refined_result.chunks
                is_sufficient, _, _ = self.sufficiency_evaluator.evaluate(query_plan, chunks, refinement_count=1)

            if not is_sufficient or not chunks:
                scope_desc = (
                    f"in the selected paper scope ({', '.join(request.target_papers)})"
                    if (request.target_papers and "all" not in request.target_papers)
                    else "in the supplied research library"
                )
                refusal_msg = (
                    f"REFUSAL: The evidence {scope_desc} does not establish the answer to this question. "
                    f"{reason or 'The paper does not contain sufficient evidence on this topic.'}"
                )
                yield {"event": "token", "content": refusal_msg, "first_token_ms": queue_wait_ms + 15.0}
                yield {
                    "event": "done",
                    "request_id": request.request_id,
                    "total_ms": total_timer.mark_complete(),
                    "queue_wait_ms": queue_wait_ms,
                    "citations": [],
                    "citation_validation": "refusal",
                    "verified": True,
                    "answer_quality": "not_evaluated",
                    "model": model_name,
                    "provider": provider_name,
                    "source": request.source.value,
                    "tokens_in": max(10, len(request.question) // 4),
                    "tokens_out": 25,
                    "retries": 0,
                    "status": "refusal"
                }
                return

            # Grounded generation streaming
            user_prompt = AnswerGenerator.build_user_prompt(request.question, chunks)
            first_token_ms = None
            accumulated_tokens = []

            # Stream from provider following identical answer-generation contract
            if isinstance(self.provider, StandInProvider):
                standin_text = AnswerGenerator.build_standin_answer(request.question, chunks, is_refusal=False)
                async for token_chunk in self.provider.stream_text(standin_text):
                    if first_token_ms is None:
                        first_token_ms = queue_wait_ms + total_timer.elapsed_ms
                    accumulated_tokens.append(token_chunk)
                    yield {
                        "event": "token",
                        "content": token_chunk,
                        "first_token_ms": first_token_ms
                    }
                full_answer = "".join(accumulated_tokens)
            else:
                try:
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
                except Exception as ex:
                    yield {
                        "event": "error",
                        "request_id": request.request_id,
                        "status": 500,
                        "error": f"Groq streaming failure: {str(ex)}",
                        "total_ms": total_timer.mark_complete()
                    }
                    return

            # Response verification & citations
            is_grounded, citations, is_refusal, audit_note = self.verifier.verify_grounding(full_answer, chunks)

            yield {
                "event": "done",
                "request_id": request.request_id,
                "first_token_ms": first_token_ms,
                "total_ms": total_timer.mark_complete(),
                "queue_wait_ms": queue_wait_ms,
                "citations": [c.model_dump() for c in citations],
                "citation_validation": "passed" if is_grounded else ("refusal" if is_refusal else "failed"),
                "verified": is_grounded,
                "answer_quality": "not_evaluated",
                "model": model_name,
                "provider": provider_name,
                "source": request.source.value,
                "tokens_in": max(10, len(user_prompt) // 4),
                "tokens_out": max(10, len(full_answer) // 4),
                "retries": 0,
                "status": "ok" if (is_grounded or is_refusal) else "unverified"
            }

        finally:
            self.queue_mgr.release_execution_slot()
