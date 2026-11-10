"""FastAPI application exposing The Relay pipeline endpoints, research assistant, and evaluation dashboard."""

import json
from pathlib import Path
from typing import AsyncIterator, List, Optional
import uuid
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.relay.config import settings
from src.relay.logging_setup import get_logger
from src.relay.pipeline.orchestrator import PipelineOrchestrator
from src.relay.schemas import (
    PipelineRequest,
    PipelineResponse,
    RequestStatus,
    RetrievalSetting,
    SourceType
)
from src.relay.server.dashboard_service import (
    get_experiment_summary,
    get_filtered_records,
    get_papers_metadata
)

logger = get_logger("relay.server")

app = FastAPI(
    title="The Relay Assistant API",
    description="Resilient, grounded research paper assistant with admission control, paper-first UI, and evaluation dashboard",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = PipelineOrchestrator()


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionMessage]
    stream: bool = False
    scenario: str = "interactive"
    chunk_size: int = 600
    top_k: int = 4
    rerank: bool = True


class FrontendQueryRequest(BaseModel):
    question: str
    paper_scope: str = "all"  # Specific arXiv ID or "all"
    chunk_size: int = 600
    top_k: int = 4
    rerank: bool = True
    stream: bool = False


# ---------------------------------------------------------
# FRONTEND & DASHBOARD ENDPOINTS
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the paper-first research assistant and evaluation dashboard SPA."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>The Relay Frontend (Template Missing)</h1>", status_code=404)


@app.get("/api/papers")
async def api_get_papers():
    """Returns authoritative metadata for the 3 target papers."""
    return get_papers_metadata()


@app.post("/api/query", response_model=PipelineResponse)
async def api_execute_query(req: FrontendQueryRequest):
    """Executes the pipeline with strict paper scope enforcement."""
    target_papers = [req.paper_scope] if (req.paper_scope and req.paper_scope != "all") else None
    pipeline_req = PipelineRequest(
        request_id=f"fe-query-{uuid.uuid4().hex[:8]}",
        question=req.question,
        scenario="interactive_frontend",
        source=SourceType.REAL if settings.PROVIDER_MODE == "groq" else SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(
            chunk_size=req.chunk_size,
            top_k=req.top_k,
            rerank=req.rerank
        ),
        target_papers=target_papers,
        stream=False
    )
    return await orchestrator.execute_request(pipeline_req)


@app.post("/api/query/stream")
async def api_execute_stream_query(req: FrontendQueryRequest):
    """Executes the pipeline with paper scope enforcement, streaming tokens via SSE."""
    target_papers = [req.paper_scope] if (req.paper_scope and req.paper_scope != "all") else None
    pipeline_req = PipelineRequest(
        request_id=f"fe-stream-{uuid.uuid4().hex[:8]}",
        question=req.question,
        scenario="interactive_frontend",
        source=SourceType.REAL if settings.PROVIDER_MODE == "groq" else SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(
            chunk_size=req.chunk_size,
            top_k=req.top_k,
            rerank=req.rerank
        ),
        target_papers=target_papers,
        stream=True
    )

    async def event_generator() -> AsyncIterator[str]:
        async for chunk in orchestrator.execute_stream_request(pipeline_req):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/experiments/summary")
async def api_experiment_summary():
    """Returns authoritative summary tables (Table 1, 2a, 2b, 3, slow breakdown, calibration)."""
    return get_experiment_summary()


@app.get("/api/experiments/records")
async def api_experiment_records(
    source: Optional[str] = None,
    scenario: Optional[str] = None,
    retrieval_setting: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0
):
    """Returns filtered individual rows from results.csv."""
    return get_filtered_records(
        source=source,
        scenario=scenario,
        retrieval_setting=retrieval_setting,
        status=status,
        limit=limit,
        offset=offset
    )


# ---------------------------------------------------------
# SYSTEM METRICS & PROTOCOL ENDPOINTS
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "provider_mode": settings.PROVIDER_MODE,
        "is_groq_available": settings.is_groq_available(),
        "active_queue_depth": orchestrator.queue_mgr.current_queue_depth,
        "peak_queue_depth": orchestrator.queue_mgr.peak_queue_depth
    }


@app.get("/metrics")
async def get_metrics():
    return {
        "queue": {
            "current_depth": orchestrator.queue_mgr.current_queue_depth,
            "peak_depth": orchestrator.queue_mgr.peak_queue_depth,
            "total_admitted": orchestrator.queue_mgr.total_admitted,
            "total_rejected_queue_full": orchestrator.queue_mgr.total_rejected_queue_full,
            "total_dropped_queue_timeout": orchestrator.queue_mgr.total_dropped_queue_timeout
        },
        "quotas": {
            "daily_requests_used": orchestrator.rate_limiter.daily_requests_used,
            "daily_tokens_used": orchestrator.rate_limiter.daily_tokens_used,
            "daily_request_limit": orchestrator.rate_limiter.rpd,
            "daily_token_limit": orchestrator.rate_limiter.tpd
        }
    }


@app.post("/v1/query", response_model=PipelineResponse)
async def query_pipeline(request: PipelineRequest):
    """Executes the full pipeline for a structured question."""
    response = await orchestrator.execute_request(request)
    if response.status == RequestStatus.RATE_LIMITED:
        raise HTTPException(status_code=429, detail=response.error_message or "Rate limit exceeded")
    elif response.status == RequestStatus.TIMEOUT:
        raise HTTPException(status_code=504, detail=response.error_message or "Request timed out")
    elif response.status == RequestStatus.ERROR:
        raise HTTPException(status_code=500, detail=response.error_message or "Internal pipeline error")
    return response


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    user_messages = [m.content for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    user_query = user_messages[-1]

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    pipeline_req = PipelineRequest(
        request_id=req_id,
        question=user_query,
        scenario=req.scenario,
        source=SourceType.REAL if settings.PROVIDER_MODE == "groq" else SourceType.STANDIN,
        retrieval_setting=RetrievalSetting(
            chunk_size=req.chunk_size,
            top_k=req.top_k,
            rerank=req.rerank
        ),
        stream=req.stream
    )

    if not req.stream:
        resp = await orchestrator.execute_request(pipeline_req)
        return {
            "id": req_id,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp.answer
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": resp.tokens_in,
                "completion_tokens": resp.tokens_out,
                "total_tokens": resp.tokens_in + resp.tokens_out
            },
            "relay_telemetry": {
                "first_token_ms": resp.first_token_ms,
                "total_ms": resp.total_ms,
                "queue_wait_ms": resp.queue_wait_ms,
                "retries": resp.retries,
                "status": resp.status,
                "answer_grade": resp.answer_grade
            }
        }
    else:
        async def event_generator() -> AsyncIterator[str]:
            async for chunk in orchestrator.execute_stream_request(pipeline_req):
                if chunk["event"] == "token":
                    chunk_data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk["content"]},
                            "finish_reason": None
                        }],
                        "relay_telemetry": {
                            "first_token_ms": chunk.get("first_token_ms")
                        }
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                elif chunk["event"] == "done":
                    meta_data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "relay_telemetry": {
                            "first_token_ms": chunk.get("first_token_ms"),
                            "total_ms": chunk.get("total_ms"),
                            "citations": chunk.get("citations", []),
                            "verified": chunk.get("verified", True)
                        }
                    }
                    yield f"data: {json.dumps(meta_data)}\n\n"
                elif chunk["event"] == "error":
                    err_data = {
                        "error": chunk.get("error"),
                        "status": chunk.get("status")
                    }
                    yield f"data: {json.dumps(err_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
