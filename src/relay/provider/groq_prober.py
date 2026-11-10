"""Real Groq service probe runner capturing genuine network telemetry and rate-limit headers."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel
from src.relay.config import settings
from src.relay.logging_setup import MonotonicTimer, get_logger

logger = get_logger("relay.prober")


class RealGroqProbeRecord(BaseModel):
    probe_id: str
    timestamp_utc: str
    target_url: str
    model: str
    http_status: int
    ttft_ms: Optional[float] = None
    total_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    response_headers: Dict[str, str] = {}
    authenticated: bool
    response_snippet: str
    error_detail: Optional[str] = None


async def run_single_probe(
    probe_id: str,
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "openai/gpt-oss-20b",
    base_url: str = "https://api.groq.com/openai/v1",
    stream: bool = True
) -> RealGroqProbeRecord:
    """Executes a genuine probe against Groq API endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    has_key = bool(api_key and api_key.strip())
    
    headers = {
        "Content-Type": "application/json"
    }
    if has_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise scientific assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.0,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None
    }
    if not stream:
        payload.pop("stream_options", None)

    timestamp = datetime.now(timezone.utc).isoformat()
    timer = MonotonicTimer()
    ttft_ms: Optional[float] = None
    accumulated_content = []
    response_headers = {}
    tokens_in = 0
    tokens_out = 0
    status_code = 0
    error_detail = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if stream and has_key:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    status_code = resp.status_code
                    response_headers = {k.lower(): v for k, v in resp.headers.items()}
                    if status_code == 200:
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if "usage" in chunk and chunk["usage"]:
                                        u = chunk["usage"]
                                        tokens_in = u.get("prompt_tokens", tokens_in)
                                        tokens_out = u.get("completion_tokens", tokens_out)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    tok = delta.get("content") or delta.get("reasoning") or ""
                                    if tok:
                                        if ttft_ms is None:
                                            ttft_ms = timer.mark_first_token()
                                        accumulated_content.append(tok)
                                except Exception:
                                    continue
                    else:
                        body = await resp.aread()
                        error_detail = body.decode("utf-8", errors="ignore")
            else:
                resp = await client.post(url, headers=headers, json=payload)
                status_code = resp.status_code
                response_headers = {k.lower(): v for k, v in resp.headers.items()}
                if status_code == 200:
                    data = resp.json()
                    accumulated_content.append(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
                else:
                    error_detail = resp.text
        except Exception as exc:
            error_detail = f"Network exception: {exc}"
            status_code = 0

    total_ms = timer.mark_complete()
    full_text = "".join(accumulated_content)

    # Filter critical rate limit headers
    clean_headers = {
        k: v for k, v in response_headers.items()
        if k.startswith("x-ratelimit-") or k in {"retry-after", "date", "server", "cf-ray"}
    }

    return RealGroqProbeRecord(
        probe_id=probe_id,
        timestamp_utc=timestamp,
        target_url=url,
        model=model,
        http_status=status_code,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out or (len(full_text) // 4),
        response_headers=clean_headers,
        authenticated=has_key,
        response_snippet=full_text[:120] if full_text else (error_detail[:120] if error_detail else "No response body"),
        error_detail=error_detail
    )


async def execute_probe_suite(
    output_dir: Path,
    num_probes: int = 3
) -> List[RealGroqProbeRecord]:
    """Runs a controlled suite of real Groq probes and saves results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None
    model = settings.GROQ_MODEL
    base_url = settings.GROQ_BASE_URL

    test_queries = [
        "State the number of attention heads in the base Transformer from Attention Is All You Need.",
        "What is the framerate of the Mimi neural audio codec in Moshi?",
        "Explain what predicts boundaries between chunks in Dynamic Chunking."
    ]

    records: List[RealGroqProbeRecord] = []
    print(f"Executing {min(num_probes, len(test_queries))} real Groq probes against {base_url} (Model: {model})...")
    print(f"API Key configured: {'YES (Masked)' if api_key else 'NO (Unauthenticated probe will document provider endpoint handshake)'}")

    for idx, q in enumerate(test_queries[:num_probes], start=1):
        probe_id = f"real_groq_probe_{idx:02d}"
        rec = await run_single_probe(
            probe_id=probe_id,
            prompt=q,
            api_key=api_key,
            model=model,
            base_url=base_url,
            stream=True
        )
        records.append(rec)
        print(f"  [{probe_id}] HTTP {rec.http_status} | Total: {rec.total_ms:.1f}ms | Auth: {rec.authenticated} | Snippet: {rec.response_snippet[:60]}...")
        if idx < num_probes:
            await asyncio.sleep(1.0)

    # Save to JSON
    json_path = output_dir / "real_groq_probes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2)

    # Save to CSV
    csv_path = output_dir / "real_groq_probes.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["probe_id", "timestamp_utc", "target_url", "model", "http_status", "ttft_ms", "total_ms", "tokens_in", "tokens_out", "authenticated", "response_snippet"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "probe_id": r.probe_id,
                "timestamp_utc": r.timestamp_utc,
                "target_url": r.target_url,
                "model": r.model,
                "http_status": r.http_status,
                "ttft_ms": round(r.ttft_ms, 1) if r.ttft_ms is not None else "",
                "total_ms": round(r.total_ms, 1),
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "authenticated": r.authenticated,
                "response_snippet": r.response_snippet
            })

    print(f"Saved real probe logs to {json_path} and {csv_path}")
    return records
