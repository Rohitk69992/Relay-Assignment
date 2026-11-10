"""Real Groq API client adapter implementing ModelProviderBase."""

import json
import time
from typing import AsyncIterator, Dict, Optional
import httpx
from src.relay.config import settings
from src.relay.logging_setup import MonotonicTimer, get_logger
from src.relay.provider.base import ModelProviderBase, ProviderResponse

logger = get_logger("relay.groq")


class GroqProvider(ModelProviderBase):
    """Integrates with Groq's live Cloud API endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0
    ):
        raw_key = api_key or (settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None)
        self.api_key = raw_key
        self.base_url = (base_url or settings.GROQ_BASE_URL).rstrip("/")
        self.model = model or settings.GROQ_MODEL
        self.timeout = timeout_seconds

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured. Set GROQ_API_KEY in environment or .env file.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ProviderResponse:
        timer = MonotonicTimer()
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=self._get_headers(), json=payload)
            total_ms = timer.mark_complete()

            resp_headers = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"}

            if response.status_code != 200:
                error_body = response.text
                error_type = "unknown_error"
                try:
                    err_json = response.json()
                    error_type = err_json.get("error", {}).get("type", f"http_{response.status_code}")
                except Exception:
                    pass

                return ProviderResponse(
                    content="",
                    tokens_in=0,
                    tokens_out=0,
                    first_token_ms=None,
                    total_ms=total_ms,
                    model=self.model,
                    headers=resp_headers,
                    http_status=response.status_code,
                    error_type=error_type,
                    is_simulation=False
                )

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

            return ProviderResponse(
                content=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=None,  # In non-streaming execution, TTFT is not measured
                total_ms=total_ms,
                model=self.model,
                headers=resp_headers,
                http_status=response.status_code,
                error_type=None,
                is_simulation=False
            )

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> AsyncIterator[str]:
        """Streams tokens from Groq via Server-Sent Events (SSE)."""
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise httpx.HTTPStatusError(
                        f"Groq API error {response.status_code}: {error_body.decode('utf-8', errors='ignore')}",
                        request=response.request,
                        response=response
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue
