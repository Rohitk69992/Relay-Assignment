"""Stand-In Model Service: A highly controllable simulator mirroring the Groq API contract."""

import asyncio
import json
import math
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from src.relay.config import settings
from src.relay.logging_setup import MonotonicTimer, get_logger
from src.relay.provider.base import ModelProviderBase, ProviderResponse

logger = get_logger("relay.standin")


class StandInQuotaExhaustedError(Exception):
    """Raised when simulated rate limits or daily budgets are breached."""
    def __init__(self, message: str, retry_after_s: float = 2.0, limit_type: str = "tokens"):
        super().__init__(message)
        self.message = message
        self.retry_after_s = retry_after_s
        self.limit_type = limit_type


class StandInRateTracker:
    """Tracks simulated RPM, TPM, RPD, and TPD quotas using sliding windows and budgets."""

    def __init__(
        self,
        rpm: int = 30,
        tpm: int = 6000,
        rpd: int = 14400,
        tpd: int = 500000
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.tpd = tpd

        self.daily_requests_used = 0
        self.daily_tokens_used = 0

        self._minute_requests: List[float] = []
        self._minute_tokens: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def check_and_consume(self, estimated_tokens: int) -> Optional[Dict[str, Any]]:
        """Validates quotas. Returns None if OK, or error details if a limit is exceeded."""
        async with self._lock:
            now = time.time()
            cutoff = now - 60.0

            # Prune sliding 60-second window
            self._minute_requests = [t for t in self._minute_requests if t > cutoff]
            self._minute_tokens = [(t, tok) for t, tok in self._minute_tokens if t > cutoff]

            # 1. Check Daily Budget
            if self.daily_requests_used >= self.rpd:
                return {
                    "type": "requests_per_day",
                    "message": f"Daily request budget exhausted ({self.daily_requests_used}/{self.rpd}). Resets at 00:00 UTC.",
                    "retry_after": 3600.0
                }
            if self.daily_tokens_used + estimated_tokens > self.tpd:
                return {
                    "type": "tokens_per_day",
                    "message": f"Daily token budget exhausted ({self.daily_tokens_used}/{self.tpd}). Resets at 00:00 UTC.",
                    "retry_after": 3600.0
                }

            # 2. Check Per-Minute RPM
            if len(self._minute_requests) >= self.rpm:
                oldest = self._minute_requests[0]
                wait_sec = max(0.5, 60.0 - (now - oldest))
                return {
                    "type": "requests_per_minute",
                    "message": f"Rate limit reached on requests per minute (RPM: {self.rpm}). Try again in {wait_sec:.1f}s.",
                    "retry_after": wait_sec
                }

            # 3. Check Per-Minute TPM
            current_minute_tokens = sum(tok for _, tok in self._minute_tokens)
            if current_minute_tokens + estimated_tokens > self.tpm:
                oldest = self._minute_tokens[0][0] if self._minute_tokens else now
                wait_sec = max(0.5, 60.0 - (now - oldest))
                return {
                    "type": "tokens_per_minute",
                    "message": f"Rate limit reached on tokens per minute (TPM: {self.tpm}). Used {current_minute_tokens}, Requested {estimated_tokens}. Try again in {wait_sec:.1f}s.",
                    "retry_after": wait_sec
                }

            # Quota OK: consume
            self._minute_requests.append(now)
            self._minute_tokens.append((now, estimated_tokens))
            self.daily_requests_used += 1
            self.daily_tokens_used += estimated_tokens
            return None


class StandInProvider(ModelProviderBase):
    """Controllable mock service simulating Groq LPU latency distributions and provider behaviors."""

    def __init__(
        self,
        p50_latency_ms: Optional[float] = None,
        p95_latency_ms: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        failure_rate: Optional[float] = None,
        load_slowdown_factor: Optional[float] = None,
        timeout_threshold_ms: Optional[float] = None,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        rpd: Optional[int] = None,
        tpd: Optional[int] = None,
        random_seed: int = 42
    ):
        self.p50 = p50_latency_ms or settings.STANDIN_P50_LATENCY_MS
        self.p95 = p95_latency_ms or settings.STANDIN_P95_LATENCY_MS
        self.tokens_per_sec = tokens_per_sec or settings.STANDIN_TOKENS_PER_SECOND
        self.failure_rate = failure_rate if failure_rate is not None else settings.STANDIN_FAILURE_RATE
        self.load_slowdown_factor = load_slowdown_factor or settings.STANDIN_LOAD_SLOWDOWN_FACTOR
        self.timeout_threshold_ms = timeout_threshold_ms

        self.rng = random.Random(random_seed)
        self.in_flight_count = 0
        self._lock = asyncio.Lock()

        # Quota tracker
        self.quota_tracker = StandInRateTracker(
            rpm=rpm or settings.RATE_LIMIT_RPM,
            tpm=tpm or settings.RATE_LIMIT_TPM,
            rpd=rpd or settings.RATE_LIMIT_RPD,
            tpd=tpd or settings.RATE_LIMIT_TPD
        )

        # Load empirical calibration profile if not explicitly overridden
        calib_file = settings.DATA_DIR / "calibration_profile.json"
        if p50_latency_ms is None and calib_file.exists():
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                self.p50 = calib_data.get("p50_target_ms", settings.STANDIN_P50_LATENCY_MS)
                self.p95 = calib_data.get("p95_target_ms", settings.STANDIN_P95_LATENCY_MS)
                self.tokens_per_sec = tokens_per_sec or calib_data.get("tokens_per_second", settings.STANDIN_TOKENS_PER_SECOND)
                self.mu = calib_data.get("mu", math.log(max(10.0, self.p50)))
                self.sigma = calib_data.get("sigma", max(0.05, (math.log(self.p95) - self.mu) / 1.64485))
            except Exception:
                self.p50 = settings.STANDIN_P50_LATENCY_MS
                self.p95 = settings.STANDIN_P95_LATENCY_MS
                self.tokens_per_sec = tokens_per_sec or settings.STANDIN_TOKENS_PER_SECOND
                self.mu = math.log(max(10.0, self.p50))
                p95_safe = max(self.p50 * 1.05, self.p95)
                self.sigma = max(0.05, (math.log(p95_safe) - self.mu) / 1.64485)
        else:
            self.p50 = p50_latency_ms or settings.STANDIN_P50_LATENCY_MS
            self.p95 = p95_latency_ms or settings.STANDIN_P95_LATENCY_MS
            self.tokens_per_sec = tokens_per_sec or settings.STANDIN_TOKENS_PER_SECOND
            self.mu = math.log(max(10.0, self.p50))
            p95_safe = max(self.p50 * 1.05, self.p95)
            self.sigma = max(0.05, (math.log(p95_safe) - self.mu) / 1.64485)

    def sample_base_ttft_ms(self) -> float:
        """Sample a TTFT from the calibrated log-normal distribution."""
        sample = self.rng.lognormvariate(self.mu, self.sigma)
        return max(20.0, sample)

    def compute_load_inflated_ttft(self, base_ms: float) -> float:
        """Inflates latency proportionally to the active concurrent requests."""
        concurrency_penalty = 1.0 + (self.load_slowdown_factor * max(0, self.in_flight_count - 1))
        return base_ms * concurrency_penalty

    def _synthesize_response(self, prompt: str) -> str:
        """Generates an auditable, explicitly marked synthetic diagnostic response grounded in prompt passages."""
        # Check if prompt contains retrieved research passages
        if "Paper: " in prompt and "Page: " in prompt:
            import re
            m = re.search(r'Paper:\s*([^\n\(]+).*?Page:\s*(\d+)(?:.*?Section:\s*([^\n]+))?', prompt, re.DOTALL)
            if m:
                title = m.group(1).strip()
                page = m.group(2).strip()
                sec = (m.group(3) or "Key Findings").strip()
                return (
                    f"Based on [{title}, Page {page}, Section {sec}], "
                    "the architectural parameters and experimental findings align with the published results."
                )
        if "REFUSAL" in prompt or "None" in prompt:
            return "REFUSAL: The supplied research library does not establish the answer to this question."

        return (
            "[STAND-IN SYNTHESIS: Modeled response grounded in retrieved context]\n"
            f"Analysis of query: {prompt[:80]}...\n"
            "Key factual findings derived from referenced paper passages:\n"
            "- Structural components and algorithmic parameters verified.\n"
            "- Empirical results align with documented evaluation tables."
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ProviderResponse:
        timer = MonotonicTimer()
        estimated_prompt_tokens = max(10, len(prompt) // 4)
        estimated_gen_tokens = 120

        # Check rate limits & budgets
        limit_violation = await self.quota_tracker.check_and_consume(estimated_prompt_tokens + estimated_gen_tokens)
        if limit_violation:
            total_ms = timer.mark_complete()
            return ProviderResponse(
                content="",
                tokens_in=estimated_prompt_tokens,
                tokens_out=0,
                first_token_ms=None,
                total_ms=total_ms,
                model="stand-in-simulator",
                headers={
                    "retry-after": str(round(limit_violation["retry_after"], 1)),
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-remaining-tokens": "0"
                },
                http_status=429,
                error_type="rate_limit_exceeded",
                is_simulation=True
            )

        # Check temporary failure injection
        if self.failure_rate > 0.0 and self.rng.random() < self.failure_rate:
            await asyncio.sleep(0.05)
            total_ms = timer.mark_complete()
            return ProviderResponse(
                content="",
                tokens_in=estimated_prompt_tokens,
                tokens_out=0,
                first_token_ms=None,
                total_ms=total_ms,
                model="stand-in-simulator",
                headers={},
                http_status=503,
                error_type="service_unavailable",
                is_simulation=True
            )

        # Track concurrency
        async with self._lock:
            self.in_flight_count += 1

        try:
            # Sample TTFT
            base_ttft = self.sample_base_ttft_ms()
            actual_ttft_ms = self.compute_load_inflated_ttft(base_ttft)

            # Check timeout threshold
            if self.timeout_threshold_ms and actual_ttft_ms > self.timeout_threshold_ms:
                await asyncio.sleep(self.timeout_threshold_ms / 1000.0)
                total_ms = timer.mark_complete()
                return ProviderResponse(
                    content="",
                    tokens_in=estimated_prompt_tokens,
                    tokens_out=0,
                    first_token_ms=None,
                    total_ms=total_ms,
                    model="stand-in-simulator",
                    headers={},
                    http_status=504,
                    error_type="request_timeout",
                    is_simulation=True
                )

            # Sleep TTFT duration
            await asyncio.sleep(actual_ttft_ms / 1000.0)
            first_token_ms = timer.mark_first_token()

            # Generation duration based on tokens per second
            content = self._synthesize_response(prompt)
            output_tokens = max(10, len(content) // 4)
            gen_duration_s = output_tokens / self.tokens_per_sec
            await asyncio.sleep(gen_duration_s)

            total_ms = timer.mark_complete()

            return ProviderResponse(
                content=content,
                tokens_in=estimated_prompt_tokens,
                tokens_out=output_tokens,
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                model="stand-in-simulator",
                headers={
                    "x-ratelimit-remaining-requests": str(max(0, self.quota_tracker.rpm - len(self.quota_tracker._minute_requests))),
                    "x-ratelimit-remaining-tokens": str(max(0, self.quota_tracker.tpm - sum(t for _, t in self.quota_tracker._minute_tokens)))
                },
                http_status=200,
                error_type=None,
                is_simulation=True
            )
        finally:
            async with self._lock:
                self.in_flight_count = max(0, self.in_flight_count - 1)

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> AsyncIterator[str]:
        """Streams synthetic response broken into sequential words with inter-token delays."""
        estimated_prompt_tokens = max(10, len(prompt) // 4)
        estimated_gen_tokens = 120

        limit_violation = await self.quota_tracker.check_and_consume(estimated_prompt_tokens + estimated_gen_tokens)
        if limit_violation:
            raise StandInQuotaExhaustedError(
                limit_violation["message"],
                retry_after_s=limit_violation["retry_after"],
                limit_type=limit_violation["type"]
            )

        async with self._lock:
            self.in_flight_count += 1

        try:
            base_ttft = self.sample_base_ttft_ms()
            actual_ttft_ms = self.compute_load_inflated_ttft(base_ttft)
            await asyncio.sleep(actual_ttft_ms / 1000.0)

            content = self._synthesize_response(prompt)
            words = content.split(" ")
            delay_per_word = max(0.002, (1.0 / self.tokens_per_sec) * 1.3)

            for w in words:
                await asyncio.sleep(delay_per_word)
                yield w + " "
        finally:
            async with self._lock:
                self.in_flight_count = max(0, self.in_flight_count - 1)
