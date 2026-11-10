"""Token bucket and sliding window rate limiter for proactive client-side throttling."""

import asyncio
import time
from typing import List, Optional, Tuple
from src.relay.config import settings


class ClientRateLimiter:
    """Proactively throttles requests to adhere to provider RPM, TPM, RPD, and TPD allocations."""

    def __init__(
        self,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        rpd: Optional[int] = None,
        tpd: Optional[int] = None
    ):
        self.rpm = rpm or settings.RATE_LIMIT_RPM
        self.tpm = tpm or settings.RATE_LIMIT_TPM
        self.rpd = rpd or settings.RATE_LIMIT_RPD
        self.tpd = tpd or settings.RATE_LIMIT_TPD

        self._minute_requests: List[float] = []
        self._minute_tokens: List[Tuple[float, int]] = []
        self.daily_requests_used: int = 0
        self.daily_tokens_used: int = 0

        self._lock = asyncio.Lock()

    async def acquire_permit(self, estimated_tokens: int = 150) -> Tuple[bool, Optional[float], str]:
        """Attempts to acquire rate-limit permit.

        Returns:
            (granted: bool, wait_seconds_suggested: Optional[float], reason: str)
        """
        async with self._lock:
            now = time.time()
            cutoff = now - 60.0

            # Prune 60-second sliding windows
            self._minute_requests = [t for t in self._minute_requests if t > cutoff]
            self._minute_tokens = [(t, tok) for t, tok in self._minute_tokens if t > cutoff]

            # 1. Daily Quota Check
            if self.daily_requests_used >= self.rpd:
                return False, None, "Daily request quota exhausted (RPD limit reached)."
            if self.daily_tokens_used + estimated_tokens > self.tpd:
                return False, None, "Daily token quota exhausted (TPD limit reached)."

            # 2. Per-Minute RPM Check
            if len(self._minute_requests) >= self.rpm:
                oldest = self._minute_requests[0]
                wait_sec = max(0.1, 60.0 - (now - oldest))
                return False, wait_sec, f"RPM limit reached ({self.rpm} req/min). Backoff suggested."

            # 3. Per-Minute TPM Check
            current_tokens = sum(tok for _, tok in self._minute_tokens)
            if current_tokens + estimated_tokens > self.tpm:
                oldest_token_time = self._minute_tokens[0][0] if self._minute_tokens else now
                wait_sec = max(0.1, 60.0 - (now - oldest_token_time))
                return False, wait_sec, f"TPM limit reached ({self.tpm} tokens/min). Backoff suggested."

            # Granted: commit consumption
            self._minute_requests.append(now)
            self._minute_tokens.append((now, estimated_tokens))
            self.daily_requests_used += 1
            self.daily_tokens_used += estimated_tokens
            return True, 0.0, "Permit granted"

    async def reset(self):
        """Resets all active counters (for test setups)."""
        async with self._lock:
            self._minute_requests.clear()
            self._minute_tokens.clear()
            self.daily_requests_used = 0
            self.daily_tokens_used = 0
