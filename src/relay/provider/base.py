"""Model provider base interface and response models."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional
from pydantic import BaseModel


class ProviderResponse(BaseModel):
    content: str
    tokens_in: int
    tokens_out: int
    first_token_ms: Optional[float] = None
    total_ms: float
    model: str
    headers: Dict[str, str] = {}
    http_status: int = 200
    error_type: Optional[str] = None
    is_simulation: bool = False


class ModelProviderBase(ABC):
    """Abstract interface implemented by both GroqProvider and StandInProvider."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> ProviderResponse:
        """Execute a non-streaming completion."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> AsyncIterator[str]:
        """Execute a token-streaming completion yielding token chunks."""
        pass
