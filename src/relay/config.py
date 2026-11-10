"""Configuration management for The Relay pipeline."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderMode(str, Enum):
    GROQ = "groq"
    STANDIN = "standin"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    PAPERS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "papers")

    # Provider Mode
    PROVIDER_MODE: ProviderMode = Field(
        default=ProviderMode.STANDIN,
        description="Mode of the model provider: 'groq' for live hosted API, 'standin' for controlled simulation"
    )

    # Groq Configuration
    GROQ_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Groq API key loaded from environment variable or .env file"
    )
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for Groq API"
    )
    GROQ_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Active model identifier on Groq"
    )

    # Stand-in Simulation Parameters
    STANDIN_P50_LATENCY_MS: float = Field(default=350.0, ge=10.0, le=10000.0)
    STANDIN_P95_LATENCY_MS: float = Field(default=850.0, ge=20.0, le=20000.0)
    STANDIN_TOKENS_PER_SECOND: float = Field(default=450.0, ge=10.0, le=2000.0)
    STANDIN_FAILURE_RATE: float = Field(default=0.0, ge=0.0, le=1.0)
    STANDIN_LOAD_SLOWDOWN_FACTOR: float = Field(default=0.05, ge=0.0, le=1.0)
    STANDIN_RANDOM_SEED: int = Field(default=42)

    # Concurrency and Rate Limits
    RATE_LIMIT_RPM: int = Field(default=30, ge=1, le=10000, description="Requests Per Minute allocation")
    RATE_LIMIT_TPM: int = Field(default=6000, ge=100, le=1000000, description="Tokens Per Minute allocation")
    RATE_LIMIT_RPD: int = Field(default=14400, ge=1, le=1000000, description="Requests Per Day allocation")
    RATE_LIMIT_TPD: int = Field(default=500000, ge=100, le=100000000, description="Tokens Per Day allocation")
    MAX_CONCURRENT_REQUESTS: int = Field(default=5, ge=1, le=100, description="Max concurrent in-flight model calls")
    MAX_QUEUE_SIZE: int = Field(default=50, ge=1, le=1000, description="Max requests buffered in admission queue")
    QUEUE_TIMEOUT_SECONDS: float = Field(default=10.0, ge=0.5, le=120.0, description="Max wait time before queue drop")
    REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, ge=1.0, le=180.0, description="End-to-end request deadline")

    # Retry Policy
    MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    RETRY_BASE_DELAY_SECONDS: float = Field(default=0.5, ge=0.05, le=10.0)
    RETRY_MAX_DELAY_SECONDS: float = Field(default=5.0, ge=0.5, le=60.0)
    RETRY_JITTER: float = Field(default=0.2, ge=0.0, le=1.0)

    # Retrieval Defaults
    DEFAULT_CHUNK_SIZE: int = Field(default=600, ge=100, le=4000)
    DEFAULT_CHUNK_OVERLAP: int = Field(default=100, ge=0, le=1000)
    DEFAULT_TOP_K: int = Field(default=4, ge=1, le=20)
    DEFAULT_RERANK_ENABLED: bool = Field(default=True)

    # Pricing (Estimated / Official reference per 1M tokens)
    INPUT_PRICE_PER_MILLION_USD: float = Field(default=0.05)
    OUTPUT_PRICE_PER_MILLION_USD: float = Field(default=0.08)

    # Server Configuration
    SERVER_HOST: str = Field(default="127.0.0.1")
    SERVER_PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")

    @field_validator("DEFAULT_CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, v, info):
        chunk_size = info.data.get("DEFAULT_CHUNK_SIZE", 600)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be strictly less than chunk size ({chunk_size})")
        return v

    def is_groq_available(self) -> bool:
        return self.GROQ_API_KEY is not None and bool(self.GROQ_API_KEY.get_secret_value().strip())


# Global settings instance
settings = Settings()
