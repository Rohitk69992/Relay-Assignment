"""Domain schemas, data models, and protocol definitions for The Relay."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RequestStatus(str, Enum):
    OK = "ok"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ERROR = "error"


class AnswerGrade(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    REFUSED_CORRECTLY = "refused_correctly"
    REFUSED_WRONGLY = "refused_wrongly"


class SourceType(str, Enum):
    REAL = "real"
    STANDIN = "stand-in"


class Citation(BaseModel):
    paper_id: str
    paper_title: str
    page: int
    section: Optional[str] = None
    quote: Optional[str] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    page_number: int
    section_name: str
    text: str
    char_count: int
    estimated_tokens: int


class RetrievalSetting(BaseModel):
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4
    rerank: bool = True

    @property
    def label(self) -> str:
        rerank_str = "rerank" if self.rerank else "no-rerank"
        return f"cs{self.chunk_size}_k{self.top_k}_{rerank_str}"


class RetrievalResult(BaseModel):
    chunks: List[DocumentChunk]
    query: str
    retrieval_time_ms: float
    is_sufficient: bool = True
    refinement_performed: bool = False
    sufficiency_reason: Optional[str] = None


class PipelineRequest(BaseModel):
    request_id: str = Field(description="Unique identifier for the request")
    question: str = Field(description="User question to answer")
    scenario: str = Field(default="interactive", description="Benchmark scenario or interactive label")
    source: SourceType = Field(default=SourceType.STANDIN)
    retrieval_setting: RetrievalSetting = Field(default_factory=RetrievalSetting)
    stream: bool = Field(default=False)


class PipelineResponse(BaseModel):
    request_id: str
    scenario: str
    source: SourceType
    start_time: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    status: RequestStatus
    answer_grade: AnswerGrade
    first_token_ms: Optional[float] = None
    total_ms: float
    queue_wait_ms: float = 0.0
    processing_ms: float = 0.0
    retry_wait_ms: float = 0.0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting: str
    error_message: Optional[str] = None


class ResultsCsvRow(BaseModel):
    """Represents an exact row in results.csv."""
    request_id: str
    scenario: str
    source: str  # 'real' or 'stand-in'
    start_time: str
    first_token_ms: Optional[float] = None
    total_ms: float
    status: str  # 'ok', 'rate_limited', 'timeout', 'error'
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: str  # 'correct', 'wrong', 'refused_correctly', 'refused_wrongly'
    # Extended telemetry columns
    queue_wait_ms: Optional[float] = None
    processing_ms: Optional[float] = None
    retry_wait_ms: Optional[float] = None
    model: Optional[str] = None
    error_type: Optional[str] = None


class EvaluationItem(BaseModel):
    question_id: str
    category: str
    question: str
    expected_answer: str
    expected_paper: str
    expected_page: int
    expected_section: str
    is_answerable: bool
    distractor_notes: Optional[str] = None
