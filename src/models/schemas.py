"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_url: Optional[str] = None
    content: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    doc_type: str = "markdown"


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str
    source: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str
    model: str
    latency_ms: float


class EvalRequest(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    citations: List[Citation]


class EvalResult(BaseModel):
    faithfulness: float
    relevance: float
    correctness: float
    hallucination_detected: bool
    overall_score: float


class AdminStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_latency_ms: float
    collection_size_mb: float
    last_ingestion: Optional[datetime]
