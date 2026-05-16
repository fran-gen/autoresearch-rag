from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Hypothesis(BaseModel):
    id: str
    title: str
    rationale: str
    expected_impact: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievalConfig(BaseModel):
    strategy: str = "dense"
    embedding_model: str
    top_k: int = 8
    use_reranker: bool = False
    reranker_model: str | None = None
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    evaluation_mode: str = "fast"
    extra: dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    id: str
    hypothesis_id: str
    name: str
    description: str
    retrieval_config: RetrievalConfig
    question_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    run_position: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkMetrics(BaseModel):
    total_questions: int = 0
    answered_questions: int = 0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    answer_correctness: float | None = None
    avg_latency_ms: float = 0.0
    invalid_extra_docs_rate: float = 0.0


class QuestionResult(BaseModel):
    question_id: str
    answer: str
    document_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    is_correct: bool | None = None


class ExperimentResult(BaseModel):
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.completed
    metrics: BenchmarkMetrics
    composite_score: float = 0.0
    baseline_score: float | None = None
    delta_vs_baseline: float | None = None
    validation_score: float | None = None
    validation_delta: float | None = None
    accepted: bool = False
    question_results: list[QuestionResult] = Field(default_factory=list)
    failure_analysis: str = ""
    improvement_summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
