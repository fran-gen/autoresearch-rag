from __future__ import annotations

from typing import Any, TypedDict

from src.models import ExperimentResult, ExperimentSpec, Hypothesis, RetrievalConfig


class ResearchLabState(TypedDict):
    iteration: int
    max_iterations: int
    should_stop: bool
    current_phase: str
    run_id: str
    hypotheses: list[Hypothesis]
    experiment_queue: list[ExperimentSpec]
    completed_experiments: list[ExperimentSpec]
    results: list[ExperimentResult]
    latest_summary: str
    best_config: RetrievalConfig | None
    best_score: float
    initial_baseline_score: float
    min_improvement_delta: float
    accepted_experiments: int
    rejected_experiments: int
    per_type_summary: str
    failure_examples: list[dict[str, Any]]
    planner_rationale: str
    candidate_config: RetrievalConfig | None
    final_report: str
    score_history: list[dict[str, Any]]
    question_focus: str
    benchmark_root: str | None
    research_setup_id: str | None
    tried_config_fingerprints: list[str]
    rejected_config_fingerprints: list[str]
    failure_taxonomy: dict[str, int]
    recommendation: str
    validation_summary: str
    dataset_readiness: dict[str, Any]
    # Karpathy mode fields
    research_mode: str
    proposed_code: str
    proposed_config: RetrievalConfig | None
    initial_pipeline_code: str
    current_pipeline_code: str
    code_history: list[dict[str, Any]]
    karpathy_branch: str
