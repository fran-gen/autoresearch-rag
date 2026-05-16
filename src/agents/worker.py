from __future__ import annotations

import asyncio
import random
from pathlib import Path

from src.agents.state import ResearchLabState
from src.benchmark.loader import BenchmarkDocument, BenchmarkQuestion, EnterpriseRagBenchLoader
from src.benchmark.metrics import composite_score
from src.benchmark.runner import BenchmarkRunner
from src.config import get_settings
from src.models import ExperimentResult, ExperimentStatus, RetrievalConfig


def _sample_questions_for_ab(
    questions: list[BenchmarkQuestion],
    run_id: str,
    sample_size: int = 60,
) -> list[BenchmarkQuestion]:
    if len(questions) <= sample_size:
        return questions
    # Keep one stable tuning subset per run so iteration-to-iteration scores
    # remain comparable for users.
    rnd = random.Random(f"ab:{run_id}")
    return rnd.sample(questions, sample_size)


def _split_tuning_holdout(
    questions: list[BenchmarkQuestion],
    run_id: str,
) -> tuple[list[BenchmarkQuestion], list[BenchmarkQuestion]]:
    if len(questions) < 10:
        return questions, questions
    rnd = random.Random(f"holdout:{run_id}")
    shuffled = list(questions)
    rnd.shuffle(shuffled)
    holdout_size = max(5, int(len(shuffled) * 0.2))
    return shuffled[holdout_size:], shuffled[:holdout_size]


def _filter_questions_for_focus(
    questions: list[BenchmarkQuestion],
    question_focus: str | None,
) -> list[BenchmarkQuestion]:
    focus = (question_focus or "all").strip()
    if not focus or focus == "all":
        return questions
    filtered = [q for q in questions if (q.question_type or "unknown") == focus]
    return filtered or questions


async def _evaluate(
    runner: BenchmarkRunner,
    documents: list[BenchmarkDocument],
    sampled_questions: list[BenchmarkQuestion],
    config: RetrievalConfig,
):
    fn = runner.run if config.evaluation_mode == "full" else runner.run_fast
    return await asyncio.to_thread(fn, documents, sampled_questions, config)


async def worker_agent(state: ResearchLabState) -> ResearchLabState:
    if not state["experiment_queue"]:
        state["current_phase"] = "worker"
        return state

    settings = get_settings()
    benchmark_root = Path(state.get("benchmark_root") or settings.benchmark_root)
    loader = EnterpriseRagBenchLoader(benchmark_root)
    documents, questions = loader.load_mvp_subset()
    questions = _filter_questions_for_focus(questions, state.get("question_focus"))
    sampled_questions = _sample_questions_for_ab(
        questions=questions,
        run_id=state.get("run_id") or "run",
    )
    tuning_questions, holdout_questions = _split_tuning_holdout(
        sampled_questions,
        state.get("run_id") or "run",
    )
    state["dataset_readiness"] = {
        "documents": len(documents),
        "questions": len(questions),
        "sampled_questions": len(sampled_questions),
        "holdout_questions": len(holdout_questions),
        "question_focus": state.get("question_focus") or "all",
    }

    experiment = state["experiment_queue"][0]
    incumbent_config = state["best_config"] or experiment.retrieval_config
    runner = BenchmarkRunner()

    # A/B: evaluate incumbent and candidate on the same sampled subset.
    _, incumbent_metrics = await _evaluate(
        runner, documents, tuning_questions, incumbent_config
    )
    incumbent_score = composite_score(incumbent_metrics)

    question_results, metrics = await _evaluate(
        runner, documents, tuning_questions, experiment.retrieval_config
    )
    candidate_score = composite_score(metrics)
    _, incumbent_validation_metrics = await _evaluate(
        runner, documents, holdout_questions, incumbent_config
    )
    _, candidate_validation_metrics = await _evaluate(
        runner, documents, holdout_questions, experiment.retrieval_config
    )
    incumbent_validation_score = composite_score(incumbent_validation_metrics)
    candidate_validation_score = composite_score(candidate_validation_metrics)

    result = ExperimentResult(
        experiment_id=experiment.id,
        status=ExperimentStatus.completed,
        metrics=metrics,
        composite_score=candidate_score,
        baseline_score=incumbent_score,
        delta_vs_baseline=candidate_score - incumbent_score,
        validation_score=candidate_validation_score,
        validation_delta=candidate_validation_score - incumbent_validation_score,
        question_results=question_results,
    )
    state["results"] = state["results"] + [result]
    state["completed_experiments"] = state["completed_experiments"] + [experiment]
    state["experiment_queue"] = state["experiment_queue"][1:]
    state["current_phase"] = "worker"
    return state
