import pytest

import src.agents.evaluator as evaluator_module
from src.agents.evaluator import evaluator_agent, _generate_final_report, _recommend_next_action
from src.models import BenchmarkMetrics, ExperimentResult, ExperimentSpec, ExperimentStatus, RetrievalConfig


def test_recommendation_for_missing_relevant_docs():
    rec = _recommend_next_action(
        "semantic: recall=0.2 precision=0.3",
        {
            "no_relevant_doc_retrieved": 8,
            "retrieval_noise": 2,
            "answer_failed_with_context": 1,
            "unknown_or_unlabeled": 0,
        },
    )
    assert "missing" in rec.lower() or "top-k" in rec.lower()


def test_recommendation_for_retrieval_noise():
    rec = _recommend_next_action(
        "basic: recall=0.9 precision=0.2",
        {
            "no_relevant_doc_retrieved": 1,
            "retrieval_noise": 6,
            "answer_failed_with_context": 0,
            "unknown_or_unlabeled": 0,
        },
    )
    assert "noisy" in rec.lower() or "reranking" in rec.lower()


def _config() -> RetrievalConfig:
    return RetrievalConfig(
        strategy="dense",
        embedding_model="test-model",
        top_k=8,
        use_reranker=False,
    )


def _state_for_result(iteration: int, delta: float) -> dict:
    config = _config()
    return {
        "iteration": iteration,
        "max_iterations": 5,
        "should_stop": False,
        "hypotheses": [],
        "completed_experiments": [
            ExperimentSpec(
                id="exp_tie",
                hypothesis_id="hyp_tie",
                name="Tie",
                description="Tie candidate",
                retrieval_config=config,
            )
        ],
        "results": [
            ExperimentResult(
                experiment_id="exp_tie",
                status=ExperimentStatus.completed,
                metrics=BenchmarkMetrics(total_questions=1, answered_questions=1),
                composite_score=0.5 + delta,
                baseline_score=0.5,
                delta_vs_baseline=delta,
                validation_score=0.5,
                validation_delta=0.0,
            )
        ],
        "latest_summary": "",
        "best_config": config,
        "best_score": 0.5,
        "initial_baseline_score": 0.5,
        "min_improvement_delta": 0.005,
        "accepted_experiments": 0,
        "rejected_experiments": 0,
        "score_history": [],
        "rejected_config_fingerprints": [],
        "planner_rationale": "",
        "research_mode": "config",
        "benchmark_root": None,
    }


@pytest.mark.anyio
async def test_second_experiment_accepts_equal_incumbent_score(monkeypatch):
    monkeypatch.setattr(evaluator_module, "_build_per_type_breakdown", lambda *args: ("No per-type data.", []))
    monkeypatch.setattr(evaluator_module, "_classify_failures", lambda *args: {})

    state = await evaluator_agent(_state_for_result(iteration=1, delta=0.0))

    assert state["accepted_experiments"] == 1
    assert state["rejected_experiments"] == 0
    assert state["results"][-1].accepted is True
    assert "Accepted" in state["score_history"][-1]["reason"]


@pytest.mark.anyio
async def test_first_experiment_still_requires_min_delta(monkeypatch):
    monkeypatch.setattr(evaluator_module, "_build_per_type_breakdown", lambda *args: ("No per-type data.", []))
    monkeypatch.setattr(evaluator_module, "_classify_failures", lambda *args: {})

    state = await evaluator_agent(_state_for_result(iteration=0, delta=0.0))

    assert state["accepted_experiments"] == 0
    assert state["rejected_experiments"] == 1
    assert state["results"][-1].accepted is False
    assert "by >= 0.0050" in state["score_history"][-1]["reason"]


@pytest.mark.anyio
async def test_final_report_handles_failed_results_without_delta(monkeypatch):
    class Settings:
        has_google_key = False

    monkeypatch.setattr(evaluator_module, "get_settings", lambda: Settings())
    state = {
        "iteration": 3,
        "accepted_experiments": 0,
        "rejected_experiments": 3,
        "initial_baseline_score": -1.0,
        "best_score": -1.0,
        "best_config": None,
        "per_type_summary": "No per-type data.",
        "results": [
            ExperimentResult(
                experiment_id="exp_failed",
                status=ExperimentStatus.failed,
                metrics=BenchmarkMetrics(),
                composite_score=0.0,
                baseline_score=None,
                delta_vs_baseline=None,
                accepted=False,
                failure_analysis="Sandbox timeout.",
            )
        ],
    }

    report = await _generate_final_report(state)

    assert "3 iterations" in report
    assert "0 accepted" in report
    assert "n/a" in report
