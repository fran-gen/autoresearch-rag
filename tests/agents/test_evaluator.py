import pytest

import src.agents.evaluator as evaluator_module
from src.agents.evaluator import _generate_final_report, _recommend_next_action
from src.models import BenchmarkMetrics, ExperimentResult, ExperimentStatus


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
