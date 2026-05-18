import asyncio
import json

from src.agents.evaluator import _recommend_next_action
from src.agents.evaluator import _build_per_type_breakdown
from src.agents.evaluator import evaluator_agent
from src.agents.graph import default_state
from src.models import BenchmarkMetrics, ExperimentResult, ExperimentSpec, ExperimentStatus, QuestionResult


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


def test_evaluator_accepts_any_positive_delta_by_default():
    state = default_state(max_iterations=2)
    config = state["best_config"]
    assert config is not None
    experiment = ExperimentSpec(
        id="exp_small_win",
        hypothesis_id="hyp_1",
        name="small win",
        description="small positive score lift",
        retrieval_config=config,
    )
    result = ExperimentResult(
        experiment_id=experiment.id,
        status=ExperimentStatus.completed,
        metrics=BenchmarkMetrics(total_questions=1, recall_at_k=0.3, precision_at_k=0.1),
        composite_score=0.2826,
        baseline_score=0.2792,
        delta_vs_baseline=0.0034,
    )
    state["completed_experiments"] = [experiment]
    state["results"] = [result]

    updated = asyncio.run(evaluator_agent(state))

    assert updated["results"][0].accepted is True
    assert updated["accepted_experiments"] == 1
    assert updated["rejected_experiments"] == 0
    assert updated["best_score"] == 0.2826


def test_per_type_breakdown_does_not_focus_unscored_info_not_found(tmp_path):
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    questions = [
        {
            "question_id": "q1",
            "question": "answerable",
            "question_type": "basic",
            "source_types": [],
            "expected_doc_ids": ["doc1"],
            "gold_answer": "",
            "answer_facts": [],
        },
        {
            "question_id": "q2",
            "question": "not answerable",
            "question_type": "info_not_found",
            "source_types": [],
            "expected_doc_ids": [],
            "gold_answer": "",
            "answer_facts": [],
        },
    ]
    (bench_dir / "questions_subset.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions),
        encoding="utf-8",
    )

    summary, failures, recalls = _build_per_type_breakdown(
        [
            QuestionResult(question_id="q1", answer="", document_ids=["doc1"]),
            QuestionResult(question_id="q2", answer="", document_ids=["doc2"]),
        ],
        top_k=5,
        benchmark_root=str(tmp_path),
    )

    assert recalls == {"basic": 1.0}
    assert "info_not_found: skipped" in summary
    assert [item["question_id"] for item in failures] == ["q1"]
