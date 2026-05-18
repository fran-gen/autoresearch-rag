from src.benchmark.metrics import composite_score, compute_retrieval_metrics
from src.models import QuestionResult


def test_retrieval_only_counts_docs_for_no_ground_truth_as_extra():
    metrics = compute_retrieval_metrics(
        question_results=[
            QuestionResult(question_id="q_info", answer="", document_ids=["doc1"]),
        ],
        ground_truth_by_question={"q_info": set()},
        top_k=5,
        retrieval_only=True,
    )

    assert metrics.invalid_extra_docs_rate == 1.0
    assert composite_score(metrics) == 0.0


def test_retrieval_only_rewards_abstention_for_no_ground_truth():
    metrics = compute_retrieval_metrics(
        question_results=[
            QuestionResult(question_id="q_info", answer="", document_ids=[]),
        ],
        ground_truth_by_question={"q_info": set()},
        top_k=5,
        retrieval_only=True,
    )

    assert metrics.invalid_extra_docs_rate == 0.0
    assert composite_score(metrics) == 0.2
