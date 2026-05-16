from src.agents.evaluator import _recommend_next_action


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
