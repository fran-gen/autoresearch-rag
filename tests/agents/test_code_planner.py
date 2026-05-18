from src.agents.code_planner import _fallback_code, _format_code_history, _is_semantic_noop


def test_semantic_noop_ignores_comments_and_docstrings():
    current = '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    """Original docstring."""
    return retriever.retrieve(question.question, top_k=top_k)
'''
    proposed = '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    """Different docstring."""
    # This comment should not count as a real pipeline edit.
    return retriever.retrieve(question.question, top_k=top_k)
'''

    assert _is_semantic_noop(proposed, current) is True


def test_semantic_noop_detects_real_retrieve_change():
    current = '''\
def retrieve(question, retriever, top_k):
    return retriever.retrieve(question.question, top_k=top_k)
'''
    proposed = '''\
def retrieve(question, retriever, top_k):
    docs = retriever.retrieve(question.question, top_k=top_k)
    if question.question_type == "comparison":
        docs = retriever.retrieve(question.question + " comparison", top_k=top_k)
    return docs[:top_k]
'''

    assert _is_semantic_noop(proposed, current) is False


def test_fallback_code_changes_pass_through_pipeline():
    current = '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    return retriever.retrieve(question.question, top_k=top_k)
'''

    fallback = _fallback_code({"current_pipeline_code": current})

    assert _is_semantic_noop(fallback, current) is False


def test_code_history_includes_rejection_reason():
    text = _format_code_history({
        "code_history": [{
            "accepted": False,
            "score": 0.0,
            "hypothesis": "Try reranker",
            "reason": "Rejected: sandbox timed out.",
            "proposed_code": "def retrieve(question, retriever, top_k):\n    return []",
            "proposed_config": {"use_reranker": True},
        }]
    })

    assert "Rejected: sandbox timed out." in text
