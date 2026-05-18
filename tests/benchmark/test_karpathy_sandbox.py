import pytest

from src.benchmark.karpathy_sandbox import _CappedRetriever
from src.retrieval.base import BaseRetriever


class FakeRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 8):
        return []


def test_capped_retriever_enforces_per_question_call_budget():
    retriever = _CappedRetriever(
        FakeRetriever(),
        top_k_cap=10,
        calls_per_question_cap=2,
    )

    retriever.begin_question()
    retriever.retrieve("q", top_k=5)
    retriever.retrieve("q again", top_k=5)
    with pytest.raises(RuntimeError, match="retrieve call budget"):
        retriever.retrieve("one too many", top_k=5)

    retriever.begin_question()
    retriever.retrieve("next question", top_k=5)
