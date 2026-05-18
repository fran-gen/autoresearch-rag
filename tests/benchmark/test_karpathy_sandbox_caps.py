from src.benchmark.karpathy_sandbox import _CappedReranker, _CappedRetriever
from src.retrieval.base import BaseRetriever, RetrievedDocument


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedDocument]:
        self.requested_top_k = top_k
        return [
            RetrievedDocument(document_id=f"doc{i}", text=query, score=float(i))
            for i in range(top_k)
        ]


class DummyReranker:
    def __init__(self) -> None:
        self.seen_candidate_count = 0

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        self.seen_candidate_count = len(candidates)
        return candidates[:top_k]


def test_capped_retriever_clamps_generated_top_k_requests():
    inner = DummyRetriever()
    retriever = _CappedRetriever(inner, top_k_cap=24)

    docs = retriever.retrieve("query", top_k=80)

    assert inner.requested_top_k == 24
    assert len(docs) == 24


def test_capped_reranker_clamps_cross_encoder_candidate_batches():
    inner = DummyReranker()
    reranker = _CappedReranker(inner, candidate_cap=12)
    candidates = [
        RetrievedDocument(document_id=f"doc{i}", text="body", score=float(i))
        for i in range(40)
    ]

    docs = reranker.rerank("query", candidates, top_k=6)

    assert inner.seen_candidate_count == 12
    assert len(docs) == 6
