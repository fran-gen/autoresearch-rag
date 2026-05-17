from __future__ import annotations

from collections import defaultdict

from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever, RetrievedDocument
from src.retrieval.dense import DenseRecord


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        records: list[DenseRecord],
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.records = records
        tokenized = [r.text.lower().split() for r in records]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedDocument]:
        if not self.records:
            return []

        scores = defaultdict(float)
        by_id: dict[str, RetrievedDocument] = {}

        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        for rank, doc in enumerate(dense_results, start=1):
            fused = self.dense_weight * (1.0 / (60 + rank))
            scores[doc.document_id] += fused
            by_id[doc.document_id] = doc

        if self.bm25 is not None:
            tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokens)
            indexed = list(enumerate(bm25_scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            for rank, (idx, _) in enumerate(indexed[: top_k * 2], start=1):
                record = self.records[idx]
                fused = self.bm25_weight * (1.0 / (60 + rank))
                scores[record.document_id] += fused
                by_id.setdefault(
                    record.document_id,
                    RetrievedDocument(
                        document_id=record.document_id,
                        text=record.text,
                        score=0.0,
                        metadata=record.metadata,
                    ),
                )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[RetrievedDocument] = []
        for doc_id, score in ranked:
            doc = by_id[doc_id]
            doc.score = float(score)
            results.append(doc)
        return results
