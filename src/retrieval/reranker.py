from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.retrieval.base import RetrievedDocument


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        output: list[RetrievedDocument] = []
        for candidate, score in ranked:
            candidate.score = float(score)
            output.append(candidate)
        if top_k is not None:
            return output[:top_k]
        return output
