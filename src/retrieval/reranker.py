from __future__ import annotations

import os
from functools import lru_cache

_THREADS = os.environ.get("MODEL_INFERENCE_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1"
for _NAME in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_NAME, _THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder

from src.retrieval.base import RetrievedDocument


def _model_threads() -> int:
    try:
        return max(1, int(os.environ.get("MODEL_INFERENCE_THREADS", "1")))
    except ValueError:
        return 1


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = _load_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=max(1, min(8, _model_threads() * 4)),
        )
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
