from __future__ import annotations

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_HF = _PROJECT_ROOT / ".hf_cache"


def _ensure_local_hf_cache() -> None:
    """Use repo-local HF cache when ~/.cache is not writable (CI/sandbox)."""
    if os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"):
        return
    try:
        _LOCAL_HF.mkdir(parents=True, exist_ok=True)
        test_file = _LOCAL_HF / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except OSError:
        return
    os.environ.setdefault("HF_HOME", str(_LOCAL_HF))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_LOCAL_HF))


_ensure_local_hf_cache()


class EmbeddingEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
