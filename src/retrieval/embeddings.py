from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_HF = _PROJECT_ROOT / ".hf_cache"


def _configure_cpu_threads() -> None:
    """Keep local embedding inference from fanning out across all host cores."""
    threads = os.environ.get("MODEL_INFERENCE_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_configure_cpu_threads()

from sentence_transformers import SentenceTransformer  # noqa: E402

try:  # pragma: no cover - torch availability depends on deployment image
    import torch  # noqa: E402

    torch_threads = max(1, int(os.environ.get("MODEL_INFERENCE_THREADS", "1")))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
except Exception:
    pass


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


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = _load_sentence_transformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
