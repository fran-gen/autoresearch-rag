"""Smoke test: fast retrieval evaluation against local Qdrant (no API server).

Run from repo root:

    python -m scripts.test_fast_loop

Expects:
  - ./index/qdrant_data with collection ``enterprise_docs``
  - ./data/docs/*.txt and ./data/bench/questions_subset.jsonl
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HF_CACHE = _REPO_ROOT / ".hf_cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE))

from src.benchmark.loader import EnterpriseRagBenchLoader
from src.benchmark.metrics import composite_score
from src.benchmark.runner import BenchmarkRunner
from src.models import RetrievalConfig


def main() -> None:
    root = _REPO_ROOT
    loader = EnterpriseRagBenchLoader(root / "data")
    if not loader.benchmark_exists():
        raise SystemExit(
            "Benchmark data missing: need data/docs/ with .txt files and "
            "data/bench/questions_subset.jsonl (or bench/questions.jsonl)."
        )

    # Limit docs for a quick smoke test; retrieval uses Qdrant only (pre-indexed corpus).
    documents, questions = loader.load_mvp_subset(max_docs=5000, max_questions=80)
    cfg = RetrievalConfig(
        embedding_model="BAAI/bge-base-en-v1.5",
        top_k=8,
        strategy="dense",
        evaluation_mode="fast",
    )
    runner = BenchmarkRunner()
    _, metrics = runner.run_fast(documents, questions, cfg)
    score = composite_score(metrics)
    print(f"Questions: {metrics.total_questions}")
    print(f"Recall@k: {metrics.recall_at_k:.4f}")
    print(f"Precision@k: {metrics.precision_at_k:.4f}")
    print(f"Avg latency ms: {metrics.avg_latency_ms:.1f}")
    print(f"Invalid extra doc rate: {metrics.invalid_extra_docs_rate:.4f}")
    print(f"Composite (fast): {score:.4f}")


if __name__ == "__main__":
    main()
