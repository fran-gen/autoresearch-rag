from __future__ import annotations

import json
import sys
from pathlib import Path

from src.benchmark.karpathy_sandbox import run_karpathy_benchmark
from src.benchmark.metrics import composite_score
from src.models import RetrievalConfig
from src.retrieval.pipeline import retrieve

RESULT_PREFIX = "__KARPATHY_RESULT__"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m src.benchmark.karpathy_sandbox_runner /path/to/state.json", file=sys.stderr)
        return 2

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    config_payload = payload.get("best_config")
    config = RetrievalConfig.model_validate(config_payload) if config_payload else None

    question_results, metrics = run_karpathy_benchmark(
        retrieve,
        benchmark_root=payload.get("benchmark_root"),
        question_focus=payload.get("question_focus") or "all",
        config=config,
    )
    result = {
        "question_results": [item.model_dump() for item in question_results],
        "metrics": metrics.model_dump(),
        "composite_score": composite_score(metrics),
    }
    print(f"{RESULT_PREFIX}{json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
