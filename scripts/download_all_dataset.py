#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark.loader import EnterpriseRagBenchLoader
from src.config import get_settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_all_dataset")


def main() -> int:
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    logger.info("Dataset download cwd: %s", Path.cwd())
    logger.info("Dataset download repo root: %s", _REPO_ROOT)
    logger.info("Dataset download BENCHMARK_ROOT env: %s", os.environ.get("BENCHMARK_ROOT", "<unset>"))
    logger.info("Dataset download benchmark root: %s", settings.benchmark_root.resolve())
    logger.info("Dataset download documents dir: %s", loader.documents_dir.resolve())
    logger.info("Dataset download bench dir: %s", loader.bench_dir.resolve())
    logger.info("Dataset download archive path: %s", (settings.benchmark_root / "all_documents.zip").resolve())
    loader.download_release_files(include_all_documents=True, document_fraction=1.0)
    logger.info("Downloaded full dataset under %s.", settings.benchmark_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
