#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
logger = logging.getLogger("download_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EnterpriseRAG-Bench data.")
    parser.add_argument(
        "size",
        choices=("full", "half"),
        help="Download the full dataset or extract half of the document archive.",
    )
    parser.add_argument(
        "--extract-sleep-seconds",
        type=float,
        default=float(os.getenv("DATASET_EXTRACT_SLEEP_SECONDS", "0")),
        help="Sleep briefly during extraction to reduce CPU pressure (default: 0).",
    )
    parser.add_argument(
        "--extract-yield-every",
        type=int,
        default=int(os.getenv("DATASET_EXTRACT_YIELD_EVERY", "500")),
        help="Apply extraction sleep after this many processed files (default: 500).",
    )
    parser.add_argument(
        "--extract-log-every",
        type=int,
        default=int(os.getenv("DATASET_EXTRACT_LOG_EVERY", "5000")),
        help="Log extraction progress after this many processed files (default: 5000).",
    )
    parser.add_argument(
        "--no-nice",
        action="store_true",
        help="Do not lower process CPU priority on platforms that support os.nice.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_nice and hasattr(os, "nice"):
        try:
            os.nice(10)
            logger.info("Lowered dataset downloader CPU priority with os.nice(10).")
        except OSError as exc:
            logger.warning("Could not lower downloader CPU priority: %s", exc)

    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    fraction = 1.0 if args.size == "full" else 0.5

    logger.info("Dataset download cwd: %s", Path.cwd())
    logger.info("Dataset download repo root: %s", _REPO_ROOT)
    logger.info("Dataset download BENCHMARK_ROOT env: %s", os.environ.get("BENCHMARK_ROOT", "<unset>"))
    logger.info("Dataset download benchmark root: %s", settings.benchmark_root.resolve())
    logger.info("Dataset download documents dir: %s", loader.documents_dir.resolve())
    logger.info("Dataset download bench dir: %s", loader.bench_dir.resolve())
    logger.info("Dataset download archive path: %s", (settings.benchmark_root / "all_documents.zip").resolve())
    logger.info("Dataset download mode: %s", args.size)

    loader.download_release_files(
        include_all_documents=True,
        document_fraction=fraction,
        extraction_log_every=args.extract_log_every,
        extraction_sleep_seconds=args.extract_sleep_seconds,
        extraction_yield_every=args.extract_yield_every,
    )
    logger.info("Downloaded %s dataset under %s.", args.size, settings.benchmark_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
