#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark.loader import EnterpriseRagBenchLoader
from src.config import get_settings
from src.retrieval.embeddings import EmbeddingEncoder
from src.retrieval.dense import DenseRecord
from src.retrieval.qdrant_dense import QdrantDenseRetriever


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("embed_dataset")


def _manifest_path(qdrant_path: Path) -> Path:
    return qdrant_path.parent / "embedding_manifest.json"


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read embedding manifest: %s", path.resolve())
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _dense_records(loader: EnterpriseRagBenchLoader):
    for doc in loader.iter_documents():
        text = f"{doc.title}\n\n{doc.body}".strip()
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        yield DenseRecord(
            document_id=doc.document_id,
            text=text,
            metadata={
                "source_type": doc.source_type,
                **meta,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local Qdrant embedding index from benchmark data.")
    parser.add_argument("--force", action="store_true", help="Rebuild the index even if it already has all documents.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)

    logger.info("Startup embedding cwd: %s", Path.cwd())
    logger.info("Startup embedding repo root: %s", _REPO_ROOT)
    logger.info("Startup embedding BENCHMARK_ROOT env: %s", os.environ.get("BENCHMARK_ROOT", "<unset>"))
    logger.info("Startup embedding benchmark root: %s", settings.benchmark_root.resolve())
    logger.info("Startup embedding documents dir: %s", loader.documents_dir.resolve())
    logger.info("Startup embedding bench dir: %s", loader.bench_dir.resolve())
    logger.info("Startup embedding qdrant path: %s", settings.qdrant_path.resolve())
    logger.info("Startup embedding qdrant url: %s", settings.qdrant_url or "<unset>")
    logger.info("Startup embedding model: %s", settings.embedding_model)
    logger.info("Startup embedding manifest path: %s", _manifest_path(settings.qdrant_path).resolve())

    if not loader.benchmark_exists():
        logger.error(
            "Benchmark data missing under %s. Startup cannot continue. "
            "Download data first with one of: `python scripts/download_dataset.py full`, "
            "`python scripts/download_dataset.py half`, `python scripts/download_all_dataset.py`, "
            "or `python scripts/download_half_dataset.py`.",
            settings.benchmark_root.resolve(),
        )
        return 1

    expected_count = loader.document_count()
    if not expected_count:
        logger.error(
            "No documents found under %s. Startup cannot continue. "
            "Download data first with one of: `python scripts/download_dataset.py full`, "
            "`python scripts/download_dataset.py half`, `python scripts/download_all_dataset.py`, "
            "or `python scripts/download_half_dataset.py`.",
            loader.documents_dir.resolve(),
        )
        return 1

    logger.info("Found %s documents under %s", expected_count, loader.documents_dir.resolve())
    retriever = QdrantDenseRetriever(
        encoder=None,
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
    )

    indexed_count = retriever.collection_points_count()
    indexed_vector_size = retriever.collection_vector_size()
    manifest_path = _manifest_path(settings.qdrant_path)
    manifest = _load_manifest(manifest_path)
    logger.info("Existing Qdrant collection points: %s", indexed_count)
    logger.info("Existing Qdrant vector size: %s", indexed_vector_size)
    logger.info("Expected Qdrant points: %s", expected_count)
    logger.info("Existing embedding manifest: %s", manifest or "<missing>")
    if not args.force and indexed_count == expected_count and indexed_vector_size and not manifest:
        logger.warning(
            "Embedding index matches document count but has no manifest. "
            "Assuming it was built with the current embedding model and writing manifest to avoid a rebuild."
        )
        _save_manifest(
            manifest_path,
            {
                "document_count": indexed_count,
                "embedding_model": settings.embedding_model,
                "qdrant_path": str(settings.qdrant_path),
                "vector_size": indexed_vector_size,
            },
        )
        logger.info("Embedding index already contains %s documents at %s.", indexed_count, settings.qdrant_path.resolve())
        return 0

    if (
        not args.force
        and indexed_count == expected_count
        and manifest.get("embedding_model") == settings.embedding_model
        and manifest.get("document_count") == expected_count
        and manifest.get("vector_size") == indexed_vector_size
    ):
        logger.info("Embedding index already contains %s documents at %s.", indexed_count, settings.qdrant_path.resolve())
        return 0

    logger.info("Preparing embedding encoder.")
    encoder = EmbeddingEncoder(settings.embedding_model)
    expected_vector_size = len(encoder.encode(["dimension probe"])[0])
    retriever.encoder = encoder
    logger.info("Expected embedding vector size: %s", expected_vector_size)

    logger.info(
        "Embedding %s documents into %s. This can take a long time on CPU; progress logs print every 10,000 documents.",
        expected_count,
        settings.qdrant_path.resolve(),
    )
    indexed = retriever.build_streaming(_dense_records(loader), progress_interval=10_000)
    final_count = retriever.collection_points_count()
    if final_count != expected_count or indexed != expected_count:
        logger.error("Embedding index has %s documents, expected %s.", final_count, expected_count)
        return 1

    _save_manifest(
        manifest_path,
        {
            "document_count": final_count,
            "embedding_model": settings.embedding_model,
            "qdrant_path": str(settings.qdrant_path),
            "vector_size": expected_vector_size,
        },
    )
    logger.info("Saved embedding manifest to %s.", manifest_path.resolve())
    logger.info("Embedding index ready with %s documents at %s.", final_count, settings.qdrant_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
