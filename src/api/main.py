from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import configure_store, router
from src.benchmark.loader import EnterpriseRagBenchLoader
from src.config import get_settings
from src.db import ExperimentStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_startup_paths() -> None:
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    questions_path = loader.resolved_questions_path()

    logger.info("API startup cwd: %s", Path.cwd())
    logger.info("API startup BENCHMARK_ROOT env: %s", os.environ.get("BENCHMARK_ROOT", "<unset>"))
    logger.info("API startup benchmark root: %s", settings.benchmark_root.resolve())
    logger.info("API startup documents dir: %s", loader.documents_dir.resolve())
    logger.info("API startup documents dir exists: %s", loader.documents_dir.exists())
    logger.info("API startup bench dir: %s", loader.bench_dir.resolve())
    logger.info("API startup questions path: %s", questions_path.resolve() if questions_path else "<missing>")
    logger.info("API startup QDRANT_URL env: %s", os.environ.get("QDRANT_URL", "<unset>"))
    logger.info("API startup qdrant url: %s", settings.qdrant_url or "<unset>")
    logger.info("API startup qdrant path fallback: %s", settings.qdrant_path.resolve())
    logger.info("API startup host repo is expected to be mounted at /app by docker-compose.yml")
    logger.info("API data status endpoint: GET /api/dataset/status")
    logger.info("API no-data behavior: service stays up and returns 'No data available' until data/index are ready")
    logger.info("API host download command, full dataset: python scripts/download_dataset.py full")
    logger.info("API host download command, half dataset: python scripts/download_dataset.py half")
    logger.info("API in-container index command: docker compose exec api python scripts/embed_dataset.py")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    log_startup_paths()
    store = ExperimentStore(settings.experiment_db_path)
    await store.init()
    configure_store(store)
    yield


app = FastAPI(title="AutoRAG Research Lab API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/up", include_in_schema=False)
async def up():
    return {"status": "ok"}


app.include_router(router, prefix="/api")
