# AutoResearch Lab

AutoResearch Lab is a multi-agent autonomous research system for improving enterprise Retrieval-Augmented Generation (RAG) pipelines through iterative, benchmark-driven experiments.

## Current Project State

The current implementation is functional end-to-end and includes:

- A LangGraph-based research loop with four core agents:
  - `Researcher`: proposes retrieval hypotheses.
  - `Planner`: mutates candidate retrieval configurations.
  - `Worker`: runs deterministic A/B benchmark experiments.
  - `Evaluator`: promotes candidates only when they beat the incumbent baseline.
- Retrieval backends:
  - Dense retrieval (sentence-transformers embeddings).
  - Hybrid retrieval (BM25 + dense fusion).
  - Optional cross-encoder reranking.
- Benchmarking stack for EnterpriseRAG-Bench style evaluation:
  - Dataset loading utilities.
  - Experiment runner.
  - Metric computation and composite scoring.
- Backend APIs for research orchestration and experiment inspection.
- Web dashboard/UI templates for monitoring runs and viewing experiment details.
- Local persistence for experiment metadata and outcomes.
- Docker-based local development support.

## High-Level Architecture

`UI / Dashboard` -> `API layer` -> `LangGraph agent loop` -> `Retrieval + Benchmark runner` -> `Experiment store`

The loop is monotonic by design: new configurations are promoted only when measured improvement is above threshold.

## Repository Layout

```text
src/
  agents/         # research, planning, execution, evaluation logic
  retrieval/      # dense/hybrid retrievers, embeddings, reranker
  benchmark/      # dataset loader, runner, metrics
  api/            # FastAPI app and routes
  dashboard/      # dashboard app and server config
  templates/      # HTML templates for web views
  static/         # CSS and assets
  db.py           # persistence layer
  config.py       # environment-driven settings
  models.py       # data models
scripts/          # dataset setup and utility scripts
Dockerfile
docker-compose.yml
pyproject.toml
requirements.txt
```

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- Google Gemini LLMs
- Google AI Studio
- LangGraph / LangChain Core
- Hugging Face Transformers
- Hugging Face Datasets
- sentence-transformers
- rank-bm25
- qdrant-client
- Flask (dashboard/auth session support)
- SQLite (experiment persistence)

## Goal

Continuously improve retrieval quality on enterprise-style RAG workloads while keeping results reproducible, measurable, and easy to inspect.
