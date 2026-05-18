from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.graph import build_research_graph, default_state
from src.agents.progress import reset_progress_reporter, set_progress_reporter
from src.agents.text_utils import extract_text_content
from src.benchmark.loader import EnterpriseRagBenchLoader
from src.benchmark.metrics import composite_score
from src.config import get_google_api_key, get_settings, set_runtime_google_api_key
from src.db import ExperimentStore
from src.models import ExperimentResult, ExperimentSpec, Hypothesis
from src.retrieval.embeddings import EmbeddingEncoder
from src.retrieval.qdrant_dense import QdrantDenseRetriever

logger = logging.getLogger(__name__)


@dataclass
class LabRuntime:
    running: bool = False
    state: dict = field(default_factory=dict)
    task: asyncio.Task | None = None
    last_error: str | None = None
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


runtime = LabRuntime(state={})
runtimes: dict[str, LabRuntime] = {}
latest_run_id: str | None = None
runtimes_lock = asyncio.Lock()
_dataset_readiness_cache: dict[str, Any] = {
    "indexed_count": 0,
    "qdrant_ready": False,
    "payload": None,
    "checked_at": 0.0,
}
MAX_RESEARCH_ITERATIONS = 10
DATASET_READINESS_STATUS_TTL_SECONDS = 30.0


async def _apply_runtime_api_key_from_header(
    x_google_api_key: str | None = Header(default=None),
) -> None:
    set_runtime_google_api_key((x_google_api_key or "").strip())


router = APIRouter(dependencies=[Depends(_apply_runtime_api_key_from_header)])
store: ExperimentStore | None = None


@lru_cache(maxsize=4)
def _chat_retriever(embedding_model: str, qdrant_path: str, qdrant_url: str) -> QdrantDenseRetriever:
    retriever = QdrantDenseRetriever(
        encoder=EmbeddingEncoder(embedding_model),
        qdrant_path=Path(qdrant_path),
        qdrant_url=qdrant_url,
    )
    retriever.load()
    return retriever


def _dataset_readiness() -> dict[str, Any]:
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    documents_dir = loader.documents_dir.resolve()
    bench_dir = loader.bench_dir.resolve()
    document_count = loader.document_count()
    has_questions = loader.resolved_questions_path() is not None
    has_documents = document_count > 0

    indexed_count = 0
    qdrant_ready = False
    qdrant_busy = False
    qdrant_error = ""
    qdrant_location = settings.qdrant_url or str(settings.qdrant_path.resolve())
    try:
        retriever = QdrantDenseRetriever(
            encoder=None,
            qdrant_path=settings.qdrant_path,
            qdrant_url=settings.qdrant_url,
        )
        indexed_count = retriever.collection_points_count()
        qdrant_ready = indexed_count > 0
        _dataset_readiness_cache["indexed_count"] = indexed_count
        _dataset_readiness_cache["qdrant_ready"] = qdrant_ready
    except Exception as exc:
        qdrant_error = str(exc)
        if "already accessed by another instance of Qdrant client" in qdrant_error:
            qdrant_busy = True
            indexed_count = int(_dataset_readiness_cache.get("indexed_count") or 0)
            # Local Qdrant lock contention means another in-process run is using
            # the index; avoid surfacing this as "no data".
            qdrant_ready = bool(_dataset_readiness_cache.get("qdrant_ready")) or (
                has_documents and has_questions
            )

    ready = has_documents and has_questions and (qdrant_ready or qdrant_busy)
    if ready:
        if qdrant_busy:
            message = "Data is available. The embedding index is currently in use by an active run."
        else:
            message = "Data and embedding index are available."
    elif not has_documents:
        message = (
            "No data available. Download data on the host with `python scripts/download_dataset.py full` "
            "or `python scripts/download_dataset.py half`."
        )
    elif not has_questions:
        message = "Documents exist, but benchmark questions are missing. Run `python scripts/download_dataset.py full` or `half`."
    else:
        message = "Documents exist, but the embedding index is not ready. Run `docker compose exec api python scripts/embed_dataset.py`."

    payload = {
        "ready": ready,
        "has_documents": has_documents,
        "has_questions": has_questions,
        "documents": document_count,
        "questions": "available" if has_questions else "missing",
        "sampled_questions": "—",
        "holdout_questions": "—",
        "document_count": document_count,
        "indexed_count": indexed_count,
        "qdrant_ready": qdrant_ready,
        "qdrant_busy": qdrant_busy,
        "benchmark_root": str(settings.benchmark_root.resolve()),
        "documents_dir": str(documents_dir),
        "bench_dir": str(bench_dir),
        "qdrant_location": qdrant_location,
        "qdrant_error": qdrant_error,
        "message": message,
        "download_full_command": "python scripts/download_dataset.py full",
        "download_half_command": "python scripts/download_dataset.py half",
        "embed_command": "docker compose exec api python scripts/embed_dataset.py",
    }
    _dataset_readiness_cache["payload"] = payload
    _dataset_readiness_cache["checked_at"] = time.time()
    return payload


def _cached_dataset_readiness() -> dict[str, Any]:
    payload = _dataset_readiness_cache.get("payload")
    if isinstance(payload, dict):
        cached = dict(payload)
        cached["cached"] = True
        cached["qdrant_busy"] = bool(cached.get("qdrant_busy")) or bool(cached.get("qdrant_ready"))
        if cached.get("ready"):
            cached["message"] = "Data and embedding index are available. Status is cached while research is running."
        return cached

    settings = get_settings()
    qdrant_ready = bool(_dataset_readiness_cache.get("qdrant_ready"))
    indexed_count = int(_dataset_readiness_cache.get("indexed_count") or 0)
    ready = qdrant_ready or indexed_count > 0
    message = (
        "Research is running. Dataset readiness has not been refreshed yet, but the active run is using the configured benchmark data."
        if ready
        else "Research is running. Live dataset readiness probe is skipped to keep status responsive."
    )
    return {
        "ready": ready,
        "has_documents": ready,
        "has_questions": ready,
        "documents": indexed_count if indexed_count > 0 else None,
        "questions": "unknown",
        "sampled_questions": "—",
        "holdout_questions": "—",
        "document_count": indexed_count if indexed_count > 0 else None,
        "indexed_count": indexed_count,
        "qdrant_ready": qdrant_ready,
        "qdrant_busy": True,
        "benchmark_root": str(settings.benchmark_root.resolve()),
        "documents_dir": "",
        "bench_dir": "",
        "qdrant_location": settings.qdrant_url or str(settings.qdrant_path.resolve()),
        "qdrant_error": "Skipped live readiness probe while research is running.",
        "message": message,
        "download_full_command": "python scripts/download_dataset.py full",
        "download_half_command": "python scripts/download_dataset.py half",
        "embed_command": "docker compose exec api python scripts/embed_dataset.py",
        "cached": True,
    }


def _status_dataset_readiness(run_runtime: LabRuntime) -> dict[str, Any]:
    if run_runtime.running:
        return _cached_dataset_readiness()
    cached_at = float(_dataset_readiness_cache.get("checked_at") or 0.0)
    if _dataset_readiness_cache.get("payload") and time.time() - cached_at < DATASET_READINESS_STATUS_TTL_SECONDS:
        return _cached_dataset_readiness()
    return _dataset_readiness()


@router.get("/dataset/status")
async def dataset_status():
    return _dataset_readiness()


@lru_cache(maxsize=16)
def _chat_llm(model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
        request_timeout=30,
        retries=2,
    )


@router.post("/settings/api-key")
async def set_api_key(payload: dict[str, Any]):
    api_key = str(payload.get("api_key") or "").strip()
    set_runtime_google_api_key(api_key)
    return {
        "status": "ok",
        "message": "Runtime API keys are request-scoped. Send X-Google-Api-Key per request.",
    }


@router.get("/settings/api-key/status")
async def api_key_status():
    settings = get_settings()
    return {
        "has_google_key": bool(settings.has_google_key),
    }


def _candidate_chat_models() -> list[str]:
    settings = get_settings()
    ordered = [
        settings.gemini_fast_model,
        settings.gemini_model,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.5-pro",
    ]
    seen: set[str] = set()
    models = []
    for model in ordered:
        if model and model.startswith("gemini-3"):
            continue
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _format_chat_history(history: list[dict[str, Any]], limit: int = 8) -> str:
    lines = []
    for message in history[-limit:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:1200]}")
    return "\n\n".join(lines)


def configure_store(experiment_store: ExperimentStore) -> None:
    global store
    store = experiment_store


async def _persist_state(state: dict) -> None:
    """Persist hypotheses, experiments and results from current state to DB."""
    assert store is not None
    for hypothesis in state.get("hypotheses", []):
        try:
            await store.upsert_hypothesis(Hypothesis.model_validate(hypothesis))
        except Exception:
            pass
    for experiment in state.get("completed_experiments", []):
        try:
            await store.upsert_experiment(ExperimentSpec.model_validate(experiment))
        except Exception:
            pass
    for result in state.get("results", []):
        try:
            await store.upsert_result(ExperimentResult.model_validate(result))
        except Exception:
            pass


async def _run_research_loop(run_id: str) -> None:
    assert store is not None
    graph = build_research_graph()
    run_runtime = runtimes[run_id]

    def report_runtime_progress(
        phase: str,
        summary: str | None,
        detail: dict[str, Any] | None,
    ) -> None:
        run_runtime.state["current_phase"] = phase
        if summary is not None:
            run_runtime.state["latest_summary"] = summary
        if detail:
            progress_detail = dict(run_runtime.state.get("progress_detail") or {})
            progress_detail.update(detail)
            progress_detail["phase"] = phase
            progress_detail["updated_at"] = time.time()
            run_runtime.state["progress_detail"] = progress_detail

    progress_token = set_progress_reporter(report_runtime_progress)
    try:
        async for event in graph.astream(run_runtime.state, stream_mode="values"):
            run_runtime.state = dict(event)
            await _persist_state(run_runtime.state)

        run_runtime.last_error = None
    except asyncio.CancelledError:
        run_runtime.last_error = None
        run_runtime.state["should_stop"] = True
        run_runtime.state["current_phase"] = "stopped"
        run_runtime.state["latest_summary"] = "Research run stopped by user."
    except Exception as exc:
        run_runtime.last_error = str(exc)
        run_runtime.state["current_phase"] = "failed"
        run_runtime.state["latest_summary"] = f"Research loop failed: {exc}"
    finally:
        reset_progress_reporter(progress_token)
        run_runtime.running = False
        run_runtime.task = None


def _select_runtime(run_id: str | None = None) -> LabRuntime:
    selected_run_id = run_id or latest_run_id
    if selected_run_id and selected_run_id in runtimes:
        return runtimes[selected_run_id]
    return runtime


def _empty_status(run_id: str | None = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "running": False,
        "phase": None,
        "iteration": 0,
        "max_iterations": 0,
        "latest_summary": "No research run has started yet.",
        "best_score": None,
        "initial_baseline_score": None,
        "accepted_experiments": 0,
        "rejected_experiments": 0,
        "per_type_summary": None,
        "planner_rationale": None,
        "candidate_config": None,
        "baseline_config": None,
        "final_report": None,
        "score_history": [],
        "failure_examples": [],
        "question_focus": None,
        "research_setup_id": None,
        "failure_taxonomy": {},
        "recommendation": None,
        "validation_summary": None,
        "dataset_readiness": {},
        "tried_config_count": 0,
        "rejected_config_count": 0,
        "last_error": "Run not found." if run_id else None,
        "research_mode": "config",
        "karpathy_branch": "",
        "initial_pipeline_code": "",
        "current_pipeline_code": "",
        "proposed_code": "",
        "code_history": [],
        "progress_detail": {},
    }


def _status_summary_payload(run_runtime: LabRuntime) -> dict[str, Any]:
    best_score = run_runtime.state.get("best_score")
    best_score = best_score if isinstance(best_score, (int, float)) and best_score >= 0 else None
    return {
        "run_id": run_runtime.state.get("run_id"),
        "running": run_runtime.running,
        "phase": run_runtime.state.get("current_phase"),
        "iteration": run_runtime.state.get("iteration"),
        "max_iterations": run_runtime.state.get("max_iterations"),
        "latest_summary": extract_text_content(run_runtime.state.get("latest_summary")),
        "best_score": best_score,
        "accepted_experiments": run_runtime.state.get("accepted_experiments"),
        "rejected_experiments": run_runtime.state.get("rejected_experiments"),
        "final_report": extract_text_content(run_runtime.state.get("final_report")),
        "last_error": run_runtime.last_error,
        "research_mode": run_runtime.state.get("research_mode", "config"),
        "progress_detail": run_runtime.state.get("progress_detail", {}),
    }


def _status_payload(run_runtime: LabRuntime) -> dict[str, Any]:
    started_at = time.perf_counter()
    timings: list[tuple[str, float]] = []

    def mark(label: str, step_started_at: float) -> float:
        now = time.perf_counter()
        timings.append((label, (now - step_started_at) * 1000))
        return now

    step_started_at = started_at
    candidate = run_runtime.state.get("candidate_config")
    baseline_cfg = run_runtime.state.get("best_config")
    best_score = run_runtime.state.get("best_score")
    initial_baseline_score = run_runtime.state.get("initial_baseline_score")
    best_score = best_score if isinstance(best_score, (int, float)) and best_score >= 0 else None
    initial_baseline_score = (
        initial_baseline_score
        if isinstance(initial_baseline_score, (int, float)) and initial_baseline_score >= 0
        else None
    )
    initial_cfg = None
    if run_runtime.state.get("completed_experiments"):
        first_exp = run_runtime.state["completed_experiments"][0]
        if hasattr(first_exp, "retrieval_config"):
            initial_cfg = first_exp.retrieval_config.model_dump()
        elif isinstance(first_exp, dict):
            initial_cfg = first_exp.get("retrieval_config")
    if initial_cfg is None and baseline_cfg:
        initial_cfg = baseline_cfg.model_dump() if hasattr(baseline_cfg, "model_dump") else baseline_cfg
    step_started_at = mark("configs", step_started_at)
    score_history = _normalize_history_text(run_runtime.state.get("score_history"))
    code_history = _normalize_history_text(run_runtime.state.get("code_history"))
    experiment_hypothesis_ids: dict[str, str] = {}
    for experiment in run_runtime.state.get("completed_experiments", []) or []:
        if hasattr(experiment, "id") and hasattr(experiment, "hypothesis_id"):
            experiment_hypothesis_ids[str(experiment.id)] = str(experiment.hypothesis_id)
        elif isinstance(experiment, dict) and experiment.get("id"):
            experiment_hypothesis_ids[str(experiment["id"])] = str(experiment.get("hypothesis_id") or "")
    for history in (score_history, code_history):
        for row in history:
            experiment_id = str(row.get("experiment_id") or "")
            if experiment_id and not row.get("hypothesis_id"):
                row["hypothesis_id"] = experiment_hypothesis_ids.get(experiment_id, "")
    step_started_at = mark("history", step_started_at)
    dataset_readiness = _status_dataset_readiness(run_runtime)
    step_started_at = mark("dataset_readiness", step_started_at)
    payload = {
        "run_id": run_runtime.state.get("run_id"),
        "running": run_runtime.running,
        "phase": run_runtime.state.get("current_phase"),
        "iteration": run_runtime.state.get("iteration"),
        "max_iterations": run_runtime.state.get("max_iterations"),
        "latest_summary": extract_text_content(run_runtime.state.get("latest_summary")),
        "best_score": best_score,
        "initial_baseline_score": initial_baseline_score,
        "accepted_experiments": run_runtime.state.get("accepted_experiments"),
        "rejected_experiments": run_runtime.state.get("rejected_experiments"),
        "per_type_summary": run_runtime.state.get("per_type_summary"),
        "planner_rationale": extract_text_content(run_runtime.state.get("planner_rationale")),
        "candidate_config": candidate.model_dump() if candidate else None,
        "baseline_config": initial_cfg,
        "final_report": extract_text_content(run_runtime.state.get("final_report")),
        "score_history": score_history,
        "failure_examples": run_runtime.state.get("failure_examples", []),
        "question_focus": run_runtime.state.get("question_focus"),
        "research_setup_id": run_runtime.state.get("research_setup_id"),
        "failure_taxonomy": run_runtime.state.get("failure_taxonomy"),
        "recommendation": extract_text_content(run_runtime.state.get("recommendation")),
        "validation_summary": extract_text_content(run_runtime.state.get("validation_summary")),
        "dataset_readiness": dataset_readiness,
        "tried_config_count": len(run_runtime.state.get("tried_config_fingerprints", []) or []),
        "rejected_config_count": len(run_runtime.state.get("rejected_config_fingerprints", []) or []),
        "last_error": run_runtime.last_error,
        "progress_detail": run_runtime.state.get("progress_detail", {}),
        # Karpathy mode fields
        "research_mode": run_runtime.state.get("research_mode", "config"),
        "karpathy_branch": run_runtime.state.get("karpathy_branch", ""),
        "initial_pipeline_code": run_runtime.state.get("initial_pipeline_code", ""),
        "current_pipeline_code": run_runtime.state.get("current_pipeline_code", ""),
        "proposed_code": run_runtime.state.get("proposed_code", ""),
        "code_history": code_history,
    }
    mark("payload", step_started_at)
    total_ms = (time.perf_counter() - started_at) * 1000
    if total_ms > 500:
        logger.warning(
            "research/status full payload slow run_id=%s total_ms=%.1f timings=%s",
            run_runtime.state.get("run_id"),
            total_ms,
            ", ".join(f"{label}={ms:.1f}ms" for label, ms in timings),
        )
    else:
        logger.debug(
            "research/status full payload run_id=%s total_ms=%.1f",
            run_runtime.state.get("run_id"),
            total_ms,
        )
    return payload


def _normalize_history_text(history: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(history, list):
        return rows
    for item in history:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        for key in ("hypothesis", "rationale", "reason", "diff_summary"):
            if key in normalized:
                normalized[key] = extract_text_content(normalized.get(key))
        rows.append(normalized)
    return rows


@router.post("/research/start")
async def start_research(
    max_iterations: int = 3,
    question_focus: str = "all",
    benchmark_root: str | None = None,
    research_setup_id: str | None = None,
    starting_config_json: str | None = None,
    research_mode: str = "config",
):
    global latest_run_id
    if store is None:
        raise HTTPException(status_code=500, detail="Experiment store is not configured.")
    if research_mode not in ("config", "karpathy"):
        raise HTTPException(status_code=400, detail="research_mode must be 'config' or 'karpathy'.")
    max_iterations = max(1, min(MAX_RESEARCH_ITERATIONS, int(max_iterations)))

    starting_config = None
    if starting_config_json:
        try:
            starting_config = json.loads(starting_config_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid starting_config_json.") from exc

    try:
        _dataset_readiness()
    except Exception as exc:
        logger.warning("Dataset readiness preflight failed before run start: %s", exc)

    state = default_state(
        max_iterations=max_iterations,
        starting_config=starting_config,
        question_focus=question_focus,
        benchmark_root=benchmark_root,
        research_setup_id=research_setup_id,
        research_mode=research_mode,
    )
    run_id = state["run_id"]
    run_runtime = LabRuntime(running=True, state=state)

    async with runtimes_lock:
        runtimes[run_id] = run_runtime
        latest_run_id = run_id

    run_runtime.task = asyncio.create_task(_run_research_loop(run_id))
    return {
        "status": "started",
        "run_id": run_id,
        "max_iterations": max_iterations,
        "research_mode": research_mode,
    }


@router.post("/research/stop")
async def stop_research(run_id: str | None = None):
    run_runtime = _select_runtime(run_id)
    if not run_runtime.state:
        raise HTTPException(status_code=404, detail="Research run not found.")

    run_runtime.state["should_stop"] = True
    run_runtime.state["current_phase"] = "stopped"
    run_runtime.state["latest_summary"] = "Research run stopped by user."
    run_runtime.last_error = None

    if run_runtime.task and not run_runtime.task.done():
        run_runtime.task.cancel()
    run_runtime.running = False

    return {"status": "stopped", "run_id": run_runtime.state.get("run_id")}


@router.get("/research/status")
async def research_status(run_id: str | None = None, detail: str = "full"):
    started_at = time.perf_counter()
    run_runtime = _select_runtime(run_id)
    if not run_runtime.state:
        return _empty_status(run_id)
    if detail == "summary":
        payload = _status_summary_payload(run_runtime)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > 250:
            logger.warning(
                "research/status summary slow run_id=%s total_ms=%.1f",
                run_runtime.state.get("run_id"),
                elapsed_ms,
            )
        return payload
    return _status_payload(run_runtime)


@router.post("/rag/chat")
async def rag_chat(payload: dict[str, Any]):
    settings = get_settings()
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    top_k = payload.get("top_k")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 4
    top_k = max(1, min(8, top_k))
    history = payload.get("history") if isinstance(payload.get("history"), list) else []

    started_at = time.perf_counter()
    readiness = _dataset_readiness()
    if not readiness["ready"]:
        return {
            "answer": f"No data available yet. {readiness['message']}",
            "documents": [],
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "model_label": "not_available",
            "dataset_readiness": readiness,
        }

    if not settings.has_google_key:
        raise HTTPException(
            status_code=400,
            detail="Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY.",
        )

    try:
        retriever = _chat_retriever(settings.embedding_model, str(settings.qdrant_path), settings.qdrant_url)
        docs = retriever.retrieve(question, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    context = "\n\n---\n\n".join(
        f"Source {index}\nDocument ID: {doc.document_id}\nSource type: {doc.metadata.get('source_type') or 'unknown'}\nScore: {doc.score:.4f}\n{doc.text[:1600]}"
        for index, doc in enumerate(docs, start=1)
    )
    prompt = (
        "You are Monotonic Labs AI Chat, a RAG assistant for enterprise retrieval testing.\n"
        "Answer using ONLY the retrieved context. If the context is insufficient, say what is missing.\n"
        "Be concise, cite document IDs inline when useful, and keep the answer practical.\n\n"
        f"Recent chat history:\n{_format_chat_history(history) or 'No prior messages.'}\n\n"
        f"User question:\n{question}\n\n"
        f"Retrieved context:\n{context or 'No documents retrieved.'}\n\n"
        "Answer:"
    )

    response = None
    model_name = ""
    errors: list[str] = []
    api_key = get_google_api_key()
    for candidate_model in _candidate_chat_models():
        model_name = candidate_model
        try:
            response = _chat_llm(candidate_model, api_key).invoke([HumanMessage(content=prompt)])
            break
        except Exception as exc:
            errors.append(f"{candidate_model}: {str(exc)[:220]}")

    if response is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini request failed for all candidate models. Check that your Google API key has access "
                "to a supported Gemini model, or set GEMINI_FAST_MODEL=gemini-2.0-flash. "
                + " | ".join(errors)
            ),
        )

    answer = str(response.content or "").strip()
    if not answer:
        answer = "Gemini returned an empty response. Try asking a more specific question or lowering top-k."
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    return {
        "answer": answer,
        "documents": [
            {
                "id": doc.document_id,
                "score": doc.score,
                "source_type": doc.metadata.get("source_type") or "unknown",
                "preview": doc.text[:420].strip(),
            }
            for doc in docs
        ],
        "latency_ms": elapsed_ms,
        "model_label": model_name,
    }


@router.get("/experiments")
async def list_experiments():
    if store is None:
        raise HTTPException(status_code=500, detail="Experiment store is not configured.")
    experiments = await store.list_experiments()
    return [item.model_dump() for item in experiments]


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    if store is None:
        raise HTTPException(status_code=500, detail="Experiment store is not configured.")
    result = await store.get_result(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment result not found.")
    return result.model_dump()


@router.get("/leaderboard")
async def leaderboard():
    if store is None:
        raise HTTPException(status_code=500, detail="Experiment store is not configured.")
    results = await store.list_results()
    board = [
        {
            "experiment_id": result.experiment_id,
            "score": result.composite_score or composite_score(result.metrics),
            "baseline_score": result.baseline_score,
            "delta_vs_baseline": result.delta_vs_baseline,
            "accepted": result.accepted,
            "metrics": result.metrics.model_dump(),
        }
        for result in results
    ]
    board.sort(key=lambda x: x["score"], reverse=True)
    return board


@router.get("/hypotheses")
async def hypotheses():
    if store is None:
        raise HTTPException(status_code=500, detail="Experiment store is not configured.")
    items = await store.list_hypotheses()
    return [item.model_dump() for item in items]


@router.post("/research/commit-pipeline")
async def commit_pipeline_route(message: str = "Manual commit from dashboard", run_id: str | None = None):
    raise HTTPException(
        status_code=403,
        detail=(
            "Pipeline commits are disabled in the demo sandbox environment. "
            "Karpathy changes are accepted only into the active run session."
        ),
    )
