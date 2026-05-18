"""Karpathy Worker -- validates code and evaluates it in a session sandbox."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import re
import sys
import tarfile
import tempfile
from pathlib import Path

from src.agents.karpathy_validation import validate_karpathy_code
from src.agents.state import ResearchLabState
from src.benchmark.karpathy_sandbox import run_karpathy_benchmark
from src.config import get_settings
from src.models import (
    BenchmarkMetrics,
    ExperimentResult,
    ExperimentStatus,
    QuestionResult,
    RetrievalConfig,
)

logger = logging.getLogger(__name__)
RESULT_PREFIX = "__KARPATHY_RESULT__"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_HF_HOME = "/app/.hf_cache"
SANDBOX_DATA_ROOT = "/app/data"
SANDBOX_QDRANT_PATH = "/app/index/qdrant_data"


def _validate_code(code: str) -> str | None:
    """Run all validation checks. Returns error message or None on success."""
    return validate_karpathy_code(code)


def _load_retrieve_function_from_code(code: str):
    """Load retrieve() from a temp module without mutating src/retrieval/pipeline.py."""
    with tempfile.NamedTemporaryFile("w", suffix="_pipeline.py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        temp_path = Path(fh.name)
    module_name = f"_karpathy_pipeline_{abs(hash(temp_path))}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to create import spec for candidate pipeline.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.retrieve
    finally:
        temp_path.unlink(missing_ok=True)


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer.getvalue()


def _tar_directory_bytes(source: Path, arcname: str = ".") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(source, arcname=arcname)
    buffer.seek(0)
    return buffer.getvalue()


def _sandbox_payload(state: ResearchLabState) -> dict:
    settings = get_settings()
    config = state.get("proposed_config") or state.get("best_config")
    if isinstance(config, RetrievalConfig):
        config_payload = config.model_dump()
    elif config:
        config_payload = config
    else:
        config_payload = None

    return {
        "run_id": state.get("run_id"),
        "question_focus": state.get("question_focus") or "all",
        "benchmark_root": state.get("benchmark_root") or str(settings.benchmark_root),
        "best_config": config_payload,
    }


def _copy_uploaded_benchmark_if_needed(container, state: ResearchLabState, payload: dict) -> None:
    explicit_benchmark_root = state.get("benchmark_root")
    benchmark_root = explicit_benchmark_root or payload.get("benchmark_root")
    if not benchmark_root:
        return

    source = Path(benchmark_root)
    if explicit_benchmark_root:
        source = source.resolve()
    elif not source.is_absolute():
        return
    if not source.exists() or not source.is_dir():
        return

    archive = _tar_directory_bytes(source, arcname="benchmark_root")
    container.put_archive("/tmp", archive)
    payload["benchmark_root"] = "/tmp/benchmark_root"


def _sandbox_data_path_for(benchmark_root: str | None) -> str:
    if not benchmark_root:
        return SANDBOX_DATA_ROOT

    path = Path(benchmark_root)
    parts = path.parts
    if "data" not in parts:
        return SANDBOX_DATA_ROOT

    data_index = parts.index("data")
    suffix_parts = parts[data_index + 1:]
    if not suffix_parts:
        return SANDBOX_DATA_ROOT
    return str(Path(SANDBOX_DATA_ROOT, *suffix_parts))


def _copy_default_benchmark_if_needed(container, data_mounted: bool, payload: dict) -> None:
    """Fallback for non-Compose runs: copy project data into the sandbox."""
    if data_mounted:
        return

    source = PROJECT_ROOT / "data"
    if not source.exists() or not source.is_dir():
        logger.info("Karpathy sandbox: no data directory found at %s", source)
        return

    logger.info("Karpathy sandbox: copying data directory into sandbox container")
    container.put_archive("/app", _tar_directory_bytes(source, arcname="data"))
    payload["benchmark_root"] = SANDBOX_DATA_ROOT


def _copy_hf_cache_if_needed(container, cache_mounted: bool) -> None:
    """Fallback for non-Compose runs: copy the project HF cache into the sandbox."""
    if cache_mounted:
        return

    source = PROJECT_ROOT / ".hf_cache"
    if not source.exists() or not source.is_dir():
        logger.info("Karpathy sandbox: no .hf_cache found at %s", source)
        return

    logger.info("Karpathy sandbox: copying .hf_cache into sandbox container")
    container.put_archive("/app", _tar_directory_bytes(source, arcname=".hf_cache"))


def _model_thread_env(thread_count: int) -> dict[str, str]:
    threads = str(max(1, thread_count))
    return {
        "MODEL_INFERENCE_THREADS": threads,
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "TOKENIZERS_PARALLELISM": "false",
    }


def _sandbox_thread_env(thread_count: int) -> dict[str, str]:
    return _model_thread_env(thread_count)


def _clean_log_excerpt(text: str, limit: int = 1200) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    excerpt = " | ".join(lines)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    return excerpt[-limit:] if len(excerpt) > limit else excerpt


def _container_logs_tail(container, limit: int = 2000) -> str:
    try:
        logs = container.logs(stdout=True, stderr=True, tail=80)
        return _clean_log_excerpt(logs.decode("utf-8", errors="replace"), limit=limit)
    except Exception as exc:
        return f"<unable to read sandbox logs: {exc}>"


def _is_docker_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "docker sdk is not installed" in text:
        return True
    if "error while fetching server api version" in text:
        return True
    if "docker daemon" in text and "permission denied" in text:
        return True

    socket_markers = (
        "no such file or directory",
        "connection aborted",
        "permission denied",
    )
    return "docker sandbox failed" in text and any(marker in text for marker in socket_markers)


def _run_benchmark_in_docker(
    state: ResearchLabState,
    pipeline_code: str,
) -> tuple[list[QuestionResult], BenchmarkMetrics]:
    settings = get_settings()
    try:
        import docker
        from docker.errors import DockerException, ImageNotFound
    except Exception as exc:  # pragma: no cover - depends on deployed env
        raise RuntimeError("Docker SDK is not installed. Install the 'docker' package.") from exc

    payload = _sandbox_payload(state)
    client = None
    container = None
    volumes = {}
    hf_cache_host_path = settings.karpathy_hf_cache_host_path.strip()
    if hf_cache_host_path:
        # Hugging Face writes small `.no_exist` sentinel files even in offline
        # mode. Keep the shared model cache mounted, but make it writable so
        # those harmless probes do not spam logs or fail on read-only mounts.
        volumes[hf_cache_host_path] = {"bind": SANDBOX_HF_HOME, "mode": "rw"}
    data_host_path = settings.karpathy_data_host_path.strip()
    if data_host_path:
        volumes[data_host_path] = {"bind": SANDBOX_DATA_ROOT, "mode": "ro"}
        payload["benchmark_root"] = _sandbox_data_path_for(state.get("benchmark_root"))
    qdrant_host_path = settings.karpathy_qdrant_host_path.strip()
    if qdrant_host_path:
        # Local Qdrant creates lock files, so this cannot be read-only.
        volumes[qdrant_host_path] = {"bind": SANDBOX_QDRANT_PATH, "mode": "rw"}

    environment = {
        "PYTHONUNBUFFERED": "1",
        "EMBEDDING_MODEL": settings.embedding_model,
        "QDRANT_PATH": SANDBOX_QDRANT_PATH if qdrant_host_path else "/tmp/qdrant_data",
        "HF_HOME": SANDBOX_HF_HOME,
        "HF_HUB_CACHE": f"{SANDBOX_HF_HOME}/hub",
        "TRANSFORMERS_CACHE": SANDBOX_HF_HOME,
    }
    environment.update(_model_thread_env(settings.karpathy_sandbox_threads))
    if settings.karpathy_sandbox_network_disabled:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"

    try:
        client = docker.from_env()
        try:
            create_kwargs = {
                "image": settings.karpathy_sandbox_image,
                "command": [
                    "python",
                    "-m",
                    "src.benchmark.karpathy_sandbox_runner",
                    "/tmp/karpathy_state.json",
                ],
                "working_dir": "/app",
                "environment": environment,
                "mem_limit": settings.karpathy_sandbox_memory,
                "network_disabled": settings.karpathy_sandbox_network_disabled,
                "volumes": volumes or None,
            }
            if settings.karpathy_sandbox_cpus > 0:
                create_kwargs["nano_cpus"] = int(settings.karpathy_sandbox_cpus * 1_000_000_000)
            container = client.containers.create(**create_kwargs)
        except ImageNotFound as exc:
            raise RuntimeError(
                f"Sandbox image '{settings.karpathy_sandbox_image}' was not found. "
                "Build it with docker compose build or set KARPATHY_SANDBOX_IMAGE."
            ) from exc

        _copy_hf_cache_if_needed(container, cache_mounted=bool(hf_cache_host_path))
        data_mounted = bool(data_host_path)
        _copy_default_benchmark_if_needed(container, data_mounted=data_mounted, payload=payload)
        if not data_mounted:
            _copy_uploaded_benchmark_if_needed(container, state, payload)
        container.put_archive(
            "/app/src/retrieval",
            _tar_bytes({"pipeline.py": pipeline_code.encode("utf-8")}),
        )
        container.put_archive(
            "/tmp",
            _tar_bytes({"karpathy_state.json": json.dumps(payload).encode("utf-8")}),
        )

        container.start()
        try:
            exit_result = container.wait(timeout=settings.karpathy_sandbox_timeout_seconds)
        except Exception as exc:
            logs = _container_logs_tail(container)
            raise RuntimeError(
                "Sandbox timed out after "
                f"{settings.karpathy_sandbox_timeout_seconds}s and was stopped. "
                "The candidate may be too slow or may call retrieve too many times. "
                f"Last logs: {logs or 'none'}"
            ) from exc
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        clean_logs = _clean_log_excerpt(logs)
        status_code = exit_result.get("StatusCode", 1) if isinstance(exit_result, dict) else 1
        if status_code != 0:
            if status_code == 137:
                raise RuntimeError(
                    "Sandbox exited with 137 (killed by Docker, commonly timeout or "
                    f"memory pressure). Current KARPATHY_SANDBOX_MEMORY="
                    f"{settings.karpathy_sandbox_memory}. Last logs: {clean_logs or 'none'}"
                )
            raise RuntimeError(f"Sandbox exited with {status_code}: {clean_logs or 'no logs'}")

        result_line = next(
            (line[len(RESULT_PREFIX):] for line in reversed(logs.splitlines()) if line.startswith(RESULT_PREFIX)),
            "",
        )
        if not result_line:
            raise RuntimeError(f"Sandbox did not emit a result payload: {clean_logs or 'no logs'}")
        result_payload = json.loads(result_line)
        question_results = [
            QuestionResult.model_validate(item)
            for item in result_payload.get("question_results", [])
        ]
        metrics = BenchmarkMetrics.model_validate(result_payload.get("metrics") or {})
        return question_results, metrics
    except DockerException as exc:
        raise RuntimeError(f"Docker sandbox failed: {exc}") from exc
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                logger.warning("Failed to remove Karpathy sandbox container", exc_info=True)


def _run_benchmark_isolated_in_process(
    state: ResearchLabState,
    pipeline_code: str,
) -> tuple[list[QuestionResult], BenchmarkMetrics]:
    retrieve_fn = _load_retrieve_function_from_code(pipeline_code)
    config = state.get("proposed_config") or state.get("best_config")
    return run_karpathy_benchmark(
        retrieve_fn,
        benchmark_root=state.get("benchmark_root"),
        question_focus=state.get("question_focus") or "all",
        config=config if isinstance(config, RetrievalConfig) else None,
    )


def _run_benchmark_for_code(
    state: ResearchLabState,
    pipeline_code: str,
) -> tuple[list[QuestionResult], BenchmarkMetrics]:
    settings = get_settings()
    if settings.karpathy_sandbox_enabled:
        try:
            return _run_benchmark_in_docker(state, pipeline_code)
        except RuntimeError as exc:
            if (
                settings.karpathy_sandbox_fallback_to_process
                and _is_docker_unavailable_error(exc)
            ):
                logger.warning(
                    "Docker sandbox unavailable; falling back to in-process Karpathy "
                    "benchmark: %s",
                    exc,
                )
                return _run_benchmark_isolated_in_process(state, pipeline_code)
            raise
    return _run_benchmark_isolated_in_process(state, pipeline_code)


def _failed_result(
    state: ResearchLabState,
    experiment_id: str,
    reason: str,
) -> ExperimentResult:
    best_score = state.get("best_score", -1.0)
    baseline = best_score if best_score >= 0 else None
    return ExperimentResult(
        experiment_id=experiment_id,
        status=ExperimentStatus.failed,
        metrics=BenchmarkMetrics(),
        composite_score=0.0,
        baseline_score=baseline,
        delta_vs_baseline=-baseline if baseline is not None and baseline > 0 else None,
        failure_analysis=reason,
    )


async def karpathy_worker_agent(state: ResearchLabState) -> ResearchLabState:
    """Validate session code, run it in a sandbox, and produce ExperimentResult."""
    state["current_phase"] = "karpathy_worker"

    if not state["experiment_queue"]:
        return state

    experiment = state["experiment_queue"][0]
    proposed_code = state.get("proposed_code", "")

    if not proposed_code:
        state["experiment_queue"] = state["experiment_queue"][1:]
        state["latest_summary"] = "No proposed code to evaluate."
        return state

    # Step 1: Static validation
    validation_error = _validate_code(proposed_code)
    if validation_error:
        logger.warning("Code validation failed: %s", validation_error)
        result = _failed_result(state, experiment.id, f"Validation failed: {validation_error}")
        state["results"] = state["results"] + [result]
        state["completed_experiments"] = state["completed_experiments"] + [experiment]
        state["experiment_queue"] = state["experiment_queue"][1:]
        state["latest_summary"] = f"Code validation failed: {validation_error}"
        return state

    # Step 2: Run benchmark with the proposed session pipeline.
    try:
        question_results, metrics = await asyncio.to_thread(
            _run_benchmark_for_code, state, proposed_code
        )
    except Exception as exc:
        logger.error("Sandbox benchmark run failed: %s", exc)
        result = _failed_result(state, experiment.id, f"Sandbox benchmark failed: {exc}")
        state["results"] = state["results"] + [result]
        state["completed_experiments"] = state["completed_experiments"] + [experiment]
        state["experiment_queue"] = state["experiment_queue"][1:]
        state["latest_summary"] = f"Sandbox benchmark failed: {exc}"
        return state

    if metrics.total_questions <= 0 or not question_results:
        result = _failed_result(
            state,
            experiment.id,
            "Benchmark returned zero questions. Check sandbox data and benchmark_root mounts.",
        )
        state["results"] = state["results"] + [result]
        state["completed_experiments"] = state["completed_experiments"] + [experiment]
        state["experiment_queue"] = state["experiment_queue"][1:]
        state["latest_summary"] = result.failure_analysis
        return state

    # Step 3: Also run the incumbent session code for first-iteration A/B comparison.
    incumbent_code = state.get("current_pipeline_code", "")
    incumbent_score = state.get("best_score", -1.0)

    if incumbent_code and incumbent_score < 0:
        try:
            _, incumbent_metrics = await asyncio.to_thread(
                _run_benchmark_for_code, state, incumbent_code
            )
            from src.benchmark.metrics import composite_score as calc_score
            incumbent_score = calc_score(incumbent_metrics)
        except Exception as exc:
            logger.warning("Incumbent benchmark failed; using current best score: %s", exc)
            incumbent_score = state.get("best_score", 0.0)

    from src.benchmark.metrics import composite_score as calc_score
    candidate_score = calc_score(metrics)

    result = ExperimentResult(
        experiment_id=experiment.id,
        status=ExperimentStatus.completed,
        metrics=metrics,
        composite_score=candidate_score,
        baseline_score=incumbent_score if incumbent_score >= 0 else candidate_score,
        delta_vs_baseline=candidate_score - incumbent_score if incumbent_score >= 0 else 0.0,
        question_results=question_results,
    )

    state["results"] = state["results"] + [result]
    state["completed_experiments"] = state["completed_experiments"] + [experiment]
    state["experiment_queue"] = state["experiment_queue"][1:]
    return state
