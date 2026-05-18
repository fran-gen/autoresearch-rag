"""Karpathy Worker -- validates code and evaluates it in a session sandbox."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import io
import json
import logging
import sys
import tarfile
import tempfile
from pathlib import Path

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

ALLOWED_IMPORTS = frozenset({
    "src.retrieval.base",
    "src.retrieval.embeddings",
    "src.retrieval.reranker",
    "src.benchmark.loader",
    "re",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "__future__",
})

FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "requests",
    "urllib", "pathlib", "shutil", "http", "ftplib",
    "smtplib", "ctypes", "multiprocessing",
})


def _validate_imports(code: str) -> str | None:
    """Check that only allowed modules are imported. Returns error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    return f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    return f"Forbidden import from: {node.module}"
    return None


def _validate_signature(code: str) -> str | None:
    """Verify the retrieve() function exists with correct positional parameters.

    Keyword-only args (encoder, reranker) are optional and not checked here.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "retrieve":
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names == ["question", "retriever", "top_k"]:
                return None
            return f"Wrong signature: retrieve({', '.join(arg_names)}). Expected (question, retriever, top_k)"
    return "Missing retrieve() function"


def _validate_code(code: str) -> str | None:
    """Run all validation checks. Returns error message or None on success."""
    # 1. Syntax check
    try:
        compile(code, "pipeline.py", "exec")
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    # 2. Import check
    err = _validate_imports(code)
    if err:
        return err

    # 3. Signature check
    err = _validate_signature(code)
    if err:
        return err

    return None


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
    client = docker.from_env()
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
    if settings.karpathy_sandbox_network_disabled:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"

    try:
        try:
            container = client.containers.create(
                settings.karpathy_sandbox_image,
                command=[
                    "python",
                    "-m",
                    "src.benchmark.karpathy_sandbox_runner",
                    "/tmp/karpathy_state.json",
                ],
                working_dir="/app",
                environment=environment,
                mem_limit=settings.karpathy_sandbox_memory,
                network_disabled=settings.karpathy_sandbox_network_disabled,
                volumes=volumes or None,
            )
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
        exit_result = container.wait(timeout=settings.karpathy_sandbox_timeout_seconds)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        status_code = exit_result.get("StatusCode", 1) if isinstance(exit_result, dict) else 1
        if status_code != 0:
            if status_code == 137:
                raise RuntimeError(
                    "Sandbox exited with 137, which usually means Docker killed it for "
                    f"memory pressure. Current KARPATHY_SANDBOX_MEMORY="
                    f"{settings.karpathy_sandbox_memory}. Last logs: {logs[-1200:]}"
                )
            raise RuntimeError(f"Sandbox exited with {status_code}: {logs[-1200:]}")

        result_line = next(
            (line[len(RESULT_PREFIX):] for line in reversed(logs.splitlines()) if line.startswith(RESULT_PREFIX)),
            "",
        )
        if not result_line:
            raise RuntimeError(f"Sandbox did not emit a result payload: {logs[-1200:]}")
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
        return _run_benchmark_in_docker(state, pipeline_code)
    return _run_benchmark_isolated_in_process(state, pipeline_code)


def _benchmark_state_with_config(
    state: ResearchLabState,
    config: RetrievalConfig | None,
) -> ResearchLabState:
    """Return a shallow state copy whose benchmark config is explicit.

    Karpathy benchmark payloads prefer `proposed_config` over `best_config`.
    Incumbent evaluations therefore must clear `proposed_config`, otherwise
    the old pipeline code is measured with the new candidate config.
    """
    benchmark_state = dict(state)
    benchmark_state["proposed_config"] = None
    if config is not None:
        benchmark_state["best_config"] = config
    return benchmark_state  # type: ignore[return-value]


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

    # Step 2: Collect all valid candidates (primary + extras from best-of-N).
    all_candidates = [{"code": proposed_code, "config": state.get("proposed_config")}]
    for extra in state.get("proposed_candidates", []):
        extra_code = extra.get("code", "")
        if extra_code and _validate_code(extra_code) is None:
            cfg = extra.get("config")
            if isinstance(cfg, dict):
                cfg = RetrievalConfig.model_validate(cfg)
            all_candidates.append({"code": extra_code, "config": cfg})

    # Step 3: Evaluate all candidates and pick the best.
    from src.benchmark.metrics import composite_score as calc_score

    best_qr: list[QuestionResult] | None = None
    best_metrics: BenchmarkMetrics | None = None
    best_score = -1.0
    best_candidate_code = proposed_code
    best_candidate_config = state.get("proposed_config")

    for i, cand in enumerate(all_candidates):
        cand_code = cand["code"]
        eval_state = _benchmark_state_with_config(state, cand.get("config"))
        eval_state["proposed_config"] = cand.get("config")
        try:
            qr, m = await asyncio.to_thread(
                _run_benchmark_for_code, eval_state, cand_code
            )
        except Exception as exc:
            logger.warning("Candidate %d/%d benchmark failed: %s", i + 1, len(all_candidates), exc)
            continue
        if m.total_questions <= 0 or not qr:
            continue
        score = calc_score(m)
        logger.info(
            "Candidate %d/%d score=%.4f run_id=%s",
            i + 1, len(all_candidates), score, state.get("run_id"),
        )
        if score > best_score:
            best_score = score
            best_qr = qr
            best_metrics = m
            best_candidate_code = cand_code
            best_candidate_config = cand.get("config")

    if best_qr is None or best_metrics is None:
        result = _failed_result(
            state,
            experiment.id,
            "All candidates failed benchmark evaluation.",
        )
        state["results"] = state["results"] + [result]
        state["completed_experiments"] = state["completed_experiments"] + [experiment]
        state["experiment_queue"] = state["experiment_queue"][1:]
        state["latest_summary"] = result.failure_analysis
        return state

    state["proposed_code"] = best_candidate_code
    if best_candidate_config is not None:
        state["proposed_config"] = best_candidate_config

    question_results = best_qr
    metrics = best_metrics
    candidate_score = best_score

    # Step 4: Also run the incumbent session code for first-iteration A/B comparison.
    incumbent_code = state.get("current_pipeline_code", "")
    incumbent_score = state.get("best_score", -1.0)
    incumbent_config = state.get("best_config")

    if incumbent_code and incumbent_score < 0:
        try:
            incumbent_state = _benchmark_state_with_config(
                state,
                incumbent_config if isinstance(incumbent_config, RetrievalConfig) else None,
            )
            _, incumbent_metrics = await asyncio.to_thread(
                _run_benchmark_for_code, incumbent_state, incumbent_code
            )
            incumbent_score = calc_score(incumbent_metrics)
        except Exception as exc:
            logger.warning("Incumbent benchmark failed; using current best score: %s", exc)
            incumbent_score = state.get("best_score", 0.0)

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
