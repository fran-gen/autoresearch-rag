from types import SimpleNamespace

import pytest

import src.agents.karpathy_worker as karpathy_worker
from src.models import BenchmarkMetrics


def test_docker_unavailable_falls_back_to_in_process(monkeypatch):
    settings = SimpleNamespace(
        karpathy_sandbox_enabled=True,
        karpathy_sandbox_fallback_to_process=True,
    )
    monkeypatch.setattr(karpathy_worker, "get_settings", lambda: settings)

    def fail_docker(state, pipeline_code):
        raise RuntimeError(
            "Docker sandbox failed: Error while fetching server API version: "
            "('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))"
        )

    def run_in_process(state, pipeline_code):
        return [], BenchmarkMetrics(total_questions=1, answered_questions=1)

    monkeypatch.setattr(karpathy_worker, "_run_benchmark_in_docker", fail_docker)
    monkeypatch.setattr(karpathy_worker, "_run_benchmark_isolated_in_process", run_in_process)

    _, metrics = karpathy_worker._run_benchmark_for_code({}, "def retrieve(): pass")

    assert metrics.total_questions == 1


def test_sandbox_timeout_does_not_fall_back_to_in_process(monkeypatch):
    settings = SimpleNamespace(
        karpathy_sandbox_enabled=True,
        karpathy_sandbox_fallback_to_process=True,
    )
    monkeypatch.setattr(karpathy_worker, "get_settings", lambda: settings)

    def fail_docker(state, pipeline_code):
        raise RuntimeError("Sandbox timed out after 180s and was stopped.")

    def run_in_process(state, pipeline_code):
        raise AssertionError("timeout failures should stay isolated to the sandbox")

    monkeypatch.setattr(karpathy_worker, "_run_benchmark_in_docker", fail_docker)
    monkeypatch.setattr(karpathy_worker, "_run_benchmark_isolated_in_process", run_in_process)

    with pytest.raises(RuntimeError, match="timed out"):
        karpathy_worker._run_benchmark_for_code({}, "def retrieve(): pass")


def test_clean_log_excerpt_collapses_progress_bar_noise():
    raw = "Loading weights:  0%|          | 0/103\rLoading weights: 100%|##########| 103/103\n"

    cleaned = karpathy_worker._clean_log_excerpt(raw)

    assert "\r" not in cleaned
    assert "\n" not in cleaned
    assert "Loading weights" in cleaned
