from src.agents.karpathy_worker import _benchmark_state_with_config, _sandbox_thread_env
from src.models import RetrievalConfig


def _config(strategy: str, top_k: int) -> RetrievalConfig:
    return RetrievalConfig(
        strategy=strategy,
        embedding_model="test-embedding",
        top_k=top_k,
        use_reranker=False,
    )


def test_benchmark_state_with_config_clears_candidate_config_for_incumbent():
    incumbent_config = _config("dense", 6)
    candidate_config = _config("hybrid", 12)
    state = {
        "best_config": incumbent_config,
        "proposed_config": candidate_config,
    }

    benchmark_state = _benchmark_state_with_config(state, incumbent_config)

    assert benchmark_state["proposed_config"] is None
    assert benchmark_state["best_config"] == incumbent_config
    assert state["proposed_config"] == candidate_config


def test_sandbox_thread_env_sets_cpu_bound_library_limits():
    env = _sandbox_thread_env(2)

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["NUMEXPR_NUM_THREADS"] == "2"
    assert env["VECLIB_MAXIMUM_THREADS"] == "2"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
