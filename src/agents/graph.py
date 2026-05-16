from __future__ import annotations

# Patch Reviver default so langgraph's module-level `LC_REVIVER = Reviver()`
# uses allowed_objects="core" instead of None, which avoids the deprecation
# warning from langchain-core.
import importlib as _importlib
from uuid import uuid4

_lc_load_mod = _importlib.import_module("langchain_core.load.load")
_OrigReviver = _lc_load_mod.Reviver


class _PatchedReviver(_OrigReviver):  # type: ignore[misc]
    def __init__(self, allowed_objects="core", **kwargs):  # noqa: ANN001
        super().__init__(allowed_objects=allowed_objects, **kwargs)


_lc_load_mod.Reviver = _PatchedReviver  # type: ignore[misc]

from langgraph.graph import END, StateGraph  # noqa: E402

_lc_load_mod.Reviver = _OrigReviver  # restore original

from src.agents.code_planner import code_planner_agent
from src.agents.evaluator import evaluator_agent
from src.agents.karpathy_worker import karpathy_worker_agent
from src.agents.planner import planner_agent
from src.agents.researcher import researcher_agent
from src.agents.state import ResearchLabState
from src.agents.worker import worker_agent
from src.config import get_settings
from src.models import RetrievalConfig


def _route_after_researcher(state: ResearchLabState) -> str:
    """Route to config planner or code planner based on research_mode."""
    if state.get("research_mode") == "karpathy":
        return "code_planner"
    return "planner"


def _route_after_evaluator(state: ResearchLabState) -> str:
    return "end" if state["should_stop"] else "researcher"


def build_research_graph():
    graph = StateGraph(ResearchLabState)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("planner", planner_agent)
    graph.add_node("worker", worker_agent)
    graph.add_node("code_planner", code_planner_agent)
    graph.add_node("karpathy_worker", karpathy_worker_agent)
    graph.add_node("evaluator", evaluator_agent)

    graph.set_entry_point("researcher")
    graph.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {
            "planner": "planner",
            "code_planner": "code_planner",
        },
    )
    graph.add_edge("planner", "worker")
    graph.add_edge("code_planner", "karpathy_worker")
    graph.add_edge("worker", "evaluator")
    graph.add_edge("karpathy_worker", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {
            "researcher": "researcher",
            "end": END,
        },
    )
    return graph.compile()


def default_state(
    max_iterations: int = 3,
    starting_config: dict | RetrievalConfig | None = None,
    question_focus: str = "all",
    benchmark_root: str | None = None,
    research_setup_id: str | None = None,
    research_mode: str = "config",
) -> ResearchLabState:
    settings = get_settings()
    if isinstance(starting_config, RetrievalConfig):
        baseline_config = starting_config
    elif isinstance(starting_config, dict):
        baseline_config = RetrievalConfig.model_validate(starting_config)
    else:
        baseline_config = RetrievalConfig(
            strategy="dense",
            embedding_model=settings.embedding_model,
            top_k=settings.default_top_k,
            use_reranker=False,
            evaluation_mode="fast",
        )
    baseline_config.embedding_model = settings.embedding_model

    run_id = uuid4().hex[:8]

    # Karpathy mode keeps code session-scoped. The shared repo file is read as
    # the starting point, but never mutated by the demo runtime.
    karpathy_branch = ""
    current_pipeline_code = ""
    if research_mode == "karpathy":
        from src.agents.git_ops import PIPELINE_PATH, PROJECT_ROOT
        pipeline_file = PROJECT_ROOT / PIPELINE_PATH
        if pipeline_file.exists():
            current_pipeline_code = pipeline_file.read_text(encoding="utf-8")
        else:
            current_pipeline_code = ""

    return {
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_stop": False,
        "current_phase": "init",
        "run_id": f"run_{run_id}",
        "hypotheses": [],
        "experiment_queue": [],
        "completed_experiments": [],
        "results": [],
        "latest_summary": "",
        "best_config": baseline_config,
        "best_score": -1.0,
        "min_improvement_delta": 0.005,
        "accepted_experiments": 0,
        "rejected_experiments": 0,
        "per_type_summary": "",
        "failure_examples": [],
        "planner_rationale": "",
        "candidate_config": None,
        "final_report": "",
        "initial_baseline_score": -1.0,
        "score_history": [],
        "question_focus": question_focus,
        "benchmark_root": benchmark_root,
        "research_setup_id": research_setup_id,
        "tried_config_fingerprints": [],
        "rejected_config_fingerprints": [],
        "failure_taxonomy": {},
        "recommendation": "Run the first tuned experiment to generate recommendations.",
        "validation_summary": "No holdout validation yet.",
        "dataset_readiness": {},
        # Karpathy mode
        "research_mode": research_mode,
        "proposed_code": "",
        "proposed_config": None,
        "initial_pipeline_code": current_pipeline_code,
        "current_pipeline_code": current_pipeline_code,
        "code_history": [],
        "karpathy_branch": karpathy_branch,
    }
