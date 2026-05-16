"""Utilities for formatting experiment history for LLM prompts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.state import ResearchLabState


def format_experiment_history(state: ResearchLabState, max_entries: int = 10) -> str:
    """One line per past experiment: config summary, score, delta, accepted/rejected."""
    experiments = state.get("completed_experiments", [])
    results = state.get("results", [])
    result_map = {r.experiment_id: r for r in results}

    if not experiments:
        return "No experiments run yet."

    lines: list[str] = []
    for exp in experiments[-max_entries:]:
        cfg = exp.retrieval_config
        cfg_desc = (
            f"strategy={cfg.strategy} top_k={cfg.top_k} "
            f"reranker={cfg.use_reranker}"
        )
        if cfg.strategy == "hybrid":
            cfg_desc += f" bm25_w={cfg.bm25_weight:.2f} dense_w={cfg.dense_weight:.2f}"
        result = result_map.get(exp.id)
        if result:
            verdict = "ACCEPTED" if result.accepted else "REJECTED"
            score_part = f"score={result.composite_score:.4f}"
            delta_part = (
                f"delta={result.delta_vs_baseline:.4f}"
                if result.delta_vs_baseline is not None
                else ""
            )
            reason = result.improvement_summary or ""
            lines.append(
                f"  {exp.id}: {cfg_desc} -> {score_part} {delta_part} [{verdict}]"
                + (f" — {reason}" if reason else "")
            )
        else:
            lines.append(f"  {exp.id}: {cfg_desc} -> (no result)")

    return "\n".join(lines)


def format_best_config_json(state: ResearchLabState) -> str:
    """Return best config as compact JSON string for LLM prompt."""
    best = state.get("best_config")
    if best is None:
        return "{}"
    return best.model_dump_json(
        indent=2,
        exclude={"evaluation_mode", "embedding_model", "extra"},
    )


def format_config_for_karpathy(state: ResearchLabState) -> str:
    """Return config JSON with extra fields flattened for Karpathy prompt readability."""
    best = state.get("best_config")
    if best is None:
        return "{}"
    d = best.model_dump(exclude={"embedding_model", "evaluation_mode"})
    extra = d.pop("extra", None) or {}
    for key in ("query_rewrite", "source_diversity"):
        if key in extra:
            d[key] = extra[key]
    return json.dumps(d, indent=2)


def format_rejected_experiments_summary(state: ResearchLabState, max_entries: int = 8) -> str:
    """Summarise rejected experiments so the LLM avoids repeating them."""
    results = state.get("results", [])
    experiments = state.get("completed_experiments", [])
    exp_map = {e.id: e for e in experiments}
    rejected = [r for r in results if not r.accepted]
    if not rejected:
        return "No rejected experiments yet."

    lines: list[str] = []
    for r in rejected[-max_entries:]:
        exp = exp_map.get(r.experiment_id)
        cfg_desc = ""
        if exp:
            cfg = exp.retrieval_config
            cfg_desc = (
                f"strategy={cfg.strategy} top_k={cfg.top_k} "
                f"reranker={cfg.use_reranker}"
            )
            if cfg.strategy == "hybrid":
                cfg_desc += f" bm25_w={cfg.bm25_weight:.2f} dense_w={cfg.dense_weight:.2f}"
        reason = r.improvement_summary or r.failure_analysis or "unknown"
        lines.append(f"  {r.experiment_id}: {cfg_desc} score={r.composite_score:.4f} — {reason}")
    return "\n".join(lines)
