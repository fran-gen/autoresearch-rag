from __future__ import annotations

import difflib
from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.agents.state import ResearchLabState
from src.benchmark.loader import EnterpriseRagBenchLoader
from src.config import get_google_api_key, get_settings
from src.models import ExperimentStatus, QuestionResult

logger = logging.getLogger(__name__)


def _config_fingerprint_from_dict(config: dict) -> str:
    payload = {k: v for k, v in config.items() if k not in {"embedding_model", "evaluation_mode", "extra"}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def _build_per_type_breakdown(
    question_results: list[QuestionResult],
    top_k: int,
    benchmark_root: str | None = None,
) -> tuple[str, list[dict]]:
    """Compute per-question-type recall/precision and pick worst failures."""
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(Path(benchmark_root) if benchmark_root else settings.benchmark_root)
    questions = loader.load_questions()
    q_map = {q.question_id: q for q in questions}

    type_recalls: dict[str, list[float]] = defaultdict(list)
    type_precisions: dict[str, list[float]] = defaultdict(list)
    type_counts: dict[str, int] = defaultdict(int)
    scored: list[tuple[float, dict]] = []

    for result in question_results:
        q = q_map.get(result.question_id)
        if q is None:
            continue
        qtype = q.question_type or "unknown"
        type_counts[qtype] += 1
        gt = set(q.expected_doc_ids) if q.expected_doc_ids else set()
        pred = set(result.document_ids[:top_k])
        if not gt:
            continue
        recall = len(pred & gt) / len(gt)
        precision = len(pred & gt) / len(pred) if pred else 0.0
        type_recalls[qtype].append(recall)
        type_precisions[qtype].append(precision)
        scored.append((recall, {
            "question_id": result.question_id,
            "question": q.question[:200],
            "question_type": qtype,
            "expected_doc_ids": q.expected_doc_ids[:5],
            "retrieved_doc_ids": result.document_ids[:top_k],
            "recall": round(recall, 3),
        }))

    lines: list[str] = []
    for qtype in sorted(type_counts, key=lambda t: type_counts[t], reverse=True):
        n = type_counts[qtype]
        recs = type_recalls.get(qtype, [])
        precs = type_precisions.get(qtype, [])
        avg_r = sum(recs) / max(len(recs), 1)
        avg_p = sum(precs) / max(len(precs), 1)
        lines.append(f"{qtype}: recall={avg_r:.3f} precision={avg_p:.3f} ({n}q)")

    per_type_summary = " | ".join(lines) if lines else "No per-type data."
    scored.sort(key=lambda x: x[0])
    failure_examples = [item for _, item in scored[:5]]
    return per_type_summary, failure_examples


def _classify_failures(
    question_results: list[QuestionResult],
    top_k: int,
    benchmark_root: str | None = None,
) -> dict[str, int]:
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(Path(benchmark_root) if benchmark_root else settings.benchmark_root)
    q_map = {q.question_id: q for q in loader.load_questions()}
    taxonomy = {
        "no_relevant_doc_retrieved": 0,
        "retrieval_noise": 0,
        "answer_failed_with_context": 0,
        "unknown_or_unlabeled": 0,
    }

    for result in question_results:
        q = q_map.get(result.question_id)
        if q is None or not q.expected_doc_ids:
            taxonomy["unknown_or_unlabeled"] += 1
            continue
        expected = set(q.expected_doc_ids)
        retrieved = set(result.document_ids[:top_k])
        overlap = expected & retrieved
        if not overlap:
            taxonomy["no_relevant_doc_retrieved"] += 1
        elif result.is_correct is False:
            taxonomy["answer_failed_with_context"] += 1
        elif len(overlap) / max(len(retrieved), 1) < 0.2:
            taxonomy["retrieval_noise"] += 1

    return taxonomy


def _recommend_next_action(per_type_summary: str, taxonomy: dict[str, int]) -> str:
    if not taxonomy:
        return "Run at least one experiment to generate a targeted recommendation."
    biggest = max(taxonomy.items(), key=lambda item: item[1])
    if biggest[0] == "no_relevant_doc_retrieved":
        return "Relevant documents are missing from top-k. Try hybrid retrieval, query rewriting, or a larger top-k."
    if biggest[0] == "retrieval_noise":
        return "Retrieved context is noisy. Try reranking, lower top-k, or stronger dense/BM25 weighting."
    if biggest[0] == "answer_failed_with_context":
        return "Relevant context is present but answers fail. Use full evaluation, answer-fact checks, or better answer prompts."
    return f"Continue tuning from the per-type breakdown: {per_type_summary or 'no per-type data yet.'}"


async def _generate_final_report(state: ResearchLabState) -> str:
    """Ask the LLM for a synthesis of the research session."""
    settings = get_settings()
    if not settings.has_google_key:
        return _fallback_report(state)

    results = state.get("results", [])
    initial = state.get("initial_baseline_score", -1.0)
    best = state.get("best_score", -1.0)
    history_lines = []
    for r in results:
        verdict = "ACCEPTED" if r.accepted else "REJECTED"
        history_lines.append(
            f"  {r.experiment_id}: score={r.composite_score:.4f} "
            f"delta={r.delta_vs_baseline:.4f} [{verdict}]"
        )

    prompt = (
        "You are a research assistant summarizing an automated RAG optimization session.\n\n"
        f"## Session stats\n"
        f"- Iterations: {state['iteration']}\n"
        f"- Initial baseline score: {initial:.4f}\n"
        f"- Final best score: {best:.4f}\n"
        f"- Improvement: {best - initial:.4f} ({((best - initial) / max(initial, 0.001)) * 100:.1f}%)\n"
        f"- Accepted: {state['accepted_experiments']}, Rejected: {state['rejected_experiments']}\n\n"
        f"## Experiment history\n{''.join(history_lines) if history_lines else 'None'}\n\n"
        f"## Final per-type performance\n{state.get('per_type_summary', 'N/A')}\n\n"
        "## Task\n"
        "Write a concise research report (3-5 paragraphs) covering:\n"
        "1. What the system tried and what worked\n"
        "2. Key strengths (which question types are well-served)\n"
        "3. Remaining weaknesses and why\n"
        "4. Concrete next steps to improve further\n\n"
        "Write in a professional but accessible tone. Use specific numbers."
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_fast_model,
            google_api_key=get_google_api_key(),
            temperature=0.3,
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return str(response.content)
    except Exception:
        return _fallback_report(state)


def _fallback_report(state: ResearchLabState) -> str:
    initial = state.get("initial_baseline_score", -1.0)
    best = state.get("best_score", -1.0)
    delta = best - initial if initial >= 0 else 0.0
    pct = (delta / max(initial, 0.001)) * 100

    return (
        f"Research session completed: {state['iteration']} iterations, "
        f"{state['accepted_experiments']} accepted, "
        f"{state['rejected_experiments']} rejected.\n\n"
        f"Score improved from {initial:.4f} to {best:.4f} "
        f"(+{delta:.4f}, +{pct:.1f}%).\n\n"
        f"Remaining weak areas: {state.get('per_type_summary', 'N/A')}"
    )


async def evaluator_agent(state: ResearchLabState) -> ResearchLabState:
    state["current_phase"] = "evaluator"
    if not state["results"]:
        state["latest_summary"] = "No experiment results yet."
        state["iteration"] += 1
        return state

    latest = state["results"][-1]
    latest_experiment = (
        state["completed_experiments"][-1] if state["completed_experiments"] else None
    )
    metrics = latest.metrics

    current_best = (
        latest.baseline_score
        if latest.baseline_score is not None
        else state["best_score"]
    )
    min_delta = state["min_improvement_delta"]
    delta = (
        latest.delta_vs_baseline
        if latest.delta_vs_baseline is not None
        else (latest.composite_score - current_best if current_best >= 0 else 0.0)
    )

    validation_delta = latest.validation_delta
    validation_ok = validation_delta is None or validation_delta >= 0
    completed_ok = latest.status == ExperimentStatus.completed
    if completed_ok and state["initial_baseline_score"] < 0:
        state["initial_baseline_score"] = (
            latest.baseline_score
            if latest.baseline_score is not None and latest.baseline_score >= 0
            else latest.composite_score
        )

    threshold_ok = current_best < 0 or delta >= min_delta
    improved = completed_ok and threshold_ok and validation_ok
    verdict_reason = "Accepted."
    if improved and latest_experiment is not None:
        state["best_score"] = latest.composite_score
        state["best_config"] = latest_experiment.retrieval_config
        state["accepted_experiments"] += 1
        latest.accepted = True
        verdict_reason = (
            f"Accepted: score improved to {latest.composite_score:.4f} "
            f"(baseline {current_best:.4f}, delta {delta:.4f})."
        )
        latest.improvement_summary = verdict_reason
    else:
        state["rejected_experiments"] += 1
        latest.accepted = False
        if not completed_ok:
            verdict_reason = (
                f"Rejected: experiment failed before producing a valid benchmark result. "
                f"{latest.failure_analysis}"
            )
        elif not threshold_ok:
            verdict_reason = (
                f"Rejected: score {latest.composite_score:.4f} did not exceed "
                f"baseline {current_best:.4f} by >= {min_delta:.4f} "
                f"(delta {delta:.4f})."
            )
        else:
            holdout_delta = validation_delta if validation_delta is not None else 0.0
            verdict_reason = (
                f"Rejected: tuning score improved (delta {delta:.4f}) but holdout "
                f"delta was negative ({holdout_delta:.4f})."
            )
        latest.improvement_summary = verdict_reason
        if latest_experiment is not None:
            fp = _config_fingerprint_from_dict(latest_experiment.retrieval_config.model_dump())
            state["rejected_config_fingerprints"] = list(
                dict.fromkeys(state.get("rejected_config_fingerprints", []) + [fp])
            )

    # Gather hypothesis title and config for timeline display
    exp_config = latest_experiment.retrieval_config.model_dump() if latest_experiment else {}
    hypothesis_title = ""
    if state.get("hypotheses"):
        hypothesis_title = state["hypotheses"][-1].title if hasattr(state["hypotheses"][-1], "title") else str(state["hypotheses"][-1].get("title", ""))
    rationale = state.get("planner_rationale", "")

    history_baseline = latest.baseline_score
    if history_baseline is None and current_best >= 0:
        history_baseline = current_best

    state["score_history"] = state.get("score_history", []) + [{
        "iteration": state["iteration"],
        "experiment_id": latest.experiment_id,
        "score": round(latest.composite_score, 4),
        "baseline": round(history_baseline, 4) if history_baseline is not None else None,
        "validation_score": round(latest.validation_score, 4) if latest.validation_score is not None else None,
        "validation_delta": round(latest.validation_delta, 4) if latest.validation_delta is not None else None,
        "accepted": latest.accepted,
        "reason": verdict_reason,
        "hypothesis": hypothesis_title,
        "rationale": rationale,
        "config": exp_config,
    }]

    # Karpathy mode: update only this run's session code. The deployed repo file
    # stays immutable, so concurrent users never fight over src/retrieval/pipeline.py.
    if state.get("research_mode") == "karpathy":
        proposed = state.get("proposed_code", "")
        previous = state.get("current_pipeline_code", "")
        diff_lines = list(difflib.unified_diff(
            previous.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile="pipeline.py (before)",
            tofile="pipeline.py (after)",
            n=2,
        ))
        diff_summary = "".join(diff_lines[:30]) if diff_lines else "no diff"

        proposed_config = state.get("proposed_config")
        proposed_config_dump = None
        if proposed_config is not None:
            try:
                proposed_config_dump = proposed_config.model_dump(
                    exclude={"embedding_model", "evaluation_mode"}
                )
            except Exception:
                proposed_config_dump = None

        if latest.accepted:
            state["current_pipeline_code"] = proposed
            if proposed_config is not None:
                state["best_config"] = proposed_config

        state["code_history"] = state.get("code_history", []) + [{
            "iteration": state["iteration"],
            "hypothesis": hypothesis_title,
            "score": round(latest.composite_score, 4),
            "accepted": latest.accepted,
            "diff_summary": diff_summary,
            "proposed_code": proposed,
            "proposed_config": proposed_config_dump,
        }]

    top_k = 8
    if latest_experiment is not None:
        top_k = latest_experiment.retrieval_config.top_k

    per_type_summary, failure_examples = _build_per_type_breakdown(
        latest.question_results, top_k, state.get("benchmark_root")
    )
    state["per_type_summary"] = per_type_summary
    state["failure_examples"] = failure_examples
    taxonomy = _classify_failures(latest.question_results, top_k, state.get("benchmark_root"))
    state["failure_taxonomy"] = taxonomy
    state["recommendation"] = _recommend_next_action(per_type_summary, taxonomy)
    if latest.validation_score is not None:
        state["validation_summary"] = (
            f"Holdout score={latest.validation_score:.4f}, "
            f"delta={(latest.validation_delta or 0.0):+.4f}."
        )

    correctness = metrics.answer_correctness
    parts = [
        f"Experiment {latest.experiment_id}",
        f"recall@k={metrics.recall_at_k:.3f}",
        f"precision@k={metrics.precision_at_k:.3f}",
    ]
    if correctness is not None:
        parts.append(f"correctness={correctness:.3f}")
    parts += [
        f"score={latest.composite_score:.4f}",
        f"baseline={current_best:.4f}" if current_best >= 0 else "baseline=n/a",
        f"delta={delta:.4f}",
        f"holdout_delta={(latest.validation_delta or 0.0):.4f}" if latest.validation_delta is not None else "holdout_delta=n/a",
        f"accepted={latest.accepted}",
    ]
    state["latest_summary"] = ", ".join(parts)

    state["iteration"] += 1
    if state["iteration"] >= state["max_iterations"]:
        state["should_stop"] = True
        state["final_report"] = await _generate_final_report(state)

    return state
