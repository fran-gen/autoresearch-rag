from __future__ import annotations

import json
import random
import hashlib
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.agents.history import format_best_config_json, format_experiment_history, format_rejected_experiments_summary
from src.agents.state import ResearchLabState
from src.agents.text_utils import extract_text_content
from src.config import get_google_api_key, get_settings
from src.models import ExperimentSpec, RetrievalConfig

PLANNER_PROMPT = """\
You are a retrieval system optimizer. Based on the evidence below, propose ONE new \
RetrievalConfig that addresses the weakest areas. Also provide a brief rationale \
(1-2 sentences) explaining your choice.

## Current best config
{best_config_json}

## Per-question-type performance
{per_type_summary}

## Failure examples (worst recall)
{failure_examples}

## Previously tried configs and outcomes
{experiment_history}

## Rejected experiments (DO NOT repeat these approaches)
{rejected_summary}

## Hypothesis to test
{hypothesis_title}: {hypothesis_rationale}

## Instructions
Return ONLY a valid JSON object with these fields:
- "strategy": "dense" or "hybrid"
- "top_k": integer 3-20
- "use_reranker": true or false
- "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2" or null
- "bm25_weight": float 0.0-1.0 (used when strategy="hybrid")
- "dense_weight": float 0.0-1.0 (used when strategy="hybrid")
- "query_rewrite": true or false
- "source_diversity": true or false
- "rationale": string with 1-2 sentence explanation

Return ONLY the JSON object, no markdown fences or extra text.
"""


def _config_fingerprint(config: RetrievalConfig) -> str:
    payload = config.model_dump(exclude={"embedding_model", "evaluation_mode", "extra"})
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _random_fallback(
    base: RetrievalConfig, iteration: int, avoided: set[str] | None = None,
) -> tuple[RetrievalConfig, str]:
    """Generate a genuinely random config perturbation (not the old cycle)."""
    rnd = random.Random(42 + iteration)
    avoided = avoided or set()
    config = base.model_copy(deep=True)
    for attempt in range(20):
        strategy = rnd.choice(["dense", "hybrid"])
        top_k = rnd.choice([4, 6, 8, 10, 12, 16, 20])
        use_reranker = rnd.random() < 0.45
        bm25_w = round(rnd.uniform(0.15, 0.85), 2)
        dense_w = round(1.0 - bm25_w, 2)

        config = base.model_copy(deep=True)
        config.strategy = strategy
        config.top_k = top_k
        config.use_reranker = use_reranker
        config.reranker_model = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None
        )
        config.bm25_weight = bm25_w
        config.dense_weight = dense_w
        config.evaluation_mode = "fast"
        config.extra = {
            **(config.extra or {}),
            "search_attempt": attempt,
            "query_rewrite": rnd.random() < 0.45,
            "source_diversity": rnd.random() < 0.55,
            "question_type_overrides": {
                "semantic": {"top_k": max(top_k, 10)},
                "multi_hop": {"top_k": max(top_k, 12)},
                "comparison": {"top_k": max(top_k, 10)},
                "basic": {"top_k": min(top_k, 8)},
            },
        }
        if _config_fingerprint(config) not in avoided:
            break

    rationale = (
        f"Random exploration: {strategy} strategy, top_k={top_k}, "
        f"reranker={'on' if use_reranker else 'off'}"
    )
    return config, rationale


def _parse_config_from_llm(
    text: str, base: RetrievalConfig,
) -> tuple[RetrievalConfig, str] | None:
    """Try to extract a RetrievalConfig + rationale from an LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    rationale = str(data.pop("rationale", "No rationale provided."))
    config = base.model_copy(deep=True)

    if data.get("strategy") in ("dense", "hybrid"):
        config.strategy = data["strategy"]
    if isinstance(data.get("top_k"), (int, float)):
        config.top_k = max(3, min(20, int(data["top_k"])))
    if isinstance(data.get("use_reranker"), bool):
        config.use_reranker = data["use_reranker"]
    if config.use_reranker:
        config.reranker_model = (
            data.get("reranker_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    else:
        config.reranker_model = None
    if isinstance(data.get("bm25_weight"), (int, float)):
        config.bm25_weight = max(0.0, min(1.0, float(data["bm25_weight"])))
    if isinstance(data.get("dense_weight"), (int, float)):
        config.dense_weight = max(0.0, min(1.0, float(data["dense_weight"])))
    config.extra = dict(config.extra or {})
    if isinstance(data.get("query_rewrite"), bool):
        config.extra["query_rewrite"] = data["query_rewrite"]
    if isinstance(data.get("source_diversity"), bool):
        config.extra["source_diversity"] = data["source_diversity"]
    config.extra.setdefault(
        "question_type_overrides",
        {
            "semantic": {"top_k": max(config.top_k, 10)},
            "multi_hop": {"top_k": max(config.top_k, 12)},
            "comparison": {"top_k": max(config.top_k, 10)},
            "basic": {"top_k": min(config.top_k, 8)},
        },
    )

    config.evaluation_mode = "fast"
    return config, rationale


def _build_planner_prompt(
    state: ResearchLabState,
    hypothesis_title: str,
    hypothesis_rationale: str,
) -> str:
    failure_ex = state.get("failure_examples", [])
    failure_text = json.dumps(failure_ex[:5], indent=2) if failure_ex else "None yet."

    return PLANNER_PROMPT.format(
        best_config_json=format_best_config_json(state),
        per_type_summary=state.get("per_type_summary") or "No breakdown yet.",
        failure_examples=failure_text,
        experiment_history=format_experiment_history(state, max_entries=8),
        rejected_summary=format_rejected_experiments_summary(state, max_entries=8),
        hypothesis_title=hypothesis_title,
        hypothesis_rationale=hypothesis_rationale,
    )


async def planner_agent(state: ResearchLabState) -> ResearchLabState:
    settings = get_settings()
    if not state["hypotheses"]:
        state["current_phase"] = "planner"
        return state

    latest_hypothesis = state["hypotheses"][-1]
    base_config = state["best_config"] or RetrievalConfig(
        strategy="dense",
        embedding_model=settings.embedding_model,
        top_k=settings.default_top_k,
        use_reranker=False,
    )

    candidate_config: RetrievalConfig | None = None
    rationale = ""

    if settings.has_google_key:
        prompt = _build_planner_prompt(
            state,
            latest_hypothesis.title,
            latest_hypothesis.rationale,
        )
        for model_name in [
            settings.gemini_fast_model,
            settings.gemini_model,
            "gemini-2.5-flash",
        ]:
            if not model_name:
                continue
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=get_google_api_key(),
                    temperature=0.3,
                )
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                parsed = _parse_config_from_llm(extract_text_content(response.content), base_config)
                if parsed is not None:
                    candidate_config, rationale = parsed
                    candidate_config.embedding_model = settings.embedding_model
                break
            except Exception:
                continue

    avoided = set(state.get("tried_config_fingerprints", [])) | set(state.get("rejected_config_fingerprints", []))
    if candidate_config is None or _config_fingerprint(candidate_config) in avoided:
        candidate_config, rationale = _random_fallback(
            base_config, state["iteration"], avoided,
        )
        candidate_config.embedding_model = settings.embedding_model

    state["planner_rationale"] = rationale
    state["candidate_config"] = candidate_config
    state["tried_config_fingerprints"] = list(dict.fromkeys(state.get("tried_config_fingerprints", []) + [_config_fingerprint(candidate_config)]))

    spec = ExperimentSpec(
        id=f"exp_{uuid4().hex[:8]}",
        hypothesis_id=latest_hypothesis.id,
        name=f"Experiment for {latest_hypothesis.title[:48]}",
        description=f"{latest_hypothesis.rationale}\n\nPlanner: {rationale}",
        retrieval_config=candidate_config,
        run_id=state.get("run_id"),
        run_position=state["iteration"] + 1,
    )
    state["experiment_queue"] = state["experiment_queue"] + [spec]
    state["current_phase"] = "planner"
    return state
