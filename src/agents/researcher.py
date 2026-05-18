from __future__ import annotations

from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.agents.history import format_best_config_json, format_experiment_history, format_rejected_experiments_summary
from src.agents.state import ResearchLabState
from src.agents.text_utils import extract_text_content
from src.config import get_google_api_key, get_settings
from src.models import Hypothesis


def _fallback_hypothesis(iteration: int) -> Hypothesis:
    return Hypothesis(
        id=f"hyp_{uuid4().hex[:8]}",
        title=f"Iteration {iteration} retrieval refinement",
        rationale="Prior experiments indicate retrieval misses cross-source context.",
        expected_impact="Improve recall on multi-hop and constrained questions.",
    )


def _candidate_models(preferred: str) -> list[str]:
    ordered = [preferred, "gemini-2.5-pro", "gemini-2.5-flash"]
    seen: set[str] = set()
    models: list[str] = []
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _format_technique_registry_for_researcher(registry: list[dict]) -> str:
    if not registry:
        return "No techniques tried yet."
    lines = []
    for entry in registry:
        verdict = "ACCEPTED" if entry.get("accepted") else "REJECTED"
        technique = entry.get("technique", "unknown")
        impact = entry.get("per_type_impact", {})
        impact_parts = [f"{t}: {d:+.3f}" for t, d in impact.items()] if impact else ["n/a"]
        lines.append(f"- [{verdict}] {technique} (impact: {', '.join(impact_parts)})")
    return "\n".join(lines)


def _build_researcher_prompt(state: ResearchLabState) -> str:
    history = state.get("latest_summary") or "No prior summary."
    per_type = state.get("per_type_summary") or "No breakdown available."
    best_cfg = format_best_config_json(state)
    exp_history = format_experiment_history(state, max_entries=8)
    rejected = format_rejected_experiments_summary(state, max_entries=8)
    recommendation = state.get("recommendation", "")
    registry = state.get("technique_registry", [])
    question_focus = state.get("question_focus", "all")

    prompt = (
        "You are a RAG research agent optimizing retrieval for enterprise QA.\n\n"
        "## Latest experiment summary\n"
        f"{history}\n\n"
        "## Per-question-type breakdown\n"
        f"{per_type}\n\n"
        "## Current best retrieval config\n"
        f"{best_cfg}\n\n"
        "## Previously tried experiments\n"
        f"{exp_history}\n\n"
        "## Rejected experiments (DO NOT repeat these)\n"
        f"{rejected}\n\n"
    )
    if registry:
        prompt += (
            "## Technique registry (what worked and what didn't)\n"
            f"{_format_technique_registry_for_researcher(registry)}\n\n"
        )
    if question_focus and question_focus != "all":
        prompt += (
            f"## CURRENT FOCUS: {question_focus}\n"
            "The system has auto-focused on this question type because of repeated "
            "rejections. Prioritize hypotheses that target this type.\n\n"
        )
    if recommendation:
        prompt += f"## System recommendation\n{recommendation}\n\n"
    prompt += (
        "## Task\n"
        "Propose ONE concise retrieval hypothesis to test next. "
        "Focus on the weakest question types. "
        "Avoid repeating approaches that were already tried and rejected.\n"
        "Return as: TITLE|||RATIONALE|||EXPECTED_IMPACT\n"
    )
    return prompt


async def researcher_agent(state: ResearchLabState) -> ResearchLabState:
    settings = get_settings()
    iteration = state["iteration"]
    hypothesis = _fallback_hypothesis(iteration)

    if settings.has_google_key:
        prompt = _build_researcher_prompt(state)
        for model_name in _candidate_models(settings.gemini_model):
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=get_google_api_key(),
                    temperature=0.4,
                )
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                text = extract_text_content(response.content)
                parts = [p.strip() for p in text.split("|||")]
                if len(parts) == 3:
                    hypothesis = Hypothesis(
                        id=f"hyp_{uuid4().hex[:8]}",
                        title=parts[0] or hypothesis.title,
                        rationale=parts[1] or hypothesis.rationale,
                        expected_impact=parts[2] or hypothesis.expected_impact,
                    )
                break
            except Exception:
                continue

    hypotheses = state["hypotheses"] + [hypothesis]
    state["hypotheses"] = hypotheses
    state["current_phase"] = "researcher"
    return state
