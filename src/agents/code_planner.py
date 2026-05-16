"""Code Planner for Karpathy mode -- generates pipeline.py code AND config via LLM."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.history import format_config_for_karpathy
from src.agents.state import ResearchLabState
from src.config import get_settings
from src.models import ExperimentSpec, RetrievalConfig

logger = logging.getLogger(__name__)

CONFIG_DELIMITER = "===CONFIG==="

CODE_PLANNER_PROMPT = """\
You are an autonomous retrieval engineer. Your goal is to improve the
retrieve() function in pipeline.py AND tune the retrieval configuration
to get better recall and precision on the benchmark.

You can change BOTH the code AND the config on each iteration.

## CURRENT CODE (composite score: {current_score})
```python
{current_code}
```

## CURRENT RETRIEVAL CONFIG
```json
{current_config}
```

## PERFORMANCE BY QUESTION TYPE
{per_type_summary}

## WORST FAILURES (lowest recall questions)
{failure_examples}

## PREVIOUSLY TRIED APPROACHES AND RESULTS (do NOT repeat failed ideas)
{code_history}

## HYPOTHESIS TO TEST
{hypothesis}

## AVAILABLE API — use ONLY these classes and methods

```python
@dataclass
class BenchmarkQuestion:
    question_id: str        # unique identifier
    question: str           # the natural-language question
    question_type: str      # e.g. "factoid", "semantic", "multi_hop", "comparison"
    source_types: list[str] # document source categories
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]
    # NOTE: there is NO .metadata attribute

@dataclass
class RetrievedDocument:
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]

class BaseRetriever(ABC):
    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedDocument]:
        ...
    # This is the ONLY method available. There is NO retrieve_bm25,
    # NO retrieve_sparse, NO search, NO other method.
```

## RETRIEVAL CONFIG OPTIONS
You can also modify the retrieval config alongside the code. Available fields:
- "strategy": "dense" or "hybrid" (hybrid adds BM25 keyword matching fused with dense)
- "top_k": integer 3-20 (number of documents to retrieve)
- "bm25_weight": float 0.0-1.0 (BM25 fusion weight, used when strategy="hybrid")
- "dense_weight": float 0.0-1.0 (dense fusion weight, used when strategy="hybrid")
- "use_reranker": true or false (cross-encoder reranker applied AFTER retrieve())
- "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2" or null

IMPORTANT NOTES ON CONFIG:
- When strategy="hybrid", the retriever passed to your function already does
  BM25+dense fusion internally. You do NOT need to implement BM25 yourself.
- When use_reranker=true, a cross-encoder reranker is applied AFTER your
  retrieve() function returns — you don't need to implement reranking.
- You CAN implement query rewriting by calling retriever.retrieve() multiple
  times with different queries and merging results.

## RULES
- Return the COMPLETE file contents (not a diff, not a snippet)
- Keep the function signature EXACTLY: retrieve(question, retriever, top_k) -> list[RetrievedDocument]
- You may add helper functions ABOVE retrieve()
- Allowed imports: src.retrieval.base, src.benchmark.loader, re, math, statistics, collections, itertools, functools
- NO imports of: os, sys, subprocess, socket, requests, urllib, pathlib, shutil
- Focus on ONE clear improvement per iteration
- The retriever ONLY has .retrieve(query, top_k). Do NOT call any other method.
- BenchmarkQuestion does NOT have .metadata. Use .question, .question_type, .source_types instead.

## OUTPUT FORMAT
Return your response in TWO sections separated by the line ===CONFIG===

First section: The complete Python code for pipeline.py (no markdown fences).

Then the line: ===CONFIG===

Then: A JSON object with retrieval config changes you want to make.

If you don't want to change the config, you may omit the ===CONFIG=== section entirely.
No markdown fences, no explanation, no commentary.
"""

_BASELINE_CODE = '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    return retriever.retrieve(question.question, top_k=top_k)
'''


def _candidate_models(fast: str, full: str) -> list[str]:
    """Deduplicated model list with reliable fallbacks."""
    ordered = [fast, full, "gemini-2.5-flash", "gemini-2.5-pro"]
    seen: set[str] = set()
    out: list[str] = []
    for m in ordered:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _format_code_history(state: ResearchLabState) -> str:
    """Format previous attempts with full code for recent entries so the LLM can learn."""
    history = state.get("code_history", [])
    if not history:
        return "No previous code attempts."

    lines: list[str] = []
    recent_cutoff = max(0, len(history) - 3)

    for i, entry in enumerate(history[-5:]):
        verdict = "ACCEPTED" if entry.get("accepted") else "REJECTED"
        score = entry.get("score", 0.0)
        hyp = entry.get("hypothesis", "")
        proposed_code = entry.get("proposed_code", "")
        proposed_config = entry.get("proposed_config")

        # Full detail for the last 3 entries, summary for older ones
        absolute_idx = len(history) - 5 + i
        if absolute_idx >= recent_cutoff and proposed_code:
            lines.append(f"### Attempt [{verdict}] score={score:.4f} — {hyp}")
            lines.append(f"```python\n{proposed_code}\n```")
            if proposed_config:
                lines.append(f"Config used: {json.dumps(proposed_config, indent=2)}")
        else:
            lines.append(f"  [{verdict}] score={score:.4f} — {hyp}")
            diff = entry.get("diff_summary", "")
            if diff:
                lines.append(f"    Changes: {diff[:500]}")
            if proposed_config:
                cfg_summary = {k: v for k, v in proposed_config.items()
                               if k in ("strategy", "top_k", "use_reranker", "bm25_weight", "dense_weight")}
                lines.append(f"    Config: {json.dumps(cfg_summary)}")

    return "\n\n".join(lines)


def _extract_text(content) -> str:
    """Pull plain text from an LLM response, ignoring extras/signature blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output, even if only one side appears."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return cleaned


def _parse_config_json(text: str, base: RetrievalConfig) -> RetrievalConfig:
    """Parse a config JSON fragment and merge into base config."""
    cleaned = _strip_markdown_fences(text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return base.model_copy(deep=True)
    if not isinstance(data, dict):
        return base.model_copy(deep=True)

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
    config.evaluation_mode = "fast"
    return config


def _parse_code_and_config(
    raw: str, base_config: RetrievalConfig,
) -> tuple[str, RetrievalConfig]:
    """Split LLM response into code and config."""
    if CONFIG_DELIMITER in raw:
        parts = raw.split(CONFIG_DELIMITER, 1)
        code = _strip_markdown_fences(parts[0].strip())
        config = _parse_config_json(parts[1], base_config)
    else:
        code = _strip_markdown_fences(raw)
        config = base_config.model_copy(deep=True)
    return code, config


def _fallback_code(state: ResearchLabState) -> str:
    """Return a simple perturbation of the current code when LLM fails."""
    current = state.get("current_pipeline_code", "")
    if not current:
        return _BASELINE_CODE

    if "relevant policy procedure documentation" not in current:
        return '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    query = question.question
    docs = retriever.retrieve(query, top_k=top_k)

    if question.question_type in ("semantic", "multi_hop", "comparison"):
        expanded = f"{query} relevant policy procedure documentation"
        extra_docs = retriever.retrieve(expanded, top_k=top_k)
        seen = {d.document_id: d for d in docs}
        for d in extra_docs:
            if d.document_id not in seen or d.score > seen[d.document_id].score:
                seen[d.document_id] = d
        docs = sorted(seen.values(), key=lambda d: d.score, reverse=True)

    return docs[:top_k]
'''
    if "effective_top_k" not in current:
        return '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    query = question.question
    effective_top_k = max(1, min(top_k, 5))
    if question.question_type in ("multi_hop", "comparison"):
        effective_top_k = top_k

    docs = retriever.retrieve(query, top_k=effective_top_k)
    return docs[:top_k]
'''
    return current


async def code_planner_agent(state: ResearchLabState) -> ResearchLabState:
    """Generate new pipeline.py code and retrieval config based on hypothesis and data."""
    settings = get_settings()
    state["current_phase"] = "code_planner"

    if not state["hypotheses"]:
        return state

    latest_hypothesis = state["hypotheses"][-1]
    current_code = state.get("current_pipeline_code", _BASELINE_CODE)
    best_score = state.get("best_score", -1.0)

    baseline_config = state.get("best_config") or RetrievalConfig(
        strategy="dense",
        embedding_model=settings.embedding_model,
        top_k=settings.default_top_k,
        use_reranker=False,
    )

    failure_ex = state.get("failure_examples", [])
    failure_text = json.dumps(failure_ex[:5], indent=2) if failure_ex else "None yet."

    prompt = CODE_PLANNER_PROMPT.format(
        current_code=current_code,
        current_score=f"{best_score:.4f}" if best_score >= 0 else "not measured",
        current_config=format_config_for_karpathy(state),
        per_type_summary=state.get("per_type_summary") or "No breakdown yet.",
        failure_examples=failure_text,
        code_history=_format_code_history(state),
        hypothesis=f"{latest_hypothesis.title}: {latest_hypothesis.rationale}",
    )

    proposed_code = ""
    proposed_config = baseline_config.model_copy(deep=True)
    rationale = ""

    if settings.has_google_key:
        for model_name in _candidate_models(settings.gemini_fast_model, settings.gemini_model):
            if not model_name:
                continue
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=settings.google_api_key,
                    temperature=0.4,
                )
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                raw = _extract_text(response.content)
                proposed_code, proposed_config = _parse_code_and_config(raw, baseline_config)
                rationale = f"Code+config generated by {model_name} for: {latest_hypothesis.title}"
                logger.info(
                    "Karpathy code planner raw response "
                    "run_id=%s iteration=%s model=%s hypothesis=%r\n%s",
                    state.get("run_id"),
                    state.get("iteration"),
                    model_name,
                    latest_hypothesis.title,
                    raw,
                )
                logger.info(
                    "Karpathy code planner parsed code "
                    "run_id=%s iteration=%s\n%s",
                    state.get("run_id"),
                    state.get("iteration"),
                    proposed_code,
                )
                logger.info(
                    "Karpathy code planner parsed config "
                    "run_id=%s iteration=%s\n%s",
                    state.get("run_id"),
                    state.get("iteration"),
                    proposed_config.model_dump_json(indent=2),
                )
                break
            except Exception as exc:
                logger.warning("Code planner LLM call failed (%s): %s", model_name, exc)
                continue

    if not proposed_code:
        proposed_code = _fallback_code(state)
        proposed_config = baseline_config.model_copy(deep=True)
        rationale = f"Fallback code perturbation for: {latest_hypothesis.title}"
        logger.info(
            "Karpathy code planner fallback candidate "
            "run_id=%s iteration=%s hypothesis=%r\n%s",
            state.get("run_id"),
            state.get("iteration"),
            latest_hypothesis.title,
            proposed_code,
        )

    if proposed_code.strip() == current_code.strip():
        proposed_code = _fallback_code(state)
        rationale = f"Fallback no-op replacement for: {latest_hypothesis.title}"
        logger.info(
            "Karpathy code planner replaced no-op candidate "
            "run_id=%s iteration=%s hypothesis=%r\n%s",
            state.get("run_id"),
            state.get("iteration"),
            latest_hypothesis.title,
            proposed_code,
        )

    proposed_config.embedding_model = settings.embedding_model

    state["proposed_code"] = proposed_code
    state["proposed_config"] = proposed_config
    state["planner_rationale"] = rationale

    spec = ExperimentSpec(
        id=f"exp_{uuid4().hex[:8]}",
        hypothesis_id=latest_hypothesis.id,
        name=f"[Karpathy] {latest_hypothesis.title[:40]}",
        description=f"Code+config edit: {rationale}",
        retrieval_config=proposed_config,
        run_id=state.get("run_id"),
        run_position=state["iteration"] + 1,
    )
    state["experiment_queue"] = state["experiment_queue"] + [spec]
    return state
