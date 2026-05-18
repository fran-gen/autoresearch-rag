"""Code Planner for Karpathy mode -- generates pipeline.py code AND config via LLM."""

from __future__ import annotations

import ast
import json
import logging
import re
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.history import format_config_for_karpathy
from src.agents.karpathy_validation import validate_karpathy_code
from src.agents.state import ResearchLabState
from src.agents.text_utils import extract_text_content
from src.config import get_google_api_key, get_settings
from src.models import ExperimentSpec, RetrievalConfig

logger = logging.getLogger(__name__)

CONFIG_DELIMITER = "===CONFIG==="
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<body>[\s\S]*?)```", re.IGNORECASE)

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
- Every iteration must include a config move. Change at least one concrete
  lever: top_k, strategy, use_reranker, bm25_weight/dense_weight,
  query_rewrite, or source_diversity.
- Do not repeat the current or recently rejected config unless the code change
  fundamentally changes how that same config is used.
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
- Comments, docstrings, and formatting-only edits do NOT count as code changes.
- Every Karpathy iteration MUST include a meaningful retrieve() behavior change.
- Config changes may accompany code changes, but config-only candidates are not
  valid in Karpathy mode. Comments are fine when they accompany a real behavior
  change.

## OUTPUT FORMAT
Return your response in TWO sections separated by the line ===CONFIG===

First section: The complete Python code for pipeline.py (no markdown fences).

Then the line: ===CONFIG===

Then: A JSON object with retrieval config changes you want to make.

Prefer always including the ===CONFIG=== section. Omit it only when the code
change deliberately requires the current config.
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
        reason = entry.get("reason") or entry.get("failure_analysis") or ""
        proposed_code = entry.get("proposed_code", "")
        proposed_config = entry.get("proposed_config")

        # Full detail for the last 3 entries, summary for older ones
        absolute_idx = len(history) - 5 + i
        if absolute_idx >= recent_cutoff and proposed_code:
            lines.append(f"### Attempt [{verdict}] score={score:.4f} — {hyp}")
            if reason:
                lines.append(f"Outcome: {reason}")
            lines.append(f"```python\n{proposed_code}\n```")
            if proposed_config:
                lines.append(f"Config used: {json.dumps(proposed_config, indent=2)}")
        else:
            lines.append(f"  [{verdict}] score={score:.4f} — {hyp}")
            if reason:
                lines.append(f"    Reason: {reason[:500]}")
            diff = entry.get("diff_summary", "")
            if diff:
                lines.append(f"    Changes: {diff[:500]}")
            if proposed_config:
                cfg_summary = {k: v for k, v in proposed_config.items()
                               if k in ("strategy", "top_k", "use_reranker", "bm25_weight", "dense_weight")}
                lines.append(f"    Config: {json.dumps(cfg_summary)}")

    return "\n\n".join(lines)


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


def _looks_like_config_json(text: str) -> bool:
    cleaned = _strip_markdown_fences(text.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _valid_python_prefix(text: str) -> tuple[str, str]:
    """Split off trailing non-Python, such as config JSON after omitted delimiter."""
    lines = text.strip().splitlines()

    for end in range(1, len(lines)):
        candidate = "\n".join(lines[:end]).strip()
        remainder = "\n".join(lines[end:]).strip()
        if (
            remainder.startswith("{")
            and _looks_like_config_json(remainder)
            and validate_karpathy_code(candidate) is None
        ):
            return candidate, remainder

    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if validate_karpathy_code(candidate) is None:
            remainder = "\n".join(lines[end:]).strip()
            return candidate, remainder
    return text.strip(), ""


def _extract_fenced_code(raw: str) -> tuple[str, str] | None:
    """Return the first valid fenced Python block and following text, if any."""
    for match in CODE_FENCE_RE.finditer(raw):
        code = match.group("body").strip()
        if validate_karpathy_code(code) is None:
            return code, raw[match.end():].strip()
    return None


def _parse_code_and_config(
    raw: str, base_config: RetrievalConfig,
) -> tuple[str, RetrievalConfig]:
    """Split LLM response into code and config."""
    if CONFIG_DELIMITER in raw:
        parts = raw.split(CONFIG_DELIMITER, 1)
        code = _strip_markdown_fences(parts[0].strip())
        config = _parse_config_json(parts[1], base_config)
    else:
        fenced = _extract_fenced_code(raw)
        if fenced:
            code, config_text = fenced
        else:
            code, config_text = _valid_python_prefix(_strip_markdown_fences(raw))
        config = (
            _parse_config_json(config_text, base_config)
            if config_text
            else base_config.model_copy(deep=True)
        )
    return code, config


def _config_fingerprint(config: RetrievalConfig) -> str:
    payload = config.model_dump(exclude={"embedding_model", "evaluation_mode"})
    return json.dumps(payload, sort_keys=True, default=str)


def _historical_config_fingerprints(state: ResearchLabState) -> set[str]:
    fingerprints: set[str] = set()
    for entry in state.get("score_history", []):
        cfg = entry.get("config") if isinstance(entry, dict) else None
        if isinstance(cfg, dict):
            try:
                fingerprints.add(_config_fingerprint(RetrievalConfig.model_validate(cfg)))
            except Exception:
                pass
    for entry in state.get("code_history", []):
        cfg = entry.get("proposed_config") if isinstance(entry, dict) else None
        if isinstance(cfg, dict):
            try:
                fingerprints.add(_config_fingerprint(RetrievalConfig.model_validate(cfg)))
            except Exception:
                pass
    return fingerprints


def _config_is_repeated(
    state: ResearchLabState,
    candidate: RetrievalConfig,
    baseline: RetrievalConfig,
) -> bool:
    candidate_fp = _config_fingerprint(candidate)
    if candidate_fp == _config_fingerprint(baseline):
        return True
    return candidate_fp in _historical_config_fingerprints(state)


def _with_config_move(
    base: RetrievalConfig,
    *,
    strategy: str | None = None,
    top_k: int | None = None,
    use_reranker: bool | None = None,
    bm25_weight: float | None = None,
    dense_weight: float | None = None,
    query_rewrite: bool | None = None,
    source_diversity: bool | None = None,
) -> RetrievalConfig:
    config = base.model_copy(deep=True)
    if strategy is not None:
        config.strategy = strategy
    if top_k is not None:
        config.top_k = max(3, min(20, int(top_k)))
    if use_reranker is not None:
        config.use_reranker = use_reranker
        config.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None
    if bm25_weight is not None:
        config.bm25_weight = max(0.0, min(1.0, bm25_weight))
    if dense_weight is not None:
        config.dense_weight = max(0.0, min(1.0, dense_weight))
    config.extra = dict(config.extra or {})
    if query_rewrite is not None:
        config.extra["query_rewrite"] = query_rewrite
    if source_diversity is not None:
        config.extra["source_diversity"] = source_diversity
    config.evaluation_mode = "fast"
    return config


def _fallback_config(state: ResearchLabState, base: RetrievalConfig) -> RetrievalConfig:
    moves = [
        _with_config_move(
            base,
            strategy="hybrid",
            top_k=max(base.top_k, 16),
            use_reranker=True,
            bm25_weight=0.35,
            dense_weight=0.65,
            query_rewrite=True,
        ),
        _with_config_move(
            base,
            strategy="hybrid",
            top_k=max(base.top_k, 14),
            use_reranker=False,
            bm25_weight=0.70,
            dense_weight=0.30,
            source_diversity=True,
        ),
        _with_config_move(
            base,
            strategy="dense",
            top_k=max(base.top_k, 18),
            use_reranker=True,
            query_rewrite=False,
            source_diversity=False,
        ),
        _with_config_move(
            base,
            strategy="hybrid",
            top_k=20,
            use_reranker=True,
            bm25_weight=0.55,
            dense_weight=0.45,
            query_rewrite=True,
            source_diversity=True,
        ),
    ]
    avoided = _historical_config_fingerprints(state) | {_config_fingerprint(base)}
    start = len(state.get("code_history", [])) % len(moves)
    for config in moves[start:] + moves[:start]:
        if _config_fingerprint(config) not in avoided:
            return config
    return moves[start]


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """Remove docstrings so comment/docs-only candidates are treated as no-ops."""
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)

    body = getattr(node, "body", None)
    if not isinstance(body, list) or not body:
        return node

    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        del body[0]
    return node


def _semantic_fingerprint(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return ast.dump(_strip_docstrings(tree), include_attributes=False)


def _is_semantic_noop(proposed_code: str, current_code: str) -> bool:
    if proposed_code.strip() == current_code.strip():
        return True

    proposed_fp = _semantic_fingerprint(proposed_code)
    current_fp = _semantic_fingerprint(current_code)
    return proposed_fp is not None and proposed_fp == current_fp


def _fallback_code(state: ResearchLabState) -> str:
    """Return a simple perturbation of the current code when LLM fails."""
    current = state.get("current_pipeline_code", "")
    if not current:
        return _BASELINE_CODE

    candidates = [
        '''\
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
''',
        '''\
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
''',
        '''\
from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def _merge_ranked(primary: list[RetrievedDocument], secondary: list[RetrievedDocument]) -> list[RetrievedDocument]:
    seen: dict[str, RetrievedDocument] = {}
    for doc in primary + secondary:
        existing = seen.get(doc.document_id)
        if existing is None or doc.score > existing.score:
            seen[doc.document_id] = doc
    return sorted(seen.values(), key=lambda d: d.score, reverse=True)


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    query = question.question
    docs = retriever.retrieve(query, top_k=top_k)

    source_terms = " ".join(str(source).replace("_", " ") for source in question.source_types)
    if source_terms:
        source_docs = retriever.retrieve(f"{query} {source_terms}", top_k=top_k)
        docs = _merge_ranked(docs, source_docs)

    return docs[:top_k]
''',
    ]

    for candidate in candidates:
        if not _is_semantic_noop(candidate, current):
            return candidate

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
                    google_api_key=get_google_api_key(),
                    temperature=0.4,
                )
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                raw = extract_text_content(response.content)
                candidate_code, candidate_config = _parse_code_and_config(raw, baseline_config)
                logger.info(
                    "Karpathy code planner raw response "
                    "run_id=%s iteration=%s model=%s hypothesis=%r\n%s",
                    state.get("run_id"),
                    state.get("iteration"),
                    model_name,
                    latest_hypothesis.title,
                    raw,
                )
                validation_error = validate_karpathy_code(candidate_code)
                if validation_error:
                    logger.warning(
                        "Karpathy code planner discarded invalid candidate "
                        "run_id=%s iteration=%s model=%s error=%s",
                        state.get("run_id"),
                        state.get("iteration"),
                        model_name,
                        validation_error,
                    )
                    continue
                proposed_code = candidate_code
                proposed_config = candidate_config
                rationale = f"Code+config generated by {model_name} for: {latest_hypothesis.title}"
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
        proposed_config = _fallback_config(state, baseline_config)
        rationale = f"Fallback code+config perturbation for: {latest_hypothesis.title}"
        logger.info(
            "Karpathy code planner fallback candidate "
            "run_id=%s iteration=%s hypothesis=%r\n%s",
            state.get("run_id"),
            state.get("iteration"),
            latest_hypothesis.title,
            proposed_code,
        )

    if _is_semantic_noop(proposed_code, current_code):
        proposed_code = _fallback_code(state)
        if _config_is_repeated(state, proposed_config, baseline_config):
            proposed_config = _fallback_config(state, baseline_config)
        rationale = f"Fallback no-op code replacement for: {latest_hypothesis.title}"
        logger.info(
            "Karpathy code planner replaced no-op/config-only candidate "
            "run_id=%s iteration=%s hypothesis=%r\n%s",
            state.get("run_id"),
            state.get("iteration"),
            latest_hypothesis.title,
            proposed_code,
        )

    if _config_is_repeated(state, proposed_config, baseline_config):
        proposed_config = _fallback_config(state, baseline_config)
        rationale = f"{rationale}; forced config exploration to avoid repeated settings"
        logger.info(
            "Karpathy code planner forced config exploration "
            "run_id=%s iteration=%s config=%s",
            state.get("run_id"),
            state.get("iteration"),
            proposed_config.model_dump_json(),
        )

    validation_error = validate_karpathy_code(proposed_code)
    if validation_error:
        logger.warning(
            "Karpathy code planner final candidate was invalid; using fallback "
            "run_id=%s iteration=%s error=%s",
            state.get("run_id"),
            state.get("iteration"),
            validation_error,
        )
        proposed_code = _fallback_code(state)
        if validate_karpathy_code(proposed_code) is not None:
            proposed_code = _BASELINE_CODE
        rationale = f"Fallback valid code replacement for: {latest_hypothesis.title}"

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
