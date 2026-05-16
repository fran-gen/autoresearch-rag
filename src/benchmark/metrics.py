from __future__ import annotations

import re
from statistics import mean

from src.models import BenchmarkMetrics, QuestionResult


def compute_fact_completeness(answer: str, answer_facts: list[str]) -> float:
    """Approximate completeness: fraction of gold answer_facts mentioned in the answer.

    Uses simple substring matching as a fast proxy. For production-quality
    evaluation, use the EnterpriseRAG-Bench LLM-judged completeness scorer.
    """
    if not answer_facts:
        return 0.0
    matched = sum(
        1
        for fact in answer_facts
        if any(
            token.lower() in answer.lower()
            for token in fact.split()
            if len(token) > 4
        )
    )
    return matched / len(answer_facts)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def compute_answer_overlap(generated: str, gold: str) -> float:
    """Token-level F1 between generated and gold answers."""
    gen_tokens = set(tokenize(generated))
    gold_tokens = set(tokenize(gold))

    if not gen_tokens or not gold_tokens:
        return 0.0

    common = gen_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_latency_score(latency_ms: float, budget_ms: float = 10_000.0) -> float:
    """Latency score in [0, 1]; 1 is fastest, 0 at or above budget."""
    if budget_ms <= 0:
        return 0.0
    return max(0.0, 1.0 - (latency_ms / budget_ms))


def compute_fast_composite(
    recall: float | None,
    precision: float | None,
    answer_overlap: float,
    latency_score: float,
) -> float:
    """Composite when retrieval + overlap + latency are all available."""
    if recall is not None and precision is not None:
        return (
            0.50 * recall
            + 0.20 * precision
            + 0.20 * answer_overlap
            + 0.10 * latency_score
        )
    return 0.70 * answer_overlap + 0.30 * latency_score


def compute_retrieval_metrics(
    question_results: list[QuestionResult],
    ground_truth_by_question: dict[str, set[str]],
    top_k: int,
    answer_facts_by_question: dict[str, list[str]] | None = None,
    retrieval_only: bool = False,
) -> BenchmarkMetrics:
    if not question_results:
        return BenchmarkMetrics()

    recalls: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    correctness_values: list[float] = []
    completeness_values: list[float] = []
    invalid_extra_counts: list[int] = []

    for result in question_results:
        pred_ids = result.document_ids[:top_k]
        gt_ids = ground_truth_by_question.get(result.question_id, set())
        pred_set = set(pred_ids)
        latencies.append(result.latency_ms)

        if not gt_ids:
            continue

        overlap = pred_set.intersection(gt_ids)
        recall = len(overlap) / len(gt_ids)
        precision = len(overlap) / len(pred_set) if pred_set else 0.0
        recalls.append(recall)
        precisions.append(precision)

        if result.is_correct is not None:
            correctness_values.append(1.0 if result.is_correct else 0.0)

        if answer_facts_by_question and not retrieval_only:
            facts = answer_facts_by_question.get(result.question_id, [])
            if facts:
                completeness_values.append(
                    compute_fact_completeness(result.answer, facts)
                )

        invalid_extra = len(pred_set - gt_ids)
        invalid_extra_counts.append(invalid_extra)

    invalid_extra_rate = (
        len([c for c in invalid_extra_counts if c > 0]) / len(invalid_extra_counts)
        if invalid_extra_counts
        else 0.0
    )

    correctness = mean(correctness_values) if correctness_values else None
    if correctness is None and completeness_values:
        correctness = mean(completeness_values)

    if retrieval_only:
        answered = len([r for r in question_results if r.document_ids])
    else:
        answered = len([r for r in question_results if r.answer.strip()])

    return BenchmarkMetrics(
        total_questions=len(question_results),
        answered_questions=answered,
        recall_at_k=mean(recalls) if recalls else 0.0,
        precision_at_k=mean(precisions) if precisions else 0.0,
        answer_correctness=correctness,
        avg_latency_ms=mean(latencies) if latencies else 0.0,
        invalid_extra_docs_rate=invalid_extra_rate,
    )


def composite_score(metrics: BenchmarkMetrics) -> float:
    """Weighted score for leaderboard ranking.

    Full tier (answer_correctness present): correctness 40%, recall 30%, precision 20%,
    invalid-extra penalty 10%.

    Fast tier (answer_correctness is None): recall 50%, precision 30%, invalid-extra 20%.
    """
    if metrics.answer_correctness is None:
        weights = {
            "recall_at_k": 0.5,
            "precision_at_k": 0.3,
            "invalid_extra_docs_rate": 0.2,
        }
        values = {
            "recall_at_k": metrics.recall_at_k,
            "precision_at_k": metrics.precision_at_k,
            "invalid_extra_docs_rate": 1.0 - metrics.invalid_extra_docs_rate,
        }
    else:
        weights = {
            "answer_correctness": 0.4,
            "recall_at_k": 0.3,
            "precision_at_k": 0.2,
            "invalid_extra_docs_rate": 0.1,
        }
        values = {
            "answer_correctness": metrics.answer_correctness,
            "recall_at_k": metrics.recall_at_k,
            "precision_at_k": metrics.precision_at_k,
            "invalid_extra_docs_rate": 1.0 - metrics.invalid_extra_docs_rate,
        }

    available = {name: value for name, value in values.items() if value is not None}
    if not available:
        return 0.0

    total_weight = sum(weights[name] for name in available)
    return sum(weights[name] * available[name] for name in available) / total_weight
