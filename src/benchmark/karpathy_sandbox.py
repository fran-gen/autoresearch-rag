from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from src.benchmark.loader import EnterpriseRagBenchLoader
from src.benchmark.metrics import compute_retrieval_metrics
from src.config import get_settings
from src.models import BenchmarkMetrics, QuestionResult, RetrievalConfig
from src.retrieval.base import BaseRetriever, RetrievedDocument
from src.retrieval.embeddings import EmbeddingEncoder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.qdrant_dense import QdrantDenseRetriever, dense_records_from_documents

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[object, BaseRetriever, int], list[RetrievedDocument]]


def _build_retriever(
    documents: list,
    config: RetrievalConfig,
    dense: QdrantDenseRetriever,
) -> BaseRetriever:
    """Build the appropriate retriever based on config strategy."""
    if config.strategy == "hybrid":
        records = dense_records_from_documents(documents)
        return HybridRetriever(
            dense_retriever=dense,
            records=records,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
        )
    return dense


def _build_reranker(config: RetrievalConfig):
    """Build a reranker if the config requests one."""
    if config.use_reranker and config.reranker_model:
        try:
            from src.retrieval.reranker import CrossEncoderReranker
            return CrossEncoderReranker(config.reranker_model)
        except Exception as exc:
            logger.warning("Failed to load reranker %s: %s", config.reranker_model, exc)
    return None


def run_karpathy_benchmark(
    retrieve_fn: RetrieveFn,
    *,
    benchmark_root: str | Path | None,
    question_focus: str,
    config: RetrievalConfig | None,
) -> tuple[list[QuestionResult], BenchmarkMetrics]:
    """Run retrieval benchmark against the supplied pipeline retrieve function."""
    settings = get_settings()
    root = Path(benchmark_root) if benchmark_root else settings.benchmark_root
    loader = EnterpriseRagBenchLoader(root)
    documents, questions = loader.load_mvp_subset()

    focus = (question_focus or "all").strip()
    if focus and focus != "all":
        filtered = [q for q in questions if (q.question_type or "unknown") == focus]
        questions = filtered or questions

    active_config = config or RetrievalConfig(
        strategy="dense",
        embedding_model=settings.embedding_model,
        top_k=settings.default_top_k,
        use_reranker=False,
        evaluation_mode="fast",
    )
    active_config.embedding_model = settings.embedding_model

    encoder = EmbeddingEncoder(active_config.embedding_model)
    dense = QdrantDenseRetriever(encoder=encoder, qdrant_path=settings.qdrant_path)
    records = dense_records_from_documents(documents)
    query_dim = len(encoder.encode(["dimension probe"])[0])
    existing_dim = dense.collection_vector_size()

    if existing_dim is None or existing_dim != query_dim:
        dense.build(records)
    else:
        dense.load()

    retriever = _build_retriever(documents, active_config, dense)
    reranker = _build_reranker(active_config)

    logger.info(
        "Karpathy benchmark: strategy=%s top_k=%s reranker=%s hybrid=%s",
        active_config.strategy,
        active_config.top_k,
        bool(reranker),
        isinstance(retriever, HybridRetriever),
    )

    top_k = active_config.top_k
    question_results: list[QuestionResult] = []

    for question in questions:
        start = time.perf_counter()
        try:
            docs = retrieve_fn(
                question, retriever, top_k,
                encoder=encoder, reranker=reranker,
            )
        except Exception:
            docs = []

        elapsed_ms = (time.perf_counter() - start) * 1000
        question_results.append(
            QuestionResult(
                question_id=question.question_id,
                answer="",
                document_ids=[doc.document_id for doc in docs],
                document_scores=[doc.score for doc in docs],
                latency_ms=elapsed_ms,
            )
        )

    ground_truth = {
        question.question_id: set(question.expected_doc_ids)
        for question in questions
        if question.expected_doc_ids
    }
    metrics = compute_retrieval_metrics(
        question_results=question_results,
        ground_truth_by_question=ground_truth,
        top_k=top_k,
        answer_facts_by_question=None,
        retrieval_only=True,
    )
    return question_results, metrics
