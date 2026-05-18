"""Mutable retrieval pipeline for the Karpathy agent.

The agent edits both this file's code AND a structured RetrievalConfig that
controls strategy (dense/hybrid), top_k, BM25/dense weights, and reranker
settings.  The function signature below is a fixed contract -- the agent may
change the body and add helper functions, but MUST NOT alter the positional
parameters or return type.  The keyword-only ``encoder`` and ``reranker``
arguments are optional tools the agent may use.
"""

from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument
from src.retrieval.embeddings import EmbeddingEncoder
from src.retrieval.reranker import CrossEncoderReranker


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
    *,
    encoder: EmbeddingEncoder | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[RetrievedDocument]:
    """Retrieve the most relevant documents for the given question.

    Args:
        question: The benchmark question (has .question, .question_type, .source_types)
        retriever: A retriever with .retrieve(query, top_k) -> list[RetrievedDocument]
        top_k: Maximum number of documents to return
        encoder: Optional embedding encoder for custom similarity computations
        reranker: Optional cross-encoder reranker

    Returns:
        Ordered list of RetrievedDocument (best first), length <= top_k
    """
    return retriever.retrieve(question.question, top_k=top_k)
