"""Mutable retrieval pipeline -- the ONLY file the Karpathy agent edits.

The function signature below is a fixed contract. The agent may change the
body and add helper functions, but MUST NOT alter the signature or return type.
"""

from __future__ import annotations

from src.benchmark.loader import BenchmarkQuestion
from src.retrieval.base import BaseRetriever, RetrievedDocument


def retrieve(
    question: BenchmarkQuestion,
    retriever: BaseRetriever,
    top_k: int,
) -> list[RetrievedDocument]:
    """Retrieve the most relevant documents for the given question.

    Args:
        question: The benchmark question (has .question, .question_type, .source_types)
        retriever: A retriever with .retrieve(query, top_k) -> list[RetrievedDocument]
        top_k: Maximum number of documents to return

    Returns:
        Ordered list of RetrievedDocument (best first), length <= top_k
    """
    return retriever.retrieve(question.question, top_k=top_k)
