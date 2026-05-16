from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.benchmark.loader import BenchmarkDocument, BenchmarkQuestion
from src.benchmark.metrics import compute_retrieval_metrics
from src.config import get_google_api_key, get_settings
from src.models import BenchmarkMetrics, QuestionResult, RetrievalConfig
from src.retrieval.base import RetrievedDocument
from src.retrieval.embeddings import EmbeddingEncoder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.qdrant_dense import QdrantDenseRetriever, dense_records_from_documents
from src.retrieval.reranker import CrossEncoderReranker


class BenchmarkRunner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm_by_model: dict[str, ChatGoogleGenerativeAI] = {}

    def _candidate_models(self) -> list[str]:
        ordered = [
            self.settings.gemini_fast_model,
            self.settings.gemini_model,
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
        seen: set[str] = set()
        models: list[str] = []
        for model in ordered:
            if model and model not in seen:
                seen.add(model)
                models.append(model)
        return models

    def _get_llm_for_model(self, model_name: str) -> ChatGoogleGenerativeAI | None:
        if not self.settings.has_google_key:
            return None
        if model_name in self._llm_by_model:
            return self._llm_by_model[model_name]
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=get_google_api_key(),
            temperature=0,
        )
        self._llm_by_model[model_name] = llm
        return llm

    def _build_retriever(
        self,
        documents: list[BenchmarkDocument],
        config: RetrievalConfig,
    ) -> QdrantDenseRetriever | HybridRetriever:
        encoder = EmbeddingEncoder(config.embedding_model)
        dense = QdrantDenseRetriever(
            encoder=encoder,
            qdrant_path=self.settings.qdrant_path,
            qdrant_url=self.settings.qdrant_url,
        )
        records = dense_records_from_documents(documents)
        query_dim = len(encoder.encode(["dimension probe"])[0])
        existing_dim = dense.collection_vector_size()

        if existing_dim is None:
            dense.build(records)
        elif existing_dim != query_dim:
            dense.build(records)
        else:
            dense.load()

        if config.strategy == "hybrid":
            return HybridRetriever(
                dense_retriever=dense,
                records=records,
                bm25_weight=config.bm25_weight,
                dense_weight=config.dense_weight,
            )
        return dense

    def _answer_with_context(self, question: str, docs: list[RetrievedDocument]) -> str:
        if not self.settings.has_google_key:
            if not docs:
                return "Insufficient context found."
            return docs[0].text[:600]

        context = "\n\n---\n\n".join(
            f"Document ID: {d.document_id}\n{d.text[:1200]}" for d in docs
        )
        prompt = (
            "You are evaluating enterprise knowledge QA.\n"
            "Use ONLY the provided context and be concise.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        for model_name in self._candidate_models():
            llm = self._get_llm_for_model(model_name)
            if llm is None:
                continue
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                return str(response.content)
            except Exception:
                continue

        if not docs:
            return "Insufficient context found."
        return docs[0].text[:600]

    def _effective_top_k(self, question: BenchmarkQuestion, config: RetrievalConfig) -> int:
        overrides = config.extra.get("question_type_overrides") if isinstance(config.extra, dict) else None
        if isinstance(overrides, dict):
            qtype_cfg = overrides.get(question.question_type or "unknown")
            if isinstance(qtype_cfg, dict) and isinstance(qtype_cfg.get("top_k"), (int, float)):
                return max(1, min(30, int(qtype_cfg["top_k"])))
        return config.top_k

    def _query_variants(self, question: BenchmarkQuestion, config: RetrievalConfig) -> list[str]:
        if not isinstance(config.extra, dict) or not config.extra.get("query_rewrite"):
            return [question.question]
        qtype = question.question_type or "unknown"
        variants = [question.question]
        variants.append(f"{question.question} relevant policy procedure source document")
        if qtype in {"semantic", "multi_hop", "comparison", "analytical"}:
            variants.append(f"Find evidence needed to answer: {question.question}")
        return list(dict.fromkeys(v.strip() for v in variants if v.strip()))[:3]

    def _diversify_docs(self, docs: list[RetrievedDocument], top_k: int) -> list[RetrievedDocument]:
        selected: list[RetrievedDocument] = []
        seen_sources: set[str] = set()
        for doc in docs:
            source = str(doc.metadata.get("source_type") or doc.document_id)
            if source in seen_sources and len(selected) < max(2, top_k // 2):
                continue
            selected.append(doc)
            seen_sources.add(source)
            if len(selected) >= top_k:
                return selected
        for doc in docs:
            if doc not in selected:
                selected.append(doc)
            if len(selected) >= top_k:
                break
        return selected

    def _retrieve_question(
        self,
        retriever: QdrantDenseRetriever | HybridRetriever,
        question: BenchmarkQuestion,
        config: RetrievalConfig,
        reranker: CrossEncoderReranker | None,
    ) -> list[RetrievedDocument]:
        top_k = self._effective_top_k(question, config)
        by_doc: dict[str, RetrievedDocument] = {}
        for query in self._query_variants(question, config):
            for doc in retriever.retrieve(query, top_k=top_k):
                existing = by_doc.get(doc.document_id)
                if existing is None or doc.score > existing.score:
                    by_doc[doc.document_id] = doc
        docs = sorted(by_doc.values(), key=lambda doc: doc.score, reverse=True)
        if reranker is not None:
            docs = reranker.rerank(question.question, docs, top_k=max(top_k, len(docs)))
        if isinstance(config.extra, dict) and config.extra.get("source_diversity"):
            docs = self._diversify_docs(docs, top_k)
        return docs[:top_k]

    def run_fast(
        self,
        documents: list[BenchmarkDocument],
        questions: list[BenchmarkQuestion],
        retrieval_config: RetrievalConfig,
        output_path: Path | None = None,
    ) -> tuple[list[QuestionResult], BenchmarkMetrics]:
        """Retrieval-only evaluation (no LLM answer generation)."""
        retriever = self._build_retriever(documents, retrieval_config)
        reranker = (
            CrossEncoderReranker(retrieval_config.reranker_model)
            if retrieval_config.use_reranker and retrieval_config.reranker_model
            else None
        )

        question_results: list[QuestionResult] = []
        for question in questions:
            start = time.perf_counter()
            docs = self._retrieve_question(retriever, question, retrieval_config, reranker)
            elapsed_ms = (time.perf_counter() - start) * 1000
            question_results.append(
                QuestionResult(
                    question_id=question.question_id,
                    answer="",
                    document_ids=[d.document_id for d in docs],
                    latency_ms=elapsed_ms,
                )
            )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                for row in question_results:
                    file.write(
                        json.dumps(
                            {
                                "question_id": row.question_id,
                                "answer": row.answer,
                                "document_ids": row.document_ids,
                            }
                        )
                    )
                    file.write("\n")

        gt = {
            q.question_id: set(q.expected_doc_ids)
            for q in questions
            if q.expected_doc_ids
        }
        metrics = compute_retrieval_metrics(
            question_results=question_results,
            ground_truth_by_question=gt,
            top_k=retrieval_config.top_k,
            answer_facts_by_question=None,
            retrieval_only=True,
        )
        return question_results, metrics

    def run(
        self,
        documents: list[BenchmarkDocument],
        questions: list[BenchmarkQuestion],
        retrieval_config: RetrievalConfig,
        output_path: Path | None = None,
    ) -> tuple[list[QuestionResult], BenchmarkMetrics]:
        retriever = self._build_retriever(documents, retrieval_config)
        reranker = (
            CrossEncoderReranker(retrieval_config.reranker_model)
            if retrieval_config.use_reranker and retrieval_config.reranker_model
            else None
        )

        question_results: list[QuestionResult] = []
        for question in questions:
            start = time.perf_counter()
            docs = self._retrieve_question(retriever, question, retrieval_config, reranker)
            answer = self._answer_with_context(question.question, docs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            question_results.append(
                QuestionResult(
                    question_id=question.question_id,
                    answer=answer,
                    document_ids=[d.document_id for d in docs],
                    latency_ms=elapsed_ms,
                )
            )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                for row in question_results:
                    file.write(
                        json.dumps(
                            {
                                "question_id": row.question_id,
                                "answer": row.answer,
                                "document_ids": row.document_ids,
                            }
                        )
                    )
                    file.write("\n")

        gt = {
            q.question_id: set(q.expected_doc_ids)
            for q in questions
            if q.expected_doc_ids
        }
        answer_facts = {
            q.question_id: q.answer_facts
            for q in questions
            if q.answer_facts
        }
        metrics = compute_retrieval_metrics(
            question_results=question_results,
            ground_truth_by_question=gt,
            top_k=retrieval_config.top_k,
            answer_facts_by_question=answer_facts,
            retrieval_only=False,
        )
        return question_results, metrics
