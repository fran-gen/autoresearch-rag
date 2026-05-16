"""Dense retrieval backed by a persistent on-disk Qdrant collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.benchmark.loader import BenchmarkDocument
from src.retrieval.base import BaseRetriever, RetrievedDocument
from src.retrieval.dense import DenseRecord
from src.retrieval.embeddings import EmbeddingEncoder

DEFAULT_COLLECTION_NAME = "enterprise_docs"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class QdrantDenseRetriever(BaseRetriever):
    """Query a local Qdrant index built with BGE embeddings (cosine, 768-dim)."""

    def __init__(
        self,
        encoder: EmbeddingEncoder | None,
        qdrant_path: Path,
        qdrant_url: str = "",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.encoder = encoder
        self.qdrant_path = Path(qdrant_path)
        self.qdrant_url = qdrant_url.strip()
        self.collection_name = collection_name
        self._client: QdrantClient | None = None

    def _ensure_client(self) -> QdrantClient:
        if self._client is None:
            if self.qdrant_url:
                self._client = QdrantClient(url=self.qdrant_url)
            else:
                self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self.qdrant_path))
        return self._client

    def collection_exists(self) -> bool:
        client = self._ensure_client()
        if not client.collection_exists(self.collection_name):
            return False
        info = client.get_collection(self.collection_name)
        return info.points_count > 0

    def collection_points_count(self) -> int:
        client = self._ensure_client()
        if not client.collection_exists(self.collection_name):
            return 0
        info = client.get_collection(self.collection_name)
        return int(info.points_count or 0)

    def load(self) -> None:
        """Attach to an existing local Qdrant store."""
        if not self.collection_exists():
            location = self.qdrant_url or str(self.qdrant_path)
            raise FileNotFoundError(
                f"No Qdrant collection '{self.collection_name}' with points at {location}. "
                "Build an index first or call build()."
            )
        self._ensure_client()

    def collection_vector_size(self) -> int | None:
        """Return configured vector size for the collection, when available."""
        client = self._ensure_client()
        if not client.collection_exists(self.collection_name):
            return None

        info = client.get_collection(self.collection_name)
        vectors = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)

        if isinstance(vectors, dict):
            # Named-vector schema: use the first configured vector.
            first = next(iter(vectors.values()), None)
            return int(getattr(first, "size", 0)) or None

        size = getattr(vectors, "size", None)
        return int(size) if size else None

    def build(self, records: list[DenseRecord]) -> None:
        """Create collection and upsert vectors from dense records (full reindex)."""
        if self.encoder is None:
            raise ValueError("An embedding encoder is required to build the Qdrant index.")
        client = self._ensure_client()
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        texts = [r.text for r in records]
        if not texts:
            return

        dim = len(self.encoder.encode([texts[0]])[0])
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        batch_size = 256
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch_emb = self.encoder.encode(texts[i : i + batch_size])
            points: list[PointStruct] = []
            for rec, vec in zip(batch, batch_emb, strict=True):
                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vec,
                        payload={
                            "doc_id": rec.document_id,
                            "text": rec.text[:500],
                            "metadata": rec.metadata,
                        },
                    )
                )
            client.upsert(collection_name=self.collection_name, points=points)

    def build_streaming(
        self,
        records: Iterable[DenseRecord],
        batch_size: int = 256,
        progress_interval: int = 10_000,
    ) -> int:
        """Create collection and upsert vectors while consuming records incrementally."""
        if self.encoder is None:
            raise ValueError("An embedding encoder is required to build the Qdrant index.")

        client = self._ensure_client()
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        iterator = iter(records)
        first = next(iterator, None)
        if first is None:
            return 0

        dim = len(self.encoder.encode([first.text])[0])
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        indexed_count = 0
        batch = [first]
        for record in iterator:
            batch.append(record)
            if len(batch) >= batch_size:
                self._upsert_batch(client, batch)
                indexed_count += len(batch)
                if indexed_count % progress_interval < batch_size:
                    print(f"Embedded {indexed_count} documents...", flush=True)
                batch = []

        if batch:
            self._upsert_batch(client, batch)
            indexed_count += len(batch)
            print(f"Embedded {indexed_count} documents...", flush=True)

        return indexed_count

    def _upsert_batch(self, client: QdrantClient, batch: list[DenseRecord]) -> None:
        texts = [record.text for record in batch]
        batch_emb = self.encoder.encode(texts) if self.encoder is not None else []
        points: list[PointStruct] = []
        for rec, vec in zip(batch, batch_emb, strict=True):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vec,
                    payload={
                        "doc_id": rec.document_id,
                        "text": rec.text[:500],
                        "metadata": rec.metadata,
                    },
                )
            )
        client.upsert(collection_name=self.collection_name, points=points)

    def _encode_query(self, query: str) -> list[float]:
        """BGE retrieval-style query prefix when using bge-* models."""
        if self.encoder is None:
            raise ValueError("An embedding encoder is required to query the Qdrant index.")
        model_lower = self.encoder.model_name.lower()
        if "bge" in model_lower:
            text = f"{BGE_QUERY_PREFIX}{query}"
        else:
            text = query
        return self.encoder.encode([text])[0]

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedDocument]:
        client = self._ensure_client()
        if not client.collection_exists(self.collection_name):
            return []

        qvec = self._encode_query(query)
        limit = max(top_k * 8, top_k * 3, 32)
        results = client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            limit=limit,
        )

        best_by_doc: dict[str, tuple[float, str, dict[str, Any]]] = {}
        for point in results.points:
            payload = point.payload or {}
            doc_id = str(payload.get("doc_id", ""))
            if not doc_id:
                continue
            text = str(payload.get("text", ""))
            meta_raw = payload.get("metadata")
            metadata: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            score = float(point.score or 0.0)
            prev = best_by_doc.get(doc_id)
            if prev is None or score > prev[0]:
                best_by_doc[doc_id] = (score, text, metadata)

        ranked = sorted(best_by_doc.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        out: list[RetrievedDocument] = []
        for doc_id, (score, text, metadata) in ranked:
            out.append(
                RetrievedDocument(
                    document_id=doc_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )
        return out


def dense_records_from_documents(documents: list[BenchmarkDocument]) -> list[DenseRecord]:
    """Build DenseRecord list from benchmark documents."""
    records: list[DenseRecord] = []
    for doc in documents:
        text = f"{doc.title}\n\n{doc.body}".strip()
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        records.append(
            DenseRecord(
                document_id=doc.document_id,
                text=text,
                metadata={
                    "source_type": doc.source_type,
                    **meta,
                },
            )
        )
    return records
