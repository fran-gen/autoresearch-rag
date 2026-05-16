from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedDocument]:
        raise NotImplementedError
