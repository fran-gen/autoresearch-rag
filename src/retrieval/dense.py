from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DenseRecord:
    document_id: str
    text: str
    metadata: dict[str, Any]
