from __future__ import annotations

import ast
import re
from typing import Any


def extract_text_content(value: Any) -> str:
    """Normalize LLM content blocks, plain strings, and legacy reprs to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return _extract_from_string(value)
    if isinstance(value, dict):
        if value.get("type") == "text" and value.get("text") is not None:
            return str(value.get("text") or "").strip()
        if value.get("text") is not None:
            return str(value.get("text") or "").strip()
        return str(value).strip()
    if isinstance(value, list):
        parts = [extract_text_content(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _extract_from_string(value: str) -> str:
    text = value.strip()
    if not text:
        return ""

    parsed = _literal_parse(text)
    if parsed is not None and parsed is not value:
        extracted = extract_text_content(parsed)
        if extracted:
            return extracted

    return _replace_embedded_content_blocks(text).strip()


def _literal_parse(text: str) -> Any | None:
    if not (text.startswith("[") or text.startswith("{")):
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _replace_embedded_content_blocks(text: str) -> str:
    patterns = [
        re.compile(r"\[\s*\{[^{}]*(?:'|\")type(?:'|\")\s*:\s*(?:'|\")text(?:'|\")[\s\S]*?\}\s*\]"),
        re.compile(r"\{\s*(?:'|\")type(?:'|\")\s*:\s*(?:'|\")text(?:'|\")[\s\S]*?\}"),
    ]
    normalized = text
    for pattern in patterns:
        normalized = pattern.sub(_block_match_to_text, normalized)
    return normalized


def _block_match_to_text(match: re.Match[str]) -> str:
    parsed = _literal_parse(match.group(0))
    if parsed is None:
        return match.group(0)
    return extract_text_content(parsed)
