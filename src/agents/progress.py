from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable

ProgressReporter = Callable[[str, str | None, dict[str, Any] | None], None]

_progress_reporter: ContextVar[ProgressReporter | None] = ContextVar(
    "progress_reporter",
    default=None,
)


def set_progress_reporter(reporter: ProgressReporter | None) -> Token[ProgressReporter | None]:
    return _progress_reporter.set(reporter)


def reset_progress_reporter(token: Token[ProgressReporter | None]) -> None:
    _progress_reporter.reset(token)


def report_progress(
    phase: str,
    summary: str | None = None,
    **detail: Any,
) -> None:
    reporter = _progress_reporter.get()
    if reporter is None:
        return
    reporter(phase, summary, detail or None)
