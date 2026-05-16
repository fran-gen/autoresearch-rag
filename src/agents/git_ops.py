"""Git operations for Karpathy mode -- branch isolation, stage/commit, revert.

Safety: these helpers NEVER run merge, push, rebase, or PR-creating commands.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = "src/retrieval/pipeline.py"

_FORBIDDEN_GIT_COMMANDS = {"merge", "push", "rebase", "pull", "pr"}


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command from the project root. Refuses dangerous operations."""
    if args and args[0] in _FORBIDDEN_GIT_COMMANDS:
        raise RuntimeError(f"Forbidden git operation: {args[0]}")
    cmd = ["git", "-C", str(PROJECT_ROOT), *args]
    logger.info("git_ops: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )


def _karpathy_commit_enabled() -> bool:
    return get_settings().karpathy_pipeline_commit


def create_karpathy_branch(run_id: str) -> str:
    """Create and switch to a karpathy experiment branch. Returns branch name."""
    branch = f"karpathy/run_{run_id}"
    result = _run_git("checkout", "-b", branch, check=False)
    if result.returncode != 0:
        if "already exists" in result.stderr:
            _run_git("checkout", branch)
        else:
            logger.error("Failed to create branch %s: %s", branch, result.stderr)
            raise RuntimeError(f"git branch creation failed: {result.stderr}")
    _run_git("add", PIPELINE_PATH, check=False)
    if _karpathy_commit_enabled():
        _run_git(
            "commit",
            "-m",
            f"karpathy: baseline pipeline.py (run {run_id})",
            "--allow-empty",
            check=False,
        )
    else:
        logger.info(
            "karpathy: baseline pipeline.py staged on %s (no commit; set KARPATHY_PIPELINE_COMMIT=1 to commit)",
            branch,
        )
    return branch


def commit_pipeline(message: str) -> bool:
    """Stage pipeline.py; commit too only if KARPATHY_PIPELINE_COMMIT=1. Returns True on success."""
    _run_git("add", PIPELINE_PATH)
    if not _karpathy_commit_enabled():
        logger.info(
            "karpathy: staged pipeline.py (%s) — no commit; set KARPATHY_PIPELINE_COMMIT=1 to commit",
            message[:80],
        )
        return True
    result = _run_git("commit", "-m", message, check=False)
    if result.returncode != 0:
        logger.warning("Commit failed (nothing to commit?): %s", result.stderr)
        return False
    return True


def force_commit_pipeline(message: str) -> bool:
    """Stage and commit pipeline.py unconditionally (ignores KARPATHY_PIPELINE_COMMIT)."""
    _run_git("add", PIPELINE_PATH)
    result = _run_git("commit", "-m", message, check=False)
    if result.returncode != 0:
        logger.warning("Force commit failed (nothing to commit?): %s", result.stderr)
        return False
    return True


def revert_pipeline() -> None:
    """Restore pipeline.py from the index (matches ``git checkout --``)."""
    _run_git("checkout", "--", PIPELINE_PATH, check=False)


def get_branch_log(branch: str, max_entries: int = 20) -> list[dict[str, str]]:
    """Return recent commits on the given branch as [{hash, message}]."""
    result = _run_git(
        "log", branch, f"--max-count={max_entries}",
        "--format=%H|||%s",
        check=False,
    )
    if result.returncode != 0:
        return []
    entries = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|||", 1)
        if len(parts) == 2:
            entries.append({"hash": parts[0][:8], "message": parts[1]})
    return entries


def current_branch() -> str:
    """Return the name of the currently checked-out branch."""
    result = _run_git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def return_to_branch(branch: str) -> None:
    """Switch back to the given branch (e.g. after the run completes)."""
    _run_git("checkout", branch, check=False)
