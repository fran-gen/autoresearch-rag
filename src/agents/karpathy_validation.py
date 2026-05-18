"""Validation helpers for Karpathy-mode pipeline candidates."""

from __future__ import annotations

import ast


FORBIDDEN_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "shutil",
    "http",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
})


def validate_imports(code: str) -> str | None:
    """Check that candidate code avoids forbidden modules."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    return f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    return f"Forbidden import from: {node.module}"
    return None


def validate_signature(code: str) -> str | None:
    """Verify retrieve() exists with the fixed Karpathy-mode parameters."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "retrieve":
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names == ["question", "retriever", "top_k"]:
                return None
            return (
                f"Wrong signature: retrieve({', '.join(arg_names)}). "
                "Expected (question, retriever, top_k)"
            )
    return "Missing retrieve() function"


def validate_karpathy_code(code: str) -> str | None:
    """Run static validation. Returns an error message or None on success."""
    try:
        compile(code, "pipeline.py", "exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    err = validate_imports(code)
    if err:
        return err

    err = validate_signature(code)
    if err:
        return err

    return None
