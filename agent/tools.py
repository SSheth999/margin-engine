"""Tool implementations for the local task set.

Tools operate inside a per-run workspace directory (sandboxed to that dir). Kept
deliberately small — enough to drive simple multi-step tasks and to make genuine
failure modes (loops, context growth) possible later.

Each tool has:
  - a schema (canonical: {"name","description","parameters"}) sent to the model
  - a Python implementation taking (workspace: Path, **kwargs) -> str result
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def _safe_path(workspace: Path, path: str) -> Path:
    """Resolve `path` inside workspace, refusing escapes."""
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        raise ValueError(f"path escapes workspace: {path}")
    return p


def read_file(workspace: Path, path: str) -> str:
    p = _safe_path(workspace, path)
    if not p.exists():
        return f"ERROR: no such file: {path}"
    return p.read_text()[:10_000]


def write_file(workspace: Path, path: str, content: str) -> str:
    p = _safe_path(workspace, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def list_dir(workspace: Path, path: str = ".") -> str:
    p = _safe_path(workspace, path)
    if not p.exists():
        return f"ERROR: no such dir: {path}"
    entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def search(workspace: Path, query: str) -> str:
    """Stub 'web search'. Returns a fixed unhelpful result on purpose.

    A tool that never yields new information is a realistic way to induce a tool
    loop (repeated identical calls) in a weaker model — useful for the failure-mode
    experiments in later passes. The result is deliberately verbose to mimic the
    token cost of real retrieved context piling up on each loop iteration.
    """
    filler = (
        "No authoritative source found. Retrieved snippets were irrelevant or "
        "low-confidence and did not contain the requested value. Consider refining "
        "the query or trying an alternative approach. "
    ) * 8
    return f"No useful results found for: {query!r}.\n{filler}"


TOOL_IMPLS: dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "search": search,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a file in the workspace (creates/overwrites).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List entries in a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "search",
        "description": "Search for information about a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def run_tool(name: str, workspace: Path, arguments: dict[str, Any]) -> str:
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return impl(workspace, **arguments)
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR: {name} failed: {e}"
