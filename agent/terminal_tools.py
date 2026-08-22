"""Terminal tools for Terminal-Bench tasks — several tools, one ExecEnv behind them.

TB2 tasks are solved in the shell, so every tool here ultimately runs a bash command
in the task environment via the injected `ExecEnv`. The surface is deliberately
PARTITIONED rather than collapsed into a single `run_command`, for one experimental
reason: `restrict_tool` is only a meaningful intervention if the agent has somewhere
else to go.

With a single-tool surface, restricting that tool empties the `tools` array, the model
can emit no tool call, and `agent/harness.py` reads "no tool_calls" as a finished run —
so `restrict` silently terminates the run instead of redirecting it. (Observed in the
first full benchmark: all 5 restrict firings ended the run on that exact step.)

Splitting the surface adds no capability the agent didn't already have through bash —
`read_file` is `sed -n`, `write_file` is a base64 heredoc, `search_files` is `grep -rn`.
It only means that losing one tool degrades the agent instead of killing it, which is
what the intervention is supposed to test.
"""

from __future__ import annotations

import base64
import shlex
from typing import Any, Callable

from agent.exec_env import ExecEnv

# Max characters returned from a single tool call. Tool output is the main driver of
# context growth, so the cap is part of the experiment's cost mechanics.
MAX_OUTPUT_CHARS = 10_000

RUN_COMMAND_SCHEMA: dict[str, Any] = {
    "name": "run_command",
    "description": (
        "Execute an arbitrary bash command in the task's terminal environment and "
        "return its combined stdout/stderr and exit code. Use this for anything the "
        "more specific tools do not cover (building, installing, running programs)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run."}
        },
        "required": ["command"],
    },
}

READ_FILE_SCHEMA: dict[str, Any] = {
    "name": "read_file",
    "description": "Read a text file from the task environment, optionally a line range.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
            "start_line": {"type": "integer", "description": "1-indexed first line."},
            "end_line": {"type": "integer", "description": "1-indexed last line."},
        },
        "required": ["path"],
    },
}

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Write text to a file in the task environment, creating parent directories and "
        "overwriting any existing content. Content is transferred verbatim."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}

LIST_DIR_SCHEMA: dict[str, Any] = {
    "name": "list_dir",
    "description": "List the contents of a directory in the task environment.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default '.')."}
        },
    },
}

SEARCH_FILES_SCHEMA: dict[str, Any] = {
    "name": "search_files",
    "description": (
        "Recursively search files for a regular expression and return matching lines "
        "with their file and line number."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Extended regex to search for."},
            "path": {"type": "string", "description": "Root to search under (default '.')."},
        },
        "required": ["pattern"],
    },
}

TERMINAL_TOOL_SCHEMAS = [
    RUN_COMMAND_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    LIST_DIR_SCHEMA,
    SEARCH_FILES_SCHEMA,
]

TERMINAL_SYSTEM_PROMPT = """You are an autonomous engineer solving a task in a Linux terminal.

You have these tools, all acting on the same machine:
  run_command   - run any bash command (building, installing, executing programs)
  read_file     - read a file, optionally a line range
  write_file    - create or overwrite a file with exact content
  list_dir      - list a directory
  search_files  - recursively grep for a regex

Work step by step: inspect the environment, make changes, and verify your work by
running commands. Each command runs in a fresh shell, so use absolute paths or `cd`
within a single command. Prefer the specific tools (read_file/write_file/list_dir/
search_files) over run_command for those operations.

If a tool you were using is no longer available to you, do not stop — accomplish the
same thing with one of the remaining tools and continue the task.

Do not repeat an identical failing command; change approach instead. When the task is
fully complete, reply with a final message beginning with 'DONE:' and a brief summary."""


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[...truncated at {MAX_OUTPUT_CHARS} chars]"


def _build_command(name: str, arguments: dict[str, Any]) -> str | None:
    """Translate a tool call into the bash command that implements it.

    Returns None for an unknown tool. Every path/pattern is shell-quoted, and file
    content goes over the wire base64-encoded so arbitrary bytes survive intact.
    """
    if name == "run_command":
        return arguments.get("command") or None

    if name == "read_file":
        path = arguments.get("path")
        if not path:
            return None
        q = shlex.quote(str(path))
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        if start or end:
            s = int(start) if start else 1
            e = int(end) if end else "$"
            return f"sed -n {shlex.quote(f'{s},{e}p')} {q}"
        return f"cat {q}"

    if name == "write_file":
        path = arguments.get("path")
        if not path:
            return None
        content = arguments.get("content") or ""
        blob = base64.b64encode(str(content).encode()).decode()
        q = shlex.quote(str(path))
        return (
            f"mkdir -p \"$(dirname {q})\" && "
            f"printf '%s' {shlex.quote(blob)} | base64 -d > {q} && "
            f"echo \"wrote $(wc -c < {q}) bytes to {q}\""
        )

    if name == "list_dir":
        path = arguments.get("path") or "."
        return f"ls -la {shlex.quote(str(path))}"

    if name == "search_files":
        pattern = arguments.get("pattern")
        if not pattern:
            return None
        path = arguments.get("path") or "."
        return f"grep -rnE {shlex.quote(str(pattern))} {shlex.quote(str(path))}"

    return None


def make_terminal_tool_runner(
    env: ExecEnv, command_timeout: int = 120
) -> Callable[[str, dict[str, Any]], str]:
    """Return a tool_runner(name, arguments) -> str bound to this ExecEnv."""

    def _runner(name: str, arguments: dict[str, Any]) -> str:
        if name not in {s["name"] for s in TERMINAL_TOOL_SCHEMAS}:
            return f"ERROR: unknown tool {name!r}"
        command = _build_command(name, arguments)
        if not command:
            return f"ERROR: missing required argument for {name}"
        res = env.exec(command, timeout=command_timeout)
        return _clip(f"(exit {res.exit_code})\n{res.output}")

    return _runner
