"""Terminal tool for Terminal-Bench tasks: a single `run_command` bound to an ExecEnv.

TB2 tasks are solved in the shell, so the agent gets one tool that executes a bash
command in the task environment and returns combined stdout/stderr plus exit code.
This is deliberately the whole toolset — it mirrors how Terminal-Bench agents operate.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.exec_env import ExecEnv

RUN_COMMAND_SCHEMA: dict[str, Any] = {
    "name": "run_command",
    "description": (
        "Execute a bash command in the task's terminal environment and return its "
        "combined stdout/stderr and exit code. Use this to inspect and modify the "
        "system to accomplish the task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run."}
        },
        "required": ["command"],
    },
}

TERMINAL_TOOL_SCHEMAS = [RUN_COMMAND_SCHEMA]

TERMINAL_SYSTEM_PROMPT = """You are an autonomous engineer solving a task in a Linux terminal.
You have one tool, run_command, which runs a bash command and returns its output and exit
code. Work step by step: inspect the environment, make changes, and verify your work with
commands. Commands run in a fresh shell each time, so use absolute paths or `cd` within a
single command. When the task is fully complete, reply with a final message beginning with
'DONE:' and a brief summary. Do not repeat an identical failing command — change approach."""


def make_terminal_tool_runner(
    env: ExecEnv, command_timeout: int = 120
) -> Callable[[str, dict[str, Any]], str]:
    """Return a tool_runner(name, arguments) -> str bound to this ExecEnv."""

    def _runner(name: str, arguments: dict[str, Any]) -> str:
        if name != "run_command":
            return f"ERROR: unknown tool {name!r}"
        command = arguments.get("command", "")
        if not command:
            return "ERROR: no command provided"
        res = env.exec(command, timeout=command_timeout)
        return f"(exit {res.exit_code})\n{res.output}"

    return _runner
