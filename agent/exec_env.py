"""Execution environment abstraction — where the agent's shell tool actually runs.

Terminal-Bench tasks are terminal tasks: the agent's one real tool is "run a shell
command", and grading runs a test script in the same environment. We abstract that
environment behind `ExecEnv` so the SAME agent harness drives either:

  - LocalExecEnv : subprocess in a local workdir. No sandbox service needed — used
                   for offline dev/tests and the toy task set.
  - E2BExecEnv   : commands + files run inside an E2B sandbox (open-source, Apache-2.0,
                   self-hostable) built from the task's Harbor Docker image. Used for
                   the real TB2 matrix.

Only the environment changes across these; the agent, gateway, and arms are identical,
preserving experiment validity.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ExecResult:
    exit_code: int
    output: str


class ExecEnv(Protocol):
    def exec(self, command: str, timeout: int | None = None) -> ExecResult: ...
    def upload_file(self, local_path: Path, remote_path: str) -> None: ...
    def read_file(self, remote_path: str) -> str | None: ...
    def mkdir(self, remote_path: str) -> None: ...
    def close(self) -> None: ...


class LocalExecEnv:
    """Runs commands via bash in a local working directory. Paths are relative to it."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def exec(self, command: str, timeout: int | None = None) -> ExecResult:
        try:
            p = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (p.stdout or "") + (p.stderr or "")
            return ExecResult(exit_code=p.returncode, output=out[:20_000])
        except subprocess.TimeoutExpired:
            return ExecResult(exit_code=124, output=f"[timed out after {timeout}s]")

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        dst = self._resolve(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dst)

    def read_file(self, remote_path: str) -> str | None:
        p = self._resolve(remote_path)
        return p.read_text() if p.exists() else None

    def mkdir(self, remote_path: str) -> None:
        self._resolve(remote_path).mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        pass

    def _resolve(self, remote_path: str) -> Path:
        # Treat absolute-looking remote paths (e.g. /tests) as workdir-relative locally.
        rel = remote_path.lstrip("/")
        return self.workdir / rel


class E2BExecEnv:
    """Runs commands + files inside an E2B sandbox. Wraps an already-created sandbox.

    E2B's `commands.run` raises CommandExitException on a non-zero exit; the agent needs
    the exit code + output regardless (failing commands are normal), so we catch it and
    normalize into ExecResult rather than letting it propagate.
    """

    def __init__(self, sandbox):
        self._sb = sandbox

    def exec(self, command: str, timeout: int | None = None) -> ExecResult:
        try:
            res = self._sb.commands.run(command, timeout=timeout)
            out = (res.stdout or "") + (res.stderr or "")
            return ExecResult(exit_code=res.exit_code, output=out[:20_000])
        except Exception as e:
            # CommandExitException carries exit_code/stdout/stderr; other errors -> 1.
            code = getattr(e, "exit_code", 1)
            out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
            if not out:
                out = str(e)
            return ExecResult(exit_code=code, output=out[:20_000])

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        self._sb.files.write(remote_path, Path(local_path).read_bytes())

    def read_file(self, remote_path: str) -> str | None:
        try:
            return self._sb.files.read(remote_path, format="text")
        except Exception:
            return None

    def mkdir(self, remote_path: str) -> None:
        try:
            self._sb.files.make_dir(remote_path)
        except Exception:
            self._sb.commands.run(f"mkdir -p {remote_path}")

    def close(self) -> None:
        try:
            self._sb.kill()
        except Exception:
            pass


def upload_dir(env: ExecEnv, local_dir: Path, remote_dir: str) -> None:
    """Upload every file under local_dir into remote_dir, preserving relative layout."""
    env.mkdir(remote_dir)
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            env.upload_file(p, f"{remote_dir.rstrip('/')}/{rel}")
