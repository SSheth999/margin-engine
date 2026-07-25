"""E2BExecEnv translation against a mocked E2B sandbox. No network / no E2B_API_KEY.

Proves exec/upload/read/mkdir map onto the E2B Sandbox API and that a failing command
(CommandExitException) is normalized into an ExecResult instead of propagating.
"""

from pathlib import Path
from types import SimpleNamespace

from agent.exec_env import E2BExecEnv


class _CommandExit(Exception):
    def __init__(self, exit_code, stdout, stderr):
        super().__init__(stderr)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _Commands:
    def __init__(self):
        self.calls = []

    def run(self, cmd, timeout=None):
        self.calls.append(cmd)
        if cmd.startswith("false") or "FAIL" in cmd:
            raise _CommandExit(3, "partial-out", "boom")
        return SimpleNamespace(stdout="ok-out", stderr="", exit_code=0, error=None)


class _Files:
    def __init__(self):
        self.written = {}
        self.dirs = []
        self.store = {"/logs/verifier/reward.txt": "1\n"}

    def write(self, path, data):
        self.written[path] = data

    def read(self, path, format="text"):
        if path not in self.store:
            raise FileNotFoundError(path)
        return self.store[path]

    def make_dir(self, path):
        self.dirs.append(path)


class _Sandbox:
    def __init__(self):
        self.commands = _Commands()
        self.files = _Files()
        self.killed = False

    def kill(self):
        self.killed = True


def test_exec_success_and_failure_are_normalized():
    sb = _Sandbox()
    env = E2BExecEnv(sb)

    ok = env.exec("echo hi")
    assert ok.exit_code == 0 and "ok-out" in ok.output

    bad = env.exec("false; FAIL")
    assert bad.exit_code == 3 and ("boom" in bad.output or "partial-out" in bad.output)


def test_upload_read_mkdir_and_close(tmp_path):
    sb = _Sandbox()
    env = E2BExecEnv(sb)

    local = tmp_path / "t.txt"
    local.write_text("payload")
    env.upload_file(local, "/tests/t.txt")
    assert sb.files.written["/tests/t.txt"] == b"payload"

    env.mkdir("/logs/verifier")
    assert "/logs/verifier" in sb.files.dirs

    assert env.read_file("/logs/verifier/reward.txt").strip() == "1"
    assert env.read_file("/does/not/exist") is None

    env.close()
    assert sb.killed is True
