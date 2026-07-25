"""Terminal-Bench verification: inject tests, run the test script, read the reward.

Replicates TB2's verification phase (kept OUT of the agent's environment until after
the agent finishes, so the agent can't read or game the tests):

  1. create /logs/verifier and /tests in the environment,
  2. upload the task's tests/ dir to /tests,
  3. run `bash /tests/test.sh` (installs uv, runs pytest, writes reward.txt),
  4. read /logs/verifier/reward.txt -> "1" means resolved.

Works against any ExecEnv, so it runs the same way in an E2B sandbox or locally.
"""

from __future__ import annotations

from agent.exec_env import ExecEnv, upload_dir
from tasks.tb_loader import TBTask

# TB2 standard absolute mount points (correct inside a real sandbox). The local test
# harness overrides these with workdir-relative paths, since LocalExecEnv can't chroot.
REMOTE_TESTS = "/tests"
REMOTE_LOGS = "/logs/verifier"
REMOTE_REWARD = "/logs/verifier/reward.txt"


def verify_resolved(
    env: ExecEnv,
    task: TBTask,
    tests_mount: str = REMOTE_TESTS,
    logs_dir: str = REMOTE_LOGS,
    reward_path: str = REMOTE_REWARD,
) -> bool:
    """Run the task's test suite in the environment and return pass/fail."""
    if not task.tests_dir.exists():
        return False

    env.mkdir(logs_dir)
    upload_dir(env, task.tests_dir, tests_mount)

    timeout = int(task.verifier_timeout_sec)
    env.exec(f"bash {tests_mount}/test.sh", timeout=timeout)

    reward = (env.read_file(reward_path) or "").strip()
    return reward == "1"
