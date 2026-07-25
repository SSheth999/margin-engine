"""TB2 task loader + verifier plumbing (offline, no sandbox service / no network)."""

from pathlib import Path

from agent.exec_env import LocalExecEnv
from tasks.tb_loader import discover_tasks, load_task
from tasks.tb_verify import verify_resolved

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tb_sample"


def test_discover_and_load_fix_git():
    dirs = discover_tasks(FIXTURES)
    assert any(d.name == "fix-git" for d in dirs)

    task = load_task(FIXTURES / "fix-git")
    assert task.task_id == "fix-git"                     # from "terminal-bench/fix-git"
    assert task.docker_image == "alexgshaw/fix-git:20260403"
    assert task.cpus == 1 and task.memory_mb == 2048 and task.memory_gb == 2
    assert task.storage_mb == 10240 and task.disk_gb == 10 and task.gpus == 0
    assert task.difficulty == "easy" and task.complexity_class == "easy"
    assert task.contracted_price == 0.50                 # easy price from defaults
    assert "recover" in task.instruction.lower()


def test_price_margin_overrides():
    task = load_task(
        FIXTURES / "fix-git",
        price_by_class={"easy": 2.0, "medium": 3.0, "hard": 4.0},
        margin_by_class={"easy": 0.5, "medium": 0.6, "hard": 0.7},
    )
    assert task.contracted_price == 2.0 and task.target_margin == 0.5


def test_verify_resolved_pass_and_fail(tmp_path):
    task = load_task(FIXTURES / "fix-git")
    # Local overrides: workdir-relative mounts (LocalExecEnv can't chroot to '/').
    kw = dict(tests_mount="tests", logs_dir="logs/verifier",
              reward_path="logs/verifier/reward.txt")

    # Failing case: no solution present -> reward 0.
    env_fail = LocalExecEnv(tmp_path / "fail")
    assert verify_resolved(env_fail, task, **kw) is False

    # Passing case: seed the marker the fixture test.sh checks for.
    env_ok = LocalExecEnv(tmp_path / "ok")
    env_ok.exec("echo 'recovered the commits' > solution.txt")
    assert verify_resolved(env_ok, task, **kw) is True
