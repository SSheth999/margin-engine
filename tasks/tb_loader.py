"""Terminal-Bench 2 task loader.

Reads TB2 task directories (from a local clone of harbor-framework/terminal-bench-2-1)
and maps them onto the experiment's per-task config. Each task dir contains:

    task.toml            metadata + environment (docker_image, cpus, memory_mb, timeouts)
    instruction.md       the prompt given to the agent
    environment/         Dockerfile etc. (image is prebuilt & referenced in task.toml)
    solution/solve.sh    oracle solution (not used by our arms)
    tests/               test.sh + test_outputs.py -> injected at verification time

task.toml pins a prebuilt registry image (e.g. "alexgshaw/fix-git:20260403"); the E2B
orchestrator wraps it into an E2B template once (built from that image) and reuses it.

Contracted price / target margin aren't part of TB2, so we synthesize them from the
task's difficulty (a knob in config/experiment.yaml) — the margin ceiling is what the
gateway measures cost against.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TBTask:
    task_id: str
    task_dir: Path
    instruction: str
    docker_image: str
    cpus: int
    memory_mb: int
    storage_mb: int
    gpus: int
    difficulty: str
    category: str
    agent_timeout_sec: float
    verifier_timeout_sec: float
    allow_internet: bool
    # Experiment economics (synthesized from difficulty, see config/experiment.yaml).
    contracted_price: float = 1.0
    target_margin: float = 0.70
    complexity_class: str = "medium"
    expected_failure_mode: str | None = None

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def memory_gb(self) -> int:
        return max(1, round(self.memory_mb / 1024))

    @property
    def disk_gb(self) -> int:
        return max(1, round(self.storage_mb / 1024))


# TB2 difficulty -> our complexity_class bucket.
_DIFFICULTY_TO_CLASS = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def load_task(
    task_dir: Path,
    price_by_class: dict[str, float] | None = None,
    margin_by_class: dict[str, float] | None = None,
) -> TBTask:
    with open(task_dir / "task.toml", "rb") as f:
        cfg = tomllib.load(f)

    env = cfg.get("environment", {})
    meta = cfg.get("metadata", {})
    task = cfg.get("task", {})

    instruction = (task_dir / "instruction.md").read_text()
    difficulty = meta.get("difficulty", "medium")
    complexity = _DIFFICULTY_TO_CLASS.get(difficulty, "medium")

    price_by_class = price_by_class or {"easy": 0.50, "medium": 1.00, "hard": 1.50}
    margin_by_class = margin_by_class or {"easy": 0.70, "medium": 0.70, "hard": 0.70}

    # task name like "terminal-bench/fix-git" -> "fix-git"
    raw_name = task.get("name", task_dir.name)
    task_id = raw_name.split("/")[-1]

    docker_image = env.get("docker_image")
    if not docker_image:
        # Harbor tasks normally pin a prebuilt registry image. A task that ships only a
        # Dockerfile would need a build step we don't do — surface it clearly.
        raise ValueError(
            f"task {task_id!r} has no environment.docker_image; building from a "
            "Dockerfile is not supported by this rig (E2B builds a template from a "
            "prebuilt image)."
        )

    return TBTask(
        task_id=task_id,
        task_dir=task_dir,
        instruction=instruction,
        docker_image=docker_image,
        cpus=int(env.get("cpus", 1)),
        memory_mb=int(env.get("memory_mb", 2048)),
        storage_mb=int(env.get("storage_mb", 10240)),
        gpus=int(env.get("gpus", 0)),
        difficulty=difficulty,
        category=meta.get("category", "unknown"),
        agent_timeout_sec=float(cfg.get("agent", {}).get("timeout_sec", 900.0)),
        verifier_timeout_sec=float(cfg.get("verifier", {}).get("timeout_sec", 900.0)),
        allow_internet=bool(env.get("allow_internet", True)),
        contracted_price=price_by_class.get(complexity, 1.0),
        target_margin=margin_by_class.get(complexity, 0.70),
        complexity_class=complexity,
    )


def discover_tasks(tasks_root: Path) -> list[Path]:
    """Every immediate subdir of tasks_root that contains a task.toml."""
    if not tasks_root.exists():
        return []
    return sorted(
        d for d in tasks_root.iterdir() if d.is_dir() and (d / "task.toml").exists()
    )
