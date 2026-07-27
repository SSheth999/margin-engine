"""E2B orchestrator — real sandboxed run matrix over Terminal-Bench 2 (Harbor) tasks.

Uses E2B (open-source, Apache-2.0, self-hostable) for the isolated per-run sandbox.

For each (task, arm, seed) run:
  1. ensure an E2B template exists for the task's Harbor Docker image (built once per
     image via Template.from_image(...).build(...), cached by alias),
  2. create an E2B sandbox from that template,
  3. start a host-side gateway (uvicorn, ephemeral port) carrying run-level context,
  4. run the fixed agent against that gateway; its `run_command` tool execs inside the
     sandbox (E2BExecEnv),
  5. verify: inject tests/, run test.sh, read reward.txt -> resolved,
  6. finalize the gateway (writes the JSON run log) and kill the sandbox.

Same phase ordering as the local orchestrator (A -> label -> B/C/C-oracle). The
experiment logic (arms/detectors/interventions/labeling/analysis) is shared and
unchanged; only the execution substrate differs.

Requires E2B access: set E2B_API_KEY (and E2B_DOMAIN for a self-hosted instance). A
local clone of the TB2 tasks provides the task defs + tests; see config/experiment.yaml
and scripts/fetch_terminal_bench.sh.

NOTE on images: E2B builds sandboxes from templates, so each Harbor image is wrapped
into an E2B template the first time it's seen. E2B currently supports Debian-based
images; the alias is cached so repeat runs of the same image skip the rebuild.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from agent.exec_env import E2BExecEnv
from agent.terminal_tools import (
    TERMINAL_SYSTEM_PROMPT,
    TERMINAL_TOOL_SCHEMAS,
    make_terminal_tool_runner,
)
from tasks.tb_loader import TBTask, discover_tasks, load_task
from tasks.tb_verify import verify_resolved

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "config"
RUNS_DIR = REPO / "runs"
LABELS_PATH = CONFIG_DIR / "labels.json"
TEMPLATE_CACHE = CONFIG_DIR / "e2b_templates.json"

ALL_ARMS = ["A", "B", "C", "C-oracle"]


def _load_experiment_cfg() -> dict:
    with open(CONFIG_DIR / "experiment.yaml") as f:
        return yaml.safe_load(f)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _primary_model() -> str:
    with open(CONFIG_DIR / "providers.yaml") as f:
        cfg = yaml.safe_load(f)
    active = cfg["active_provider"]
    return cfg["providers"][active]["model_pair"]["primary"]


def _wait_healthy(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"gateway on port {port} did not become healthy")


def _spawn_gateway(port: int, arm: str, task: TBTask, seed: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "MARGIN_ARM": arm,
            "MARGIN_TASK_ID": task.task_id,
            "MARGIN_SEED": str(seed),
            "MARGIN_COMPLEXITY": task.complexity_class,
            "MARGIN_RUN_DIR": str(RUNS_DIR),
        }
    )
    if LABELS_PATH.exists():
        env["MARGIN_LABELS"] = str(LABELS_PATH)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gateway.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(REPO), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def _require_api_key() -> None:
    if not os.environ.get("E2B_API_KEY"):
        raise RuntimeError(
            "E2B_API_KEY not set. Export it (and E2B_DOMAIN for a self-hosted E2B "
            "instance) before running the E2B matrix. The SDK reads both from env."
        )


def _template_alias(image: str) -> str:
    """Deterministic, E2B-safe alias for a Harbor image (lowercase, [a-z0-9-])."""
    slug = re.sub(r"[^a-z0-9]+", "-", image.lower()).strip("-")
    return f"tb-{slug}"[:63]


def _load_template_cache() -> dict[str, str]:
    if TEMPLATE_CACHE.exists():
        return json.loads(TEMPLATE_CACHE.read_text())
    return {}


def _save_template_cache(cache: dict[str, str]) -> None:
    TEMPLATE_CACHE.write_text(json.dumps(cache, indent=2))


def ensure_template(image: str, cache: dict[str, str], task: TBTask) -> str:
    """Return an E2B template alias for a Harbor image, building it once if needed."""
    from e2b import Template

    if image in cache:
        return cache[image]

    alias = _template_alias(image)
    if not Template.alias_exists(alias):
        print(f"[e2b] building template {alias} from {image} (first use)...")
        builder = Template().from_image(image)
        Template.build(
            builder,
            alias=alias,
            cpu_count=task.cpus,
            memory_mb=task.memory_mb,
            on_build_logs=lambda e: None,
        )
    cache[image] = alias
    _save_template_cache(cache)
    return alias


def _create_sandbox(alias: str, task: TBTask):
    from e2b import Sandbox

    return Sandbox.create(
        template=alias,
        timeout=max(120, int(task.agent_timeout_sec)),
        allow_internet_access=task.allow_internet,
    )


def run_matrix(
    tasks: list[TBTask], seeds: list[int], arms: list[str], max_steps: int
) -> list[dict]:
    from agent.harness import run_agent

    _require_api_key()
    model = _primary_model()
    template_cache = _load_template_cache()
    results: list[dict] = []

    for arm in arms:
        for task in tasks:
            alias = ensure_template(task.docker_image, template_cache, task)
            for seed in seeds:
                run_id = f"{task.task_id}__{arm}__seed{seed}"
                port = _free_port()
                proc = None
                env = None
                try:
                    sandbox = _create_sandbox(alias, task)
                    env = E2BExecEnv(sandbox)

                    proc = _spawn_gateway(port, arm, task, seed)
                    gateway_url = f"http://127.0.0.1:{port}"
                    _wait_healthy(port)

                    summary = run_agent(
                        gateway_url=gateway_url,
                        model=model,
                        outcome_id=task.task_id,
                        contracted_price=task.contracted_price,
                        target_margin=task.target_margin,
                        prompt=task.instruction,
                        tool_schemas=TERMINAL_TOOL_SCHEMAS,
                        tool_runner=make_terminal_tool_runner(env),
                        resolve_fn=lambda _t, _f, e=env, tk=task: verify_resolved(e, tk),
                        system_prompt=TERMINAL_SYSTEM_PROMPT,
                        max_steps=max_steps,
                        seed=seed,
                    )
                    summary["run_id"] = run_id
                    results.append(summary)
                    print(
                        f"[{run_id}] resolved={summary.get('resolved')} "
                        f"cost={summary.get('final_cost')} "
                        f"margin_held={summary.get('margin_held')}"
                    )
                except Exception as e:
                    print(f"[{run_id}] ERROR: {e!r}")
                    results.append({"run_id": run_id, "error": repr(e)})
                finally:
                    if proc is not None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    if env is not None:
                        env.close()  # kills the sandbox

    return results


def prebuild_templates(tasks: list[TBTask]) -> dict[str, bool]:
    """Build an E2B template for every distinct task image up front.

    Surfaces incompatible images (e.g. non-Debian) BEFORE any model tokens are spent.
    Returns {image: ok}. Successful aliases are cached for the real run to reuse.
    """
    _require_api_key()
    cache = _load_template_cache()
    by_image: dict[str, TBTask] = {}
    for t in tasks:
        by_image.setdefault(t.docker_image, t)

    outcomes: dict[str, bool] = {}
    for image, task in by_image.items():
        try:
            alias = ensure_template(image, cache, task)
            outcomes[image] = True
            print(f"[prebuild] OK   {image} -> {alias}")
        except Exception as e:
            outcomes[image] = False
            print(f"[prebuild] FAIL {image}: {e!r}")

    ok = sum(1 for v in outcomes.values() if v)
    print(f"[prebuild] {ok}/{len(outcomes)} images built successfully")
    return outcomes


def _label_and_backfill() -> None:
    from analysis.labeling import backfill_true_mode, build_labels

    labels = build_labels(RUNS_DIR)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))
    n = backfill_true_mode(labels, RUNS_DIR)
    print(f"[labeling] {len(labels)} labels written, backfilled into {n} logs")


def _select_tasks(cfg: dict) -> list[TBTask]:
    tb = cfg["terminal_bench"]
    tasks_root = Path(tb["tasks_root"]).expanduser()
    include = set(tb.get("include_tasks") or [])
    limit = tb.get("limit")
    skip_gpu = bool(tb.get("skip_gpu_tasks", True))

    dirs = discover_tasks(tasks_root)
    if not dirs:
        raise RuntimeError(
            f"No TB2 tasks found under {tasks_root}. Clone them first: "
            "scripts/fetch_terminal_bench.sh"
        )
    tasks = [
        load_task(d, cfg.get("price_by_class"), cfg.get("margin_by_class"))
        for d in dirs
    ]
    if include:
        tasks = [t for t in tasks if t.task_id in include]
    if skip_gpu:
        before = len(tasks)
        tasks = [t for t in tasks if t.gpus == 0]
        if len(tasks) < before:
            print(f"[select] skipped {before - len(tasks)} GPU task(s) (skip_gpu_tasks)")
    if limit:
        tasks = tasks[: int(limit)]
    return tasks


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--full", action="store_true",
                    help="phased: run A, label, then B/C/C-oracle")
    ap.add_argument("--build-only", action="store_true",
                    help="build/validate E2B templates for selected tasks, then exit "
                         "(no model calls) — surfaces bad images before spending tokens")
    args = ap.parse_args()

    from config.env import load_env

    load_env()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = _load_experiment_cfg()
    tasks = _select_tasks(cfg)
    print(f"selected {len(tasks)} TB2 tasks: {[t.task_id for t in tasks]}")

    if args.build_only:
        prebuild_templates(tasks)
        return

    results: list[dict] = []
    if args.full:
        print("=== phase 1: Arm A ===")
        results += run_matrix(tasks, seeds, ["A"], args.max_steps)
        print("=== phase 2: labeling ===")
        _label_and_backfill()
        print("=== phase 3: Arms B / C / C-oracle ===")
        results += run_matrix(tasks, seeds, ["B", "C", "C-oracle"], args.max_steps)
    else:
        arms = ALL_ARMS if args.arms == "all" else [a for a in args.arms.split(",") if a]
        results += run_matrix(tasks, seeds, arms, args.max_steps)

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
