"""Local orchestrator — the run-matrix driver for the local-first phase.

For each (task, arm, seed) triple it:
  1. creates an isolated workspace and seeds the task's files,
  2. spawns one gateway process (uvicorn on an ephemeral port) with run-level context
     passed via env vars,
  3. waits for the gateway to become healthy,
  4. runs the agent in-process against that gateway,
  5. tears the gateway down.

The gateway writes the JSON run log itself on /v1/finalize. This module just drives
the matrix and prints a summary. This pass runs Arm A only (see ARMS below); adding
B/C/C-oracle later is a one-line change here plus the arms.py switch.

Real E2B sandboxing replaces steps 2/3/5 in e2b_orchestrator.py.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "config"
RUNS_DIR = REPO / "runs"
WORKSPACES_DIR = RUNS_DIR / "workspaces"
LABELS_PATH = CONFIG_DIR / "labels.json"

ALL_ARMS = ["A", "B", "C", "C-oracle"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_tasks() -> list[dict]:
    with open(CONFIG_DIR / "tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]


def _primary_model() -> str:
    with open(CONFIG_DIR / "providers.yaml") as f:
        cfg = yaml.safe_load(f)
    active = cfg["active_provider"]
    return cfg["providers"][active]["model_pair"]["primary"]


def _seed_workspace(task: dict, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for relpath, content in (task.get("files") or {}).items():
        p = workspace / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _wait_healthy(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"gateway on port {port} did not become healthy")


def _spawn_gateway(port: int, arm: str, task: dict, seed: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "MARGIN_ARM": arm,
            "MARGIN_TASK_ID": task["task_id"],
            "MARGIN_SEED": str(seed),
            "MARGIN_COMPLEXITY": task["complexity_class"],
            "MARGIN_RUN_DIR": str(RUNS_DIR),
        }
    )
    if LABELS_PATH.exists():
        env["MARGIN_LABELS"] = str(LABELS_PATH)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gateway.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc


TOY_SYSTEM_PROMPT = """You are a capable autonomous agent working inside a sandboxed workspace.
You solve the given task using the provided tools. Think step by step. Use tools to
inspect and modify files. When the task is fully complete, reply with a final message
(no tool call) that begins with 'DONE:' followed by a short summary of what you did.
Do not repeat the same tool call with identical arguments — if a tool isn't helping,
change approach or finish."""


def _grade(task: dict, final_text: str, workspace: Path, finished: bool) -> bool:
    """Toy grader: success_substring in final text and/or check_file contents."""
    ok = True
    checked = False
    sub = task.get("success_substring")
    if sub is not None:
        checked = True
        ok = ok and (sub.lower() in final_text.lower())
    cf = task.get("check_file")
    if cf is not None:
        checked = True
        p = workspace / cf["path"]
        if not p.exists():
            ok = False
        elif "contains" in cf:
            ok = ok and (cf["contains"] in p.read_text())
    return ok if checked else finished


def run_matrix(
    tasks: list[dict], seeds: list[int], arms: list[str], max_steps: int
) -> list[dict]:
    from agent.harness import run_agent
    from agent.tools import TOOL_SCHEMAS, run_tool

    model = _primary_model()
    results = []

    for arm in arms:
        for task in tasks:
            for seed in seeds:
                run_id = f"{task['task_id']}__{arm}__seed{seed}"
                workspace = WORKSPACES_DIR / run_id
                _seed_workspace(task, workspace)

                port = _free_port()
                proc = _spawn_gateway(port, arm, task, seed)
                gateway_url = f"http://127.0.0.1:{port}"
                try:
                    _wait_healthy(port)
                    summary = run_agent(
                        gateway_url=gateway_url,
                        model=model,
                        outcome_id=task["task_id"],
                        contracted_price=float(task["contracted_price"]),
                        target_margin=float(task["target_margin"]),
                        prompt=task["prompt"],
                        tool_schemas=TOOL_SCHEMAS,
                        tool_runner=lambda n, a, ws=workspace: run_tool(n, ws, a),
                        resolve_fn=lambda text, fin, t=task, ws=workspace: _grade(
                            t, text, ws, fin
                        ),
                        system_prompt=TOY_SYSTEM_PROMPT,
                        max_steps=max_steps,
                        seed=seed,
                    )
                    summary["run_id"] = run_id
                    results.append(summary)
                    print(
                        f"[{run_id}] resolved={summary.get('resolved')} "
                        f"cost={summary.get('final_cost'):.4f} "
                        f"margin_held={summary.get('margin_held')}"
                    )
                except Exception as e:
                    print(f"[{run_id}] ERROR: {e!r}")
                    results.append({"run_id": run_id, "error": repr(e)})
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()

    return results


def _label_and_backfill() -> None:
    """Between-phase step: label Arm A runs, write labels.json, backfill true modes."""
    from analysis.labeling import backfill_true_mode, build_labels

    labels = build_labels(RUNS_DIR)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))
    n = backfill_true_mode(labels, RUNS_DIR)
    print(f"[labeling] {len(labels)} labels written, backfilled into {n} logs")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2", help="comma-separated seed list")
    ap.add_argument("--task", default=None, help="run only this task_id")
    ap.add_argument("--arms", default="A", help="comma-separated arms, or 'all'")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument(
        "--full",
        action="store_true",
        help="phased full matrix: run A, label, then run B/C/C-oracle",
    )
    args = ap.parse_args()

    from config.env import load_env

    load_env()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = _load_tasks()
    if args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]
        if not tasks:
            print(f"no task with id {args.task!r}")
            sys.exit(1)

    results: list[dict] = []
    if args.full:
        # C-oracle depends on labels from Arm A, so A must complete and be labeled first.
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
