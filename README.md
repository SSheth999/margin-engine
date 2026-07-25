# Margin Engine — Experiment Rig

A controlled experiment testing one hypothesis (see `AGENTS.MD` for the full spec):

> A gateway that detects *why* an agent run is going over budget and applies the
> *matching* fix reduces worst-case cost-per-outcome (P99 variance) without making the
> agent worse at its job.

This repo is the smallest rig that can prove or disprove that on real agent
trajectories — not the production product.

## How it works

An agent runs a task in a loop, sending every model call through a **gateway**. The
gateway tracks cumulative cost against the task's *margin ceiling*
(`price × (1 − target_margin)`), and — depending on the experiment **arm** — may
intervene before a call:

| Arm | Policy |
|-----|--------|
| **A** control | never intervene (baseline) |
| **B** generic | downshift to a cheaper model once over budget-trend |
| **C** matched | diagnose the failure mode, apply the *matching* fix |
| **C-oracle** | same as C but using the *true* mode from post-hoc labeling |

Failure mode → matched intervention:

| Mode | Detection (heuristic) | Fix |
|------|-----------------------|-----|
| tool_loop | same (tool, args) ≥3× | restrict that tool |
| context_blowout | context growth + rising cost | compact messages |
| wrong_model | high cost, flat context, no loop | downshift |
| inherent_difficulty | expensive from early, no waste | block / escalate |

The arm switch lives in exactly one place — `gateway/arms.py` — so A/B/C/C-oracle
share an identical agent, environment, tasks, provider, and state store. That is what
makes the comparison valid.

## Layout

```
gateway/       schema, cost, state_store, detectors, interventions, arms, run_log, server
providers/     base + ollama / anthropic / mock adapters (swappable behind one interface)
agent/         hand-rolled ReAct harness, tools, outcome-context headers
orchestrator/  local (subprocess) + e2b (TB2 sandbox) run-matrix drivers
analysis/      labeling, metrics, plots, report (verdict)
config/        providers.yaml, rate_card.yaml, detection.yaml, tasks.yaml
tests/         cost / adapter / detectors / interventions / arms
```

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pytest tests/ -q          # 23 tests, no network
```

## Run the matrix (local)

```bash
# full phased matrix: Arm A -> label true modes -> Arms B/C/C-oracle
./.venv/bin/python -m orchestrator.local_orchestrator --full --seeds 1,2,3 --max-steps 12

# verdict + per-mode table + distribution plot
./.venv/bin/python -m analysis.report
```

Outputs: one JSON run log per run under `runs/`, and `analysis_out/report.json` +
`cost_distributions.png`.

## Run the matrix on Terminal-Bench 2 tasks (E2B)

Real agent tasks from [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)
(~89 tasks), each executed in an isolated [E2B](https://github.com/e2b-dev/E2B) sandbox
(open-source, Apache-2.0, self-hostable) built from the task's Harbor Docker image.

```bash
# 1. vendor the task defs (sparse clone of tasks/)
bash scripts/fetch_terminal_bench.sh

# 2. pick which tasks to run + economics in config/experiment.yaml
#    (include_tasks: [...], limit: N, skip_gpu_tasks, price/margin per difficulty)

# 3. credentials: copy .env.example -> .env and fill in
#    E2B_API_KEY (+ E2B_DOMAIN if self-hosted) and ANTHROPIC_API_KEY.
#    The orchestrator auto-loads .env (python-dotenv).
cp .env.example .env    # then edit

# 4. real provider (TB2 tasks need a capable model)
#    set active_provider: anthropic in config/providers.yaml

# 5. OPTIONAL but recommended: build/validate E2B templates first (no model calls),
#    so incompatible images surface before you spend tokens.
./.venv/bin/python -m orchestrator.e2b_orchestrator --build-only

# 6. run the phased matrix, then the verdict
./.venv/bin/python -m orchestrator.e2b_orchestrator --full --seeds 1,2,3 --max-steps 30
./.venv/bin/python -m analysis.report
```

Per run: each Harbor `docker_image` is built once into an E2B template
(`Template.from_image(...).build(...)`, cached in `config/e2b_templates.json`); a sandbox
is created from it; the agent's sole tool (`run_command`) execs bash inside it via the
gateway; on finish the task's `tests/` are injected and `test.sh` is run, and
`/logs/verifier/reward.txt == "1"` sets `resolved`. The gateway/arms/detectors/
interventions/analysis are all shared with the local path — only the execution substrate
(`agent/exec_env.py`: `E2BExecEnv`) differs.

> E2B builds sandboxes from templates, so Harbor images are wrapped into E2B templates on
> first use (E2B currently supports Debian-based images). GPU tasks are skipped by default
> (`skip_gpu_tasks: true`) unless your E2B instance provides GPU sandboxes.

## Providers (swappable — one line, no code change)

Set `active_provider` in `config/providers.yaml`:

- **`mock`** — deterministic test double / failure-mode simulator. Zero inference,
  zero network. Proves the whole pipeline (what the current results use).
- **`ollama`** — local Ollama (OpenAI-compatible) for free dev iteration on a real
  model. Requires `ollama serve` + a pulled Qwen pair; edit `model_pair` to the tags
  you pulled.
- **`anthropic`** — native Anthropic SDK for the credible final matrix. Requires
  `ANTHROPIC_API_KEY`. Adapter is unit-tested against a mocked SDK, so switching needs
  no code change.

Costs come from real token counts × the per-model rate card in `config/rate_card.yaml`
(synthetic for mock/ollama, real published pricing for anthropic).

## Status vs the AGENTS.MD build order

- [x] 1. Gateway skeleton (proxy, cost, state, logging, fail-open) — Arm A
- [x] 2. Local task set + Arm A validation
- [x] 3. Heuristic detection (shadow-logged on every arm)
- [x] 4. Interventions + Arms B/C
- [x] 5. Post-hoc labeling + Arm C-oracle
- [x] 7. Analysis (metrics, bootstrap CIs, non-inferiority test, per-mode table, plot)
- [x] 6. Real E2B + Terminal-Bench 2 wiring (loader, sandbox exec, verifier, orchestrator)
- [ ] Live TB2 run — needs `E2B_API_KEY` + `ANTHROPIC_API_KEY` (code is ready; not yet executed here)
- [ ] Real-model validation run (flip provider to ollama/anthropic)

The `mock` results are for **wiring validation**, not the paper's finding: whether
matched (C) beats generic (B) is the empirical question the real-model runs answer.
The mock is deliberately not rigged to guarantee that outcome.
