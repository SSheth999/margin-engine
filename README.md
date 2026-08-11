# Margin Engine — Experiment Rig

A controlled experiment testing one hypothesis (see `AGENTS.MD` for the full spec):

> A gateway that detects *why* an agent run is going over budget and applies the
> *matching* fix reduces worst-case cost-per-outcome (P99 variance) without making the
> agent worse at its job.

This repo is the smallest rig that can prove or disprove that on real agent
trajectories — not the production product.

## Results (full Terminal-Bench 2 benchmark)

**Setup:** all 89 Terminal-Bench 2.1 (Harbor) tasks × 4 arms × 1 seed = **349 completed
runs** (7 tasks have images that can't run in every arm). Agent model **gpt-5.6** driving
real tasks in isolated **E2B** sandboxes; **gpt-5-mini** as the downshift target;
objective pass/fail grading via each task's own test suite. Per-difficulty margin ceilings
were calibrated to the observed gpt-5.6 cost regime so tasks *can* blow budget. Total
real spend: **$34.64**, with **569 interventions** fired.

| arm | mean | P90 | **P99** | std | **margin-blown** | resolved |
|-----|------|-----|---------|-----|------------------|----------|
| **A** control | $0.123 | $0.282 | $0.452 | 0.112 | **41.4%** | 29.9% |
| **B** generic downshift | $0.062 | $0.097 | **$0.117** | 0.028 | **2.3%** | 23.0% |
| **C** matched | $0.086 | $0.159 | $0.215 | 0.055 | 32.2% | 26.4% |
| **C-oracle** (true mode) | $0.126 | $0.305 | $0.549 | 0.129 | 37.5% | 23.9% |

**Verdict: the "matching" hypothesis is *not* supported — generic downshift wins.**

- **Generic downshift (B) dominates on cost/variance:** it cut margin-blown from **41% → 2.3%**
  and P99 nearly 4× vs the A control. Blunt "switch to a cheaper model when trending over
  budget" is by far the most effective lever.
- **Matched (C) beats the control but loses decisively to B** (32% vs 2.3% blown). Matching
  the intervention to the failure mode did *not* beat the generic policy.
- **C-oracle — a *perfect* detector — has the worst tail of all** ($0.549 P99). Applying the
  "correct" matched fix actively hurt.

**Why (per-failure-mode breakdown):** the matched interventions are weak cost controllers.
For **context_blowout** (the dominant mode), downshift → 2.4% blown while *compaction* (the
matched fix) → 45–61% blown. For **tool_loop**, downshift → 12.5% blown while *tool
restriction* (the matched fix) → 87.5%. The clever fixes barely move cost; the dumb generic
one dominates. There is a real cost/quality trade-off: B saves the most money but also loses
the most resolution (23% vs A's 30%).

This is a **statistically-backed negative result** (n≈87/arm): at full benchmark scale,
budget interventions clearly help, but *matching the fix to the diagnosed failure mode does
not beat simply downshifting.* Distribution plot + per-mode table in `analysis_out/`.

> Caveats: one model (gpt-5.6 at reasoning_effort≈none), one seed, ceilings synthesized
> from difficulty, and gpt-5.6 pricing is approximate. Directional, not the last word —
> more seeds would tighten the CIs.

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

## Architecture

One isolated run = one `(task, arm, seed)` triple. The orchestrator spins up a sandbox
and a gateway, drives the agent, verifies, and tears everything down. Every model call
flows through the gateway's two interceptors:

```
┌──────────────────────── one run: (task × arm × seed) ────────────────────────┐
│                                                                              │
│   ORCHESTRATOR ── spawns ──► GATEWAY (uvicorn, ephemeral port)               │
│        │                         ▲          │                               │
│        │ creates                 │ HTTP     │ native SDK / HTTP             │
│        ▼                         │          ▼                               │
│   E2B SANDBOX ◄── run_command ── AGENT     PROVIDER (openai / anthropic /   │
│   (Harbor image) exec in sandbox (ReAct)    ollama / mock)                  │
│        │                                                                     │
│        └── verify: inject tests/ → run test.sh → read reward.txt            │
│                                                                              │
│   GATEWAY per model call:                                                    │
│     PRE-CALL   1. read outcome state (cum cost, steps, ctx sizes, tool hist) │
│                2. cost_ratio = cum_cost / margin_ceiling                     │
│                3. detect() failure mode (shadow, logged on every arm)        │
│                4. arms.decide(arm, …) → passthrough | rewrite req | block    │
│     ── forward to provider ──►                                               │
│     POST-CALL  5. cost = usage × rate card                                   │
│                6. update state store   7. append per-step record to run log  │
│                                                                              │
│   fail-open: any gateway-logic error → forward unchanged + log it            │
└──────────────────────────────────────────────────────────────────────────────┘
              │ one JSON run log per run  ──►  runs/<task>__<arm>__seed<n>.json
              ▼
   ANALYSIS (offline): post-hoc labeling → metrics (P99, blown%, bootstrap CIs,
   non-inferiority test) → per-mode breakdown → verdict + distribution plot
```

**Component responsibilities**

| Component | File(s) | Responsibility |
|---|---|---|
| **Gateway server** | `gateway/server.py` | FastAPI proxy, one process per run. Parses outcome headers, runs the pre/post interceptors, fails open. |
| **Canonical schema** | `gateway/schema.py` | Provider-agnostic `ChatRequest` / `ChatResponse` / `Usage`. Interventions mutate this shape, never a wire format. |
| **State store** | `gateway/state_store.py` | Per-outcome cum cost, step count, context-size history, `(tool, args_hash)` history. In-memory (one outcome per gateway). |
| **Cost** | `gateway/cost.py` | `usage × rate_card` with additive cached-token discount. |
| **Detectors** | `gateway/detectors.py` | The four heuristic rules → `DetectionResult`. Also `mode_triggered()` for C-oracle timing parity. |
| **Interventions** | `gateway/interventions.py` | `downshift` / `compact` / `restrict_tool`, operating on `ChatRequest`. `block` is signalled by the arm. |
| **Arm switch** | `gateway/arms.py` | The *only* per-arm fork. Maps detected/true mode → intervention. |
| **Providers** | `providers/*` | One adapter per backend behind `Provider`. Translate canonical ↔ wire, normalize usage. |
| **Agent** | `agent/harness.py` | Fixed hand-rolled ReAct loop. Pluggable tools + grader + exec env; identical across arms. |
| **Exec env** | `agent/exec_env.py` | `LocalExecEnv` (subprocess) vs `E2BExecEnv` (sandbox) behind one `ExecEnv` protocol. |
| **Orchestrators** | `orchestrator/*` | Drive the run matrix (phased A → label → B/C/C-oracle). Resumable; skip already-logged runs. |
| **Analysis** | `analysis/*` | Labeling, metrics, plots, verdict — fully offline, reads `runs/`. |

## Technical details

**Failure-mode detection** (`detectors.py`) — deterministic heuristics over the trajectory,
gated on a `trigger_cost_ratio` (default 0.60) so they never fire on cheap/easy runs:
- `tool_loop` — the most-repeated `(tool_name, args_hash)` occurs ≥3× (fires regardless of
  cost, since a tight loop blows budget fast).
- `context_blowout` — context size grew ≥1.5× over the last 3 steps *and* cost is trending over.
- `inherent_difficulty` — cost ratio ≥0.90 within the first ~4 steps, no waste signal.
- `wrong_model` — cost trending over, context flat, no loop.

**Interventions** (`interventions.py`) operate on the canonical `ChatRequest`, so they're
provider-agnostic by construction:
- `downshift` — swap `model` to the cheap sibling.
- `compact` — drop stale middle turns, keeping system + the first (task) turn + the last N
  turns. Works at **turn granularity** so a `tool_use` is never split from its `tool_result`
  (both wire formats reject an orphaned tool result); the elision note folds into the task
  message to avoid two consecutive user turns (which Anthropic rejects).
- `restrict_tool` — remove the looping tool from the `tools` array.
- `block` — the arm returns `Decision(block=True)`; the gateway returns a stop signal the
  agent catches to end gracefully.

**C-oracle timing parity** — C-oracle differs from C *only* in which mode it acts on
(true, from post-hoc labeling) vs detected. `mode_triggered()` makes it fire at the same
point C would for that mode, so the comparison isolates *diagnosis correctness* from *timing*.

**Provider abstraction** — the canonical schema means the agent, gateway logic, and analysis
never touch a wire format. Adapters: `openai_provider` (Chat Completions; auto-handles GPT-5
reasoning models — `max_completion_tokens`, no temperature, `reasoning_effort` with a
`none → minimal` fallback), `anthropic_provider` (native SDK, system-param + tool_use blocks),
`ollama_provider` (local, OpenAI-compatible), `mock_provider` (deterministic double / failure
simulator). Switching backends is one line in `config/providers.yaml`.

**Terminal-Bench verification** — the task's `tests/` are injected into the sandbox *after*
the agent finishes (so it can't read/game them), `test.sh` runs (installs `uv`, runs pytest
with `pytest-json-ctrf`), and `/logs/verifier/reward.txt == "1"` sets `resolved`. Objective,
per-task, not model-judged.

**Analysis methodology** (`analysis/`) — post-hoc labeling assigns each run a
`true_failure_mode` from its whole trajectory. Metrics lead with tail/variance (P99, P90,
margin-blown rate, spread), report mean secondarily, add **percentile-bootstrap CIs** on the
noisy tail metrics, and run a **two-proportion non-inferiority z-test** on resolution rate
(no scipy). Everything breaks down per failure mode.

**Engineering notes** (hard-won during the full run):
- **Resumable sweeps** — every completed run is a JSON file on disk; the orchestrator skips
  existing `(task, arm, seed)` logs, so a killed sweep (laptop sleep, etc.) relaunches and
  picks up where it left off. Template builds are cached the same way.
- **Fail-open everywhere** — gateway logic errors never crash a run; provider errors are
  surfaced to the agent, which ends gracefully.
- **E2B specifics** — Harbor images are wrapped into E2B templates on first use (Debian-based
  only); sandbox timeout is clamped to E2B's 1-hour cap; a failed template build skips that
  task instead of sinking the whole sweep.

## Layout

```
gateway/       schema, cost, state_store, detectors, interventions, arms, run_log, server
providers/     base + openai / anthropic / ollama / mock adapters (one swappable interface)
agent/         hand-rolled ReAct harness, tools, terminal_tools, exec_env, outcome-context
tasks/         tb_loader (Harbor task.toml), tb_verify (inject tests → reward)
orchestrator/  local (subprocess) + e2b (TB2 sandbox) run-matrix drivers, both resumable
analysis/      labeling, metrics, plots, report (verdict)
config/        providers.yaml, rate_card.yaml, detection.yaml, tasks.yaml, experiment.yaml, env
scripts/       fetch_terminal_bench.sh (sparse-clone the ~89 TB2 tasks)
tests/         cost / adapters / detectors / interventions / arms / tb_loader / e2b_exec_env
```

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pytest tests/ -q          # 28 tests, no network
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
#    E2B_API_KEY (+ E2B_DOMAIN if self-hosted) and OPENAI_API_KEY (or ANTHROPIC_API_KEY).
#    The orchestrator auto-loads .env (python-dotenv).
cp .env.example .env    # then edit

# 4. real provider (TB2 tasks need a capable model)
#    set active_provider: openai in config/providers.yaml (the full run used gpt-5.6)

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

- **`openai`** — native OpenAI Chat Completions, incl. the GPT-5 reasoning family
  (**what the full-benchmark results above used**: `gpt-5.6-terra` primary + `gpt-5-mini`
  downshift). Requires `OPENAI_API_KEY`.
- **`anthropic`** — native Anthropic SDK (Claude). Requires `ANTHROPIC_API_KEY`.
- **`ollama`** — local Ollama (OpenAI-compatible) for free dev iteration on a real model.
  Requires `ollama serve` + a pulled model pair (e.g. a Qwen large/small pair).
- **`mock`** — deterministic test double / failure-mode simulator. Zero inference, zero
  network. Used to validate the whole pipeline before spending on a real provider.

Each adapter is exercised by a unit test (mocked SDK where relevant), so switching backends
needs no code change. Costs come from **real token counts × the per-model rate card**
(`config/rate_card.yaml`) — synthetic rates for mock/ollama, real pricing for hosted APIs.

## Status vs the AGENTS.MD build order

- [x] 1. Gateway skeleton (proxy, cost, state, logging, fail-open) — Arm A
- [x] 2. Local task set + Arm A validation
- [x] 3. Heuristic detection (shadow-logged on every arm)
- [x] 4. Interventions + Arms B/C
- [x] 5. Post-hoc labeling + Arm C-oracle
- [x] 7. Analysis (metrics, bootstrap CIs, non-inferiority test, per-mode table, plot)
- [x] 6. Real E2B + Terminal-Bench 2 wiring (loader, sandbox exec, verifier, orchestrator)
- [x] Live full-benchmark run — 349 runs across all 89 TB2 tasks × 4 arms on gpt-5.6 via E2B
      (see **Results** above)

The `mock` results are for **wiring validation**, not the paper's finding: whether
matched (C) beats generic (B) is the empirical question the real-model runs answer.
The mock is deliberately not rigged to guarantee that outcome.
