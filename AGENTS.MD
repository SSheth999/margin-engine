# AGENT.md — Margin Engine Experiment

## What you are building

A controlled experiment that tests one hypothesis:

> **A gateway that detects *why* an AI agent run is going over budget and applies the *matching* fix reduces the worst-case cost per outcome (P99 variance) without making the agent worse at its job.**

You are NOT building the production product. You are building the smallest rig that can prove or disprove the hypothesis on real agent trajectories. Optimize for a clean, reproducible result — not for scale, auth, multi-tenancy, or polish.

---

## The core idea in one paragraph

An agent runs tasks in a sandbox. Each task has a *contracted price* and a *margin ceiling* (price × (1 − target_margin)). A gateway sits between the agent and the model provider, tracks cumulative cost per task in real time, and — depending on the experiment arm — intervenes before a call when the task is trending over its ceiling. We compare cost-per-outcome distributions and resolution quality across arms. The win condition is variance reduction (P99), not mean cost reduction, with resolution quality held within a non-inferiority margin.

---

## System architecture

```
┌──────────────────────────────────────────────────────────────┐
│  DAYTONA ORCHESTRATOR                                        │
│  - reads run matrix (task × arm × seed)                      │
│  - spawns one Docker sandbox per run                         │
│  - collects the JSON log record when the run finishes        │
└───────────────────────────┬──────────────────────────────────┘
                            │ one isolated sandbox per run
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  DOCKER SANDBOX (one Harbor task run)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  AGENT (fixed backbone, ReAct/LangGraph-style)      │     │
│  │  - runs the task, makes tool calls + model calls    │     │
│  │  - model calls point at the GATEWAY, not provider   │     │
│  │  - carries outcome context in request headers:      │     │
│  │    X-Outcome-Id, X-Contracted-Price, X-Target-Margin│     │
│  └───────────────────────────┬────────────────────────┘     │
│                              │ model call (OpenAI/Anthropic  │
│                              │ compatible request)           │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  GATEWAY (the experiment rig)                       │     │
│  │                                                    │     │
│  │  PRE-CALL INTERCEPTOR                               │     │
│  │   1. read outcome state (cum cost, steps, context, │     │
│  │      tool-call history) from STATE STORE           │     │
│  │   2. cost_ratio = cum_cost / margin_ceiling        │     │
│  │   3. DETECT failure mode (heuristic rules)         │     │
│  │   4. DECIDE per arm:                                │     │
│  │        A → always passthrough                       │     │
│  │        B → generic intervention at threshold        │     │
│  │        C → matched intervention for detected mode   │     │
│  │        C-oracle → matched intervention for TRUE mode│     │
│  │   5. APPLY: rewrite request (downshift / compact /  │     │
│  │      restrict tool) OR return stop signal (block)   │     │
│  │                                                    │     │
│  │  ── forwards to provider ──►  (native client)      │     │
│  │                                                    │     │
│  │  POST-CALL INTERCEPTOR                              │     │
│  │   6. compute cost (input+output−cache, rate card)  │     │
│  │   7. update STATE STORE (cum cost, step, context,  │     │
│  │      tool hash)                                     │     │
│  │   8. append per-step record to RUN LOG             │     │
│  └───────────────────────────┬────────────────────────┘     │
│                              │                               │
│         ┌────────────────────┴───────────────┐              │
│         ▼                                     ▼              │
│  ┌───────────────┐                   ┌──────────────────┐   │
│  │ STATE STORE   │                   │ PROVIDER          │   │
│  │ (Redis or     │                   │ Anthropic /       │   │
│  │  in-memory)   │                   │ OpenAI (native    │   │
│  │ per-outcome   │                   │ client, NOT       │   │
│  │ counters      │                   │ LiteLLM)          │   │
│  └───────────────┘                   └──────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  RUN LOG (one JSON record per run)                  │     │
│  │  final cost, margin_held, resolved, step_count,     │     │
│  │  detected + true failure mode, interventions[],     │     │
│  │  per_step[] trajectory                              │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
                            │ JSON log records from all runs
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  ANALYSIS (offline, after the matrix completes)             │
│  - post-hoc failure-mode labeling → true_failure_mode        │
│  - metrics: P99, P90, margin-blown rate, spread, mean        │
│  - non-inferiority test on resolution rate                   │
│  - per-failure-mode breakdown                                │
│  - distribution plots + bootstrap CIs                        │
│  - verdict against success criteria                          │
└──────────────────────────────────────────────────────────────┘
```

**Component responsibilities:**

- **Daytona orchestrator** — owns the run matrix, spawns sandboxes, collects logs. Knows nothing about interventions; it just runs (task, arm, seed) triples and gathers results.
- **Agent** — unchanged across arms. Points its model calls at the gateway. Carries outcome context in headers. Must be able to catch a "block" stop signal and end gracefully.
- **Gateway** — the only thing that changes behavior by arm. Pre-call interceptor makes the decision; post-call interceptor does the accounting. Fails open on any error.
- **State store** — per-outcome running cost, step count, context size history, tool-call (name, args-hash) history. Redis if you want it realistic; in-memory dict is fine for a single-sandbox run.
- **Provider** — native Anthropic/OpenAI client for exact token + cache accounting. Never LiteLLM in the experiment.
- **Run log** — the single source of truth for analysis. Per-step trajectory logged on every run, even when nothing intervenes, so post-hoc labeling and the C-oracle arm are possible.
- **Analysis** — fully offline, reads logs, produces the verdict. Never touches the live run.

**The arm switch is the only behavioral variable.** A, B, C, and C-oracle share the exact same agent, environment, task set, provider, and state store. The only difference is step 4 of the pre-call interceptor. Keep that switch in one place in the code so the arms can't drift apart in any other way — that is what makes the comparison valid.

---

## Experiment design — FOUR arms

Run the SAME task set, SAME agent, SAME environment across all arms. Only the gateway policy changes.

| Arm | Policy | Purpose |
|-----|--------|---------|
| **A — Control** | No intervention. Agent runs untouched. | Baseline cost + quality |
| **B — Generic** | One fixed intervention (model downshift) at the trigger threshold. No failure-mode diagnosis. | Isolates whether *matching* matters vs intervention alone |
| **C — Failure-mode-centric** | Detect the failure mode, apply the matching intervention. | The tested policy |
| **C-oracle** | Same as C, but given the TRUE failure mode (from post-hoc labeling) instead of the detector's guess. | Isolates the *matching idea* from *detector quality* |

**Arm B is non-negotiable.** Prior work (SUPERVISORAGENT, ICLR 2026) already showed generic graduated intervention reduces variance. Our novel claim is that failure-mode *matching* beats generic. If C only beats A but ties B, the contribution collapses. B is what makes the result meaningful.

**C-oracle matters** because if C ties B but C-oracle beats B, we learn the matching idea is right but the detector needs work — a useful, publishable result rather than an ambiguous null.

---

## Environment

- **Orchestration:** Daytona managing Docker sandboxes, one per task run, fully isolated.
- **Tasks:** Harbor-style agent tasks (multi-step, real tool calls). Must produce genuine failure modes, not staged ones.
- **Agent:** one fixed agent backbone with tool-calling (e.g. a LangGraph or equivalent ReAct-style agent). Same across all arms.
- **Provider:** real model calls (Anthropic and/or OpenAI). Native client — do NOT route through LiteLLM for the experiment (we want exact token/cost accounting).
- **Reproducibility:** fixed task set, fixed seeds list, all config in version control. Every run must be re-runnable from a config file.

---

## The four failure modes and their matched interventions

| Failure mode | Detection signal (heuristic — fine for MVP) | Matched intervention |
|--------------|---------------------------------------------|----------------------|
| **Tool loop** | Same (tool_name, args_hash) repeated ≥3 times | Restrict that tool for remaining steps |
| **Context blowout** | Context growth rate above threshold AND cost ratio rising | Compact context (drop expired/redundant messages, keep system + recent N turns) |
| **Wrong model** | Cost ratio high, but context flat and no tool repeats | Downshift to cheaper sibling model |
| **Inherent difficulty** | Cost tracks the class-high from early steps, no waste signal | Block / early escalation (end run, mark for handoff) |

Use HEURISTIC detection for this experiment. Do NOT build a trained classifier or fine-tuned model yet. The novelty being tested is the *matching*, not the sophistication of the detector. Keep detection simple, deterministic, and logged.

---

## Gateway requirements (minimal)

The gateway is an OpenAI/Anthropic-compatible proxy. It must:

1. **Accept outcome context** per request via headers: `X-Outcome-Id`, `X-Contracted-Price`, `X-Target-Margin`.
2. **Track per-outcome state** in memory or Redis: cumulative cost, step count, per-step cost, context size per step, tool-call (name, args-hash) history.
3. **Compute cost accurately** per call: input tokens × input rate + output tokens × output rate − cached tokens × cache discount. Maintain a per-model rate card in config.
4. **Pre-call decision** (the interceptor): read current state, compute cost_ratio = cumulative_cost / margin_ceiling, and — per arm — decide passthrough vs intervene.
5. **Apply the intervention** by rewriting the outgoing request (downshift = change model param; compact = trim messages; restrict = remove tool from tools list; block = return a structured stop signal the agent harness catches).
6. **Log everything** (see logging schema below).
7. **Fail open** — if anything errors, pass the request through unchanged and log the error. Never crash a task run because of gateway logic.

Latency is NOT a concern for the experiment. Correctness and logging are.

---

## Per-task configuration

Each task in the task set needs:

```
task_id
complexity_class      # easy | medium | hard
contracted_price      # dollars, e.g. 1.50
target_margin         # e.g. 0.70  -> ceiling = 1.50 * (1 - 0.70) = 0.45
expected_failure_mode # for hard tasks: which mode it tends to exhibit (for validation)
```

Task set composition:
- ~40% easy (resolve cheaply, well under ceiling — Margin Engine should do NOTHING here)
- ~35% medium (sometimes drift over ceiling)
- ~25% hard / failure-prone (reliably exhibit loops, context blowout, or inherent difficulty)

**Validate the task set BEFORE the real run:** run Arm A alone, inspect where cost blows up, confirm the hard tasks actually produce the intended failure modes. If they don't, fix the tasks. A task set that never blows margin proves nothing.

---

## Run matrix

```
N tasks × 4 arms × K seeds
```
- Start with N ≈ 40–50, K = 3–5 seeds (agent behavior is stochastic; you need repeats).
- Parallelize across Daytona sandboxes.
- Every (task, arm, seed) triple is one isolated run producing one log record.

---

## Logging schema (per run)

Write one structured record (JSON) per run:

```json
{
  "task_id": "...",
  "arm": "A | B | C | C-oracle",
  "seed": 3,
  "complexity_class": "hard",
  "contracted_price": 1.50,
  "margin_ceiling": 0.45,
  "final_cost": 0.71,
  "margin_held": false,
  "resolved": true,
  "step_count": 14,
  "detected_failure_mode": "tool_loop",
  "true_failure_mode": "tool_loop",
  "interventions": [
    {"step": 5, "mode": "tool_loop", "action": "restrict_tool", "cost_at_step": 0.19}
  ],
  "per_step": [
    {"step": 1, "model": "...", "input_tokens": 1200, "output_tokens": 380,
     "cost": 0.028, "cum_cost": 0.028, "context_size": 1580,
     "tool": "search", "args_hash": "abc123"}
  ]
}
```

Log per-step data even when no intervention fires — the trajectory is needed for post-hoc failure-mode labeling (for the true_failure_mode field and C-oracle arm).

---

## Metrics (compute these from the logs)

**Primary — variance / tail (this is the hypothesis):**
- P99 cost-per-outcome, per arm
- P90 cost-per-outcome, per arm
- Margin-blown rate = % of runs where final_cost > margin_ceiling, per arm
- Standard deviation / spread of cost-per-outcome, per arm

**Secondary — mean (report, do not lead with):**
- Mean cost-per-outcome, per arm. Expect it to move LESS than P99 — that is the point (effect is variance, not average).

**Constraint — quality (non-inferiority guardrail):**
- Resolution rate (% resolved), per arm. Arm C must be within a preset non-inferiority margin (e.g. 2–3 pp) of Arm A.
- Also check resolution rate holds across seeds (durability).

**Mechanism check (the novel claim):**
- Compare C vs B on P99 and margin-blown rate. C must beat B, not just A.
- Compare C vs C-oracle to separate matching-idea quality from detector quality.
- Break all metrics down BY failure mode — show which modes intervention helps and which it doesn't.

---

## Analysis requirements

- Plot cost-per-outcome DISTRIBUTIONS (histograms) per arm, not just point stats. The story is the shape of the tail.
- Bootstrap confidence intervals on P99 and margin-blown rate (tail metrics are noisy).
- Run an actual non-inferiority test on resolution rate (not eyeballing "looks similar").
- Produce a per-failure-mode breakdown table.

---

## Success criteria (state the verdict explicitly)

The hypothesis is SUPPORTED if ALL of:
1. Arm C shows materially lower P99 cost-per-outcome AND lower margin-blown rate than Arm A.
2. Arm C beats Arm B on P99 and margin-blown rate (matching > generic).
3. Arm C resolution rate is within the non-inferiority margin of Arm A (quality preserved).

If C beats A but ties B → intervention helps but matching is unproven (report honestly).
If C ties B but C-oracle beats B → matching idea is right, detector needs work (report honestly).

---

## Build order

1. Gateway skeleton: proxy + outcome header parsing + cost tracking + logging. Test with Arm A (passthrough only).
2. Validate the task set on Arm A — confirm failure modes actually occur.
3. Add heuristic failure-mode detection (the four rules). Log detected mode without acting yet (shadow).
4. Add the four interventions. Wire Arm B (generic) and Arm C (matched).
5. Add post-hoc failure-mode labeling to produce true_failure_mode, enabling C-oracle.
6. Run the full matrix on Daytona.
7. Compute metrics, generate plots, write the verdict.

---

## Hard rules

- **Do NOT** build auth, multi-tenancy, dashboards, billing, or a trained classifier. Out of scope.
- **Do NOT** route through LiteLLM — use native provider clients for exact cost accounting.
- **Do NOT** let the gateway crash a run — fail open, always.
- **Do NOT** intervene on easy tasks — if interventions fire on tasks that were never going to blow margin, that is a false-positive bug; log and investigate.
- **DO** keep detection heuristic and deterministic — the novelty is matching, not the detector.
- **DO** log per-step trajectory data on every run for post-hoc labeling.
- **DO** make every run reproducible from a config file + seed.
- **DO** report honest results including the C-vs-B comparison even if it is null.

---

## Reference: prior work to be aware of

- **SUPERVISORAGENT (ICLR 2026):** graduated real-time intervention reduced token cost 35% and variance 63% in multi-agent systems. This is the baseline our Arm B approximates. Our novel claim is failure-mode *matching* (C) beats generic graduated intervention (B), tied to contracted-outcome margin. Cite it; differentiate from it.
- **Intervention Paradox (Writer, arXiv 2602.03338):** accurate failure prediction can still cause 26pt regressions. This is why quality is a hard non-inferiority constraint, and why we report per-mode breakdowns rather than assuming intervention always helps.
- **AgentDiet (arXiv 2509.23586):** trajectory-waste removal cuts tokens 40–60% with no quality loss — the basis for the compaction intervention.