# Margin Engine — Experiment Rig

A controlled experiment testing one hypothesis (see `AGENTS.MD` for the full spec):

> A gateway that detects *why* an agent run is going over budget and applies the
> *matching* fix reduces worst-case cost-per-outcome (P99 variance) without making the
> agent worse at its job.

This repo is the smallest rig that can prove or disprove that on real agent
trajectories — not the production product.

## Results (full Terminal-Bench 2 benchmark)

Everything below is the output of `python -m analysis.report` over the 349 run logs in
`runs/`, reproducible offline with no API keys. Raw numbers: `analysis_out/report.json`.

**Setup:** all 89 Terminal-Bench 2.1 (Harbor) tasks × 4 arms × 1 seed = **349 completed
runs** (7 tasks have images that can't run in every arm). Agent model **gpt-5.6** driving
real tasks in isolated **E2B** sandboxes; **gpt-5-mini** as the downshift target;
objective pass/fail grading via each task's own test suite. Per-difficulty margin ceilings
were calibrated to the observed gpt-5.6 cost regime so tasks *can* blow budget. Total
real spend: **$34.64**, with **569 interventions** fired.

### 1. Cost per outcome, by arm

Tail and variance first (that is the hypothesis); mean is secondary. Brackets are
percentile-bootstrap 95% CIs (2000 resamples).

| arm | n | mean | P90 | **P99** [95% CI] | std | **margin-blown** [95% CI] | resolved |
|-----|---|------|-----|------------------|-----|---------------------------|----------|
| **A** control | 87 | $0.1229 | $0.2819 | $0.4521 [0.343, 0.472] | 0.1121 | **41.4%** [31.0, 51.7] | 29.9% |
| **B** generic downshift | 87 | $0.0618 | $0.0965 | **$0.1170** [0.102, 0.124] | 0.0284 | **2.3%** [0.0, 5.7] | 23.0% |
| **C** matched | 87 | $0.0861 | $0.1591 | $0.2148 [0.186, 0.260] | 0.0552 | 32.2% [23.0, 42.5] | 26.4% |
| **C-oracle** (true mode) | 88 | $0.1260 | $0.3047 | $0.5486 [0.387, 0.627] | 0.1287 | 37.5% [27.3, 47.8] | 23.9% |

B is the only arm whose margin-blown CI excludes the control's. C's CI overlaps A's
heavily; C-oracle's overlaps everything.

### 2. Significance tests — is the null rejected?

Every criterion is decided by a test, not by `>` on two point estimates. Cost tests are
**paired on task** (the arms run the same tasks, and per-task cost spans orders of
magnitude, so unpaired comparisons throw away most of the signal). Positive = first arm
costs more; `!` = 95% CI excludes zero.

| comparison | margin-blown diff | p | paired cost diff [95% CI] | paired P99 diff [95% CI] | resolution p |
|---|---|---|---|---|---|
| A vs **B** | +39.1pp | **4.4×10⁻¹⁰** \*\*\* | +$0.0611 [+0.042, +0.082] ! | +$0.3352 [+0.226, +0.366] ! | 0.146 ns |
| A vs **C** | +9.2pp | 0.21 ns | +$0.0368 [+0.021, +0.054] ! | +$0.2374 [+0.105, +0.276] ! | 0.453 ns |
| **B** vs C | −29.9pp | **1.8×10⁻⁷** \*\*\* | −$0.0243 [−0.033, −0.016] ! | −$0.0978 [−0.154, −0.068] ! | 0.508 ns |
| **B** vs C-oracle | −35.2pp | **5.9×10⁻⁹** \*\*\* | −$0.0628 [−0.088, −0.040] ! | −$0.4334 [−0.521, −0.274] ! | 1.000 ns |
| A vs C-oracle | +3.9pp | 0.60 ns | −$0.0021 [−0.028, +0.022] | −$0.0981 [−0.232, +0.065] | 0.180 ns |

**Which null is rejected depends on which null you mean:**

| null hypothesis | result |
|---|---|
| "interventions don't reduce cost" (A vs B) | **REJECTED** — p = 4.4×10⁻¹⁰ on blown rate; paired P99 CI excludes 0; B cheaper on 64/87 tasks (sign p = 1.3×10⁻⁵) |
| "matching doesn't beat the control" (A vs C) | **REJECTED**, but on the paired tail only — the blown-rate gap is *not* significant (p = 0.21) |
| **"matching beats generic" (C vs B) — the actual hypothesis** | **NOT REJECTED.** The reverse is: B significantly beats C (p = 1.8×10⁻⁷) |
| "a perfect detector beats generic" (C-oracle vs B) | **NOT REJECTED.** B wins decisively (p = 5.9×10⁻⁹) |
| "any arm changes resolution quality" | **NOT REJECTED** for any pair — all paired exact tests p ≥ 0.146 |

Note the last row of the comparison table: **C-oracle is statistically indistinguishable
from the do-nothing control** (both cost CIs straddle zero). The compaction-only arm
achieved nothing at all.

**Verdict: `REJECTED IN REVERSE — generic (B) significantly beats matched (C) on the
tail, so matching as implemented is worse than the generic policy.`** With the caveat
from §8 that "as implemented" is doing real work in that sentence.

#### The quality criterion: argue about the margin, not the sample size

The non-inferiority test is now **paired** (McNemar-based), because the arms run the same
tasks and the unpaired test pays for between-task variance the design already controls.
A and C agree on 80 of 87 tasks, so only 7 discordant pairs carry information:

| | unpaired | **paired** |
|---|---|---|
| SE on the difference | 6.82pp | **3.02pp** (2.3× tighter) |
| n/arm needed for a 3pp margin | 1,217 (≈14 seeds) | **239 (≈3 seeds)** |
| margin decidable at n=87 (1 seed) | ≥11.2pp | **≥5.0pp** |

What each seed count buys, at 87 tasks:

| seeds | smallest resolution drop you can rule out |
|---|---|
| **1** | **5.0pp** |
| 2 | 3.5pp |
| 3 | 2.9pp |
| 5 | 2.2pp |

So the configured 3pp margin needs ~3 seeds, not 14. More to the point, **3pp was never a
meaningful target**: it is 2.6 tasks out of 87, while re-running the *identical* policy
flips **10.8%** of resolution outcomes (§3). Asking for a tolerance ~3.6× tighter than the
noise of the apparatus is not an underfunded experiment, it is a broken specification. A
5–6pp margin is decidable at one seed and is the honest claim this rig can support.

Either way, `c_quality_ok: False` on the current data is *not* a quality finding. The
report says `INCONCLUSIVE (underpowered by design)`; the previous `INFERIOR` was an
inferential error, since failing to establish non-inferiority is absence of evidence.
Note the tail comparisons needed none of this — they were already significant at
p < 10⁻⁷ with one seed.

Success criteria as tested:

| criterion | result |
|---|---|
| `c_beats_a` — C significantly beats A on the tail | ✅ **PASS** (paired P99 and cost) |
| `c_beats_b` — C beats B (matching > generic) | ❌ **fail** — reversed, and significantly so |
| `c_quality_ok` — C resolution non-inferior to A | ❌ **fail** — *untestable at this n* |
| `oracle_beats_b` — a perfect detector beats generic | ❌ **fail** |

### 3. Noise floor — the number that reframes everything above

C-oracle passes through unconditionally when it has no label, so on the 37 tasks with no
true failure mode it *is* arm A. That makes it an accidental placebo, and pairing it
against A measures how much two runs of the **identical policy** differ:

| statistic | value |
|---|---|
| paired runs (same policy) | 37 |
| median \|cost difference\| | **21%** |
| mean \|cost difference\| | **71%** |
| P90 \|cost difference\| | **164%** |
| median step-count difference | 1 |
| resolution flips | **4 / 37** (10.8%) |
| worst case | `fix-ocaml-gc`: A $0.0527 / 7 steps → C-oracle **$0.5244 / 30 steps** (**+896%**) |

That worst case is a run where *nothing was done differently*, and it is the third-largest
contributor to C-oracle's headline $0.5486 P99. The resolution flip rate under an identical
policy (10.8%) is larger than the −3.4pp resolution gap the non-inferiority test just
called "inferior". **Arm differences smaller than this floor are not effects.** Arm
`A-prime` now measures this deliberately instead of by accident.

### 4. Per intervention *actually applied*

Keyed on what was done to each run, not on the mode it was diagnosed with. These are
different things, and conflating them produced a false claim in the previous write-up.

| applied | arm | n | mean | P99 | margin-blown | resolved |
|---|---|---|---|---|---|---|
| `downshift` | B | 47 | $0.0827 | $0.1202 | **4.3%** | 6.4% |
| `downshift` | C | 10 | $0.1053 | $0.1550 | 40.0% | 10.0% |
| `compact+downshift` | C | 24 | $0.1495 | $0.2463 | 87.5% | 4.2% |
| `compact` | C-oracle | 22 | $0.2470 | **$0.6081** | **86.4%** | 9.1% |
| `restrict` | C | **1** | $0.1136 | $0.1136 | 0.0% | 0.0% |
| `restrict` | C-oracle | 4 | $0.1551 | $0.1992 | 75.0% | 0.0% |
| *(none)* | A | 87 | $0.1229 | $0.4521 | 41.4% | 29.9% |
| *(none)* | B | 40 | $0.0372 | $0.0760 | 0.0% | 42.5% |
| *(none)* | C | 52 | $0.0526 | $0.1532 | 5.8% | 40.4% |
| *(none)* | C-oracle | 62 | $0.0812 | $0.4270 | 17.7% | 30.6% |

**Ranking of the levers: downshift ≫ doing nothing ≫ compaction.** Compaction-only
(C-oracle, n=22) has both a worse mean and a worse P99 than the untouched control — it is
not a weak cost controller, it is a cost *increase*. `restrict` fired **once** in arm C
across all 349 runs.

> The `none` rows are not a causal comparison across arms: interventions fire on runs that
> are already expensive, so each arm's un-intervened subset is its cheap tail by selection.
> The ranking among the *intervened* cells is the signal.

### 5. Per true failure mode (the label each run executed against)

| mode | arm | n | mean | P99 | margin-blown | resolved |
|---|---|---|---|---|---|---|
| context_blowout | A | 42 | $0.1676 | $0.4472 | 66.7% | 16.7% |
| context_blowout | B | 42 | $0.0761 | $0.1182 | **2.4%** | 7.1% |
| context_blowout | C | 42 | $0.1072 | $0.2385 | 45.2% | 14.3% |
| context_blowout | C-oracle | 41 | $0.1761 | $0.5910 | 61.0% | 9.8% |
| tool_loop | A | 8 | $0.2885 | $0.4666 | 100.0% | 25.0% |
| tool_loop | B | 8 | $0.0883 | $0.1144 | **12.5%** | 0.0% |
| tool_loop | C | 8 | $0.1620 | $0.1995 | 87.5% | 12.5% |
| tool_loop | C-oracle | 8 | $0.1873 | $0.3552 | 75.0% | 0.0% |
| none | A | 37 | $0.0363 | $0.0600 | 0.0% | 45.9% |
| none | B | 37 | $0.0398 | $0.0949 | 0.0% | 45.9% |
| none | C | 37 | $0.0456 | $0.1463 | 5.4% | 43.2% |
| none | C-oracle | 39 | $0.0607 | $0.4617 | 5.1% | 43.6% |

Generic downshift wins *within every failure mode*, including the ones it wasn't matched
to. Note also that all four arms leave the `none` (easy-task) rows essentially alone,
which is the "do not intervene on easy tasks" hard rule holding.

### 6. Label diagnostics

`true_failure_mode` is what a run actually executed against; v2 relabeling writes
`relabeled_failure_mode` beside it rather than overwriting history.

| labeler | context_blowout | tool_loop | wrong_model | inherent_difficulty | none |
|---|---|---|---|---|---|
| **v1** (executed) | 42 | 8 | **0 — unreachable** | **0 — unreachable** | 37 |
| **v2** (current) | 25 | 8 | **17** | 0 (absent in this data) | 37 |

**17 of 87 Arm A runs change label.** So the completed C-oracle runs do not measure "the
oracle with a correct diagnosis" — that arm needs re-running before the headline can be
claimed. `inherent_difficulty` is now reachable but genuinely does not occur here: no run
reached 90% of its ceiling within 4 steps, because cost accrues gradually.

### 7. The mechanism: compaction fights the prompt cache

Cached input is **52% of Arm A's total spend** ($5.51 of $10.69). Compaction drops middle
turns, which invalidates the KV prefix, so the *surviving* prompt is re-billed at the full
input rate — 4× the cached rate. Measured at the step immediately before vs. after the
first compaction of every affected run:

| | before compact | after |
|---|---|---|
| cached tokens | 10,577 | **0** |
| uncached input tokens | 3,440 | 7,187 |
| step cost | $0.01735 | **$0.01970** |

And it fired **3.6× per affected run** (36 of 46 runs compacted more than once), so the
cache was destroyed repeatedly. Whole-run cost decomposition:

| arm | prompt tokens | uncached input $ | cached input $ | output $ | **total** |
|---|---|---|---|---|---|
| A | 9,915,389 | $2.75 | $5.51 | $2.43 | $10.69 |
| B | 7,096,349 (−28%) | $2.13 | $1.60 | $1.65 | **$5.38** |
| C | 6,493,562 (−35%) | $3.43 | $2.20 | $1.86 | $7.49 |
| C-oracle | 9,522,333 (−4%) | **$3.96** | $4.96 | $2.16 | **$11.09** |

The compaction-heavy arm is the clearest case: C-oracle carried **4% fewer** prompt tokens
than the control and paid **8% more** for them ($8.92 vs $8.26 of input spend), because
repeated invalidation shifted the mix from cached to full-price. Cutting tokens is not the
same as cutting cost once the cache is in the picture. The break-even is `R(1−m)/(Dm)` further calls — `3R/D` at a 0.25 cache multiplier — so drop
half the prompt and you need 3 more calls just to recover the re-priming cost. A small trim
taken often is strictly worse than a large trim taken once, and v1 sat in exactly that
losing quadrant.

### 8. Four validity defects found in this result — read before citing it

1. **The labeler was degenerate, so C-oracle was never a "perfect detector".** v1 split
   context_blowout from wrong_model on `max(context)/context[0] ≥ 1.5`, which 99% of runs
   pass (median total growth **17.8×**) — all 50 near-ceiling Arm A runs landed on
   context_blowout, and `wrong_model` / `inherent_difficulty` were unreachable. C-oracle
   therefore fired 97 compacts, 4 restricts and **zero downshifts**: a compaction-only arm,
   structurally denied the one lever that works. (Arm C beats it only because the *live*
   detector does emit `wrong_model`, so C accidentally downshifts 112 times.) v2 keys on
   where the money went instead — see §6 and `analysis/labeling.py`.
2. **`restrict_tool` was a disguised kill switch.** The TB2 agent had exactly one tool, so
   restricting it emptied the `tools` array, the model could emit no tool call, and the
   harness read that as *finished*. All 5 restrict firings ended the run on that exact step,
   all unresolved. It also fired **once** in arm C — an earlier version of this README
   claimed "for tool_loop, tool restriction → 87.5% blown", which was wrong: those runs were
   compacted and downshifted, not restricted. The tool surface is now partitioned into 5
   tools over the same sandbox, so restriction degrades the agent instead of killing it.
3. **The noise floor is larger than the effect** (§3). Unmeasured at the time; 3 of B's 9
   lost resolutions had **zero interventions**, i.e. a third of its apparent quality cost is
   nondeterminism, not policy.
4. **The quality criterion was unfalsifiable** (§2). The non-inferiority test needed n≥1,217
   per arm for its 3pp margin and had 87, so its best possible z was 0.44 against a 1.645
   threshold — it could not pass whatever the data said. It was also *mislabelled* as
   "INFERIOR" rather than inconclusive, turning absence of evidence into a finding. And the
   two tail criteria were raw `>` comparisons with no test at all: `c_beats_a` passes on a
   paired test but its margin-blown component is not significant (p = 0.21).

Also: **`block` never fired once in 349 runs**, because `inherent_difficulty` was never
diagnosed. One of the four interventions was dead in practice.

### 9. What this result does and does not support

**Supported, with a formal test.** Budget interventions work, and downshift is the effective
one: blown 41.4% → 2.3% (p = 4.4×10⁻¹⁰), P99 $0.452 → $0.117 (paired CI [+0.226, +0.366]),
total spend $10.69 → $5.38, cheaper on 64/87 tasks (sign p = 1.3×10⁻⁵). This is the one
finding robust to the noise floor.

**Not supported: the cost/quality trade-off.** B resolves least (23.0% vs A's 29.9%), but
the paired exact test on resolution gives **p = 0.146** — not significant, and 3 of its 9
lost resolutions had no intervention at all. The trade-off may well be real; this run
cannot show it.

**Supported.** Compaction under cached pricing is a cost *increase*, not a weak saving. This
is the load-bearing new finding and it is mechanistically explained, not just observed.

**Supported.** Overrunning the ceiling almost never buys the outcome: Arm A resolved **3 of
36** blown runs vs **23 of 51** held. Resolved runs finish by 15 steps (P90); unresolved ones
grind to the 30-step cap. A hard stop is therefore close to free, which is what arm D's top
rung does.

**Rejected in reverse — for this implementation.** Generic significantly beats matched
(p = 1.8×10⁻⁷), and C-oracle is statistically indistinguishable from doing nothing. But the
oracle arm was crippled by the labeler and one of its interventions was a kill switch, so
what has been rejected is *this* implementation of matching, not the idea. The re-run is
what settles it — and per §2 it needs **~14 seeds** to be conclusive on quality, not 3.

> Caveats: one model (gpt-5.6 at reasoning_effort≈none), **one seed**, ceilings synthesized
> from difficulty, and gpt-5.6 pricing is approximate. Directional, not the last word.

Distribution plot: `analysis_out/cost_distributions.png`.

## How it works

An agent runs a task in a loop, sending every model call through a **gateway**. The
gateway tracks cumulative cost against the task's *margin ceiling*
(`price × (1 − target_margin)`), and — depending on the experiment **arm** — may
intervene before a call:

| Arm | Policy |
|-----|--------|
| **A** control | never intervene (baseline) |
| **A-prime** placebo | identical policy to A — measures the run-to-run noise floor |
| **B** generic | downshift to a cheaper model once over budget-trend |
| **C** matched | diagnose the failure mode, apply the *matching* fix |
| **C-oracle** | same as C but using the *true* mode from post-hoc labeling |
| **D** laddered | escalate on cost ratio alone: downshift → cache-aware compaction → hard stop |

**Arm D** is not a matched arm; it is the policy the first benchmark's evidence points to,
with each rung earned by a measurement rather than by the taxonomy:

- **≥ 0.60 → downshift.** The only intervention that moved the tail (blown 41.4% → 2.3%).
- **≥ 0.85 → also compact, once.** Gated on `compaction_gain()` clearing the cache
  break-even and on the prompt not already being mostly cache hits, because repeated small
  trims were measurably cost-*increasing*.
- **≥ 1.00 (projected) → hard stop.** Overrunning the ceiling almost never bought the
  outcome: Arm A resolved **3/36 blown runs vs 23/51 held**. The stop tests the *projected*
  cost (cum + trailing-mean call cost), because a spent-only check waves a run through at
  ratio 0.99 and overshoots by a full call. This is also the first arm in which `block` can
  actually fire.

Failure mode → matched intervention:

| Mode | Detection (heuristic) | Fix | Measured |
|------|-----------------------|-----|----------|
| tool_loop | same (tool, args) ≥3× | restrict that tool | fired 1× in arm C — untestable on a 1-tool agent (now fixed) |
| context_blowout | context growth + rising cost | compact messages | **cost-increasing** under cached pricing |
| wrong_model | high cost, flat context, no loop | downshift | the one lever that works |
| inherent_difficulty | expensive from early, no waste | block / escalate | never fired in 349 runs |

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
│   E2B SANDBOX ◄── 5 shell tools ─ AGENT     PROVIDER (openai / anthropic /   │
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
| **Arm switch** | `gateway/arms.py` | The *only* per-arm fork. Maps detected/true mode → intervention; also holds arm D's ladder and the A-prime placebo. |
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
- `restrict_tool` — remove the looping tool from the `tools` array. Only meaningful if the
  agent has somewhere else to go, hence the 5-tool terminal surface (`agent/terminal_tools.py`:
  `run_command` / `read_file` / `write_file` / `list_dir` / `search_files`, all execing in the
  same sandbox). With one tool this silently *ends* the run instead of redirecting it.
- `compaction_gain` — not an intervention but the guard on one: the fraction of the prompt a
  trim would drop, so a caller can check the prompt-cache break-even (`3R/D` further calls at
  a 0.25 cache multiplier) before paying the invalidation cost.
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

**Statistics** (`analysis/metrics.py`) — no scipy. Cost comparisons are **paired on
(task, seed)**, because arms share the task set and between-task cost variance is far larger
than the between-arm effect: `paired_cost_test` bootstraps the per-task difference and also
the *P99 difference* (resampling tasks so both arms are recomputed on the same resample),
giving a CI on the quantity the hypothesis is actually about. `two_proportion_test` covers
margin-blown rate, `paired_resolution_test` is an exact McNemar-style test on discordant
pairs only, and `non_inferiority_resolution` reports whether it is even *feasible* at the
given n — a test whose best-case z (true difference zero) is below the critical value cannot
pass, and reporting that as "inferior" would be an inferential error. `seeds_needed` converts
the required n into seeds over the available task set.

**Analysis methodology** (`analysis/`) — post-hoc labeling assigns each run a
`true_failure_mode` from its whole trajectory. Labeling is **v2**: it discriminates on
`input_cost_share` (what fraction of the bill was carrying context vs generating tokens,
rate-card weighted so cached input is correctly discounted) rather than on raw context
growth, which carries no information here. Re-labeling after a sweep writes
`relabeled_failure_mode` and never overwrites `true_failure_mode`, because that field is the
label a completed run actually *executed* against — rewriting it would silently change the
meaning of finished C-oracle runs. The report also emits a **noise floor** (paired
same-policy divergence), a breakdown keyed on the **intervention actually applied**, and
**label diagnostics** (executed vs relabeled). Metrics lead with tail/variance (P99, P90,
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
analysis/      labeling (v2), metrics (+ noise floor, per-intervention), plots, report
config/        providers.yaml, rate_card.yaml, detection.yaml, tasks.yaml, experiment.yaml, env
scripts/       fetch_terminal_bench.sh (sparse-clone the ~89 TB2 tasks)
tests/         cost / adapters / detectors / interventions / arms / tb_loader / e2b_exec_env
```

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pytest tests/ -q          # 51 tests, no network
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
# --full is phased: A + A-prime (control + placebo) -> label -> B / C / C-oracle / D
./.venv/bin/python -m orchestrator.e2b_orchestrator --full --seeds 1,2,3 --max-steps 30
./.venv/bin/python -m analysis.report

# or a single arm at a time (resumable — completed runs are skipped)
./.venv/bin/python -m orchestrator.e2b_orchestrator --arms D --seeds 1,2,3 --max-steps 30
```

Per run: each Harbor `docker_image` is built once into an E2B template
(`Template.from_image(...).build(...)`, cached in `config/e2b_templates.json`); a sandbox
is created from it; the agent's 5 shell tools all exec bash inside it via the
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
- [x] 8. Validity fixes on the back of that run: v2 labeler (all four modes reachable),
      5-tool terminal surface (so `restrict` is testable), `A-prime` placebo arm, and
      analysis keyed on the intervention actually applied
- [x] 9. Arm D — laddered downshift → cache-aware compaction → predictive hard stop
- [x] 10. Real significance tests: paired bootstrap on cost/P99 differences, two-proportion
      tests on margin-blown, McNemar on resolution, and a feasibility guard that flags a
      non-inferiority test which cannot pass at the given n. Criteria are now
      significance-based rather than `>` on point estimates.
- [x] 11. Paired non-inferiority test + duration logging (`sandbox_sec`, `agent_loop_sec`,
      per-step `latency_sec`), so infrastructure cost is measurable per (task, arm) instead
      of only recoverable from a provider dashboard.
- [ ] 12. **Re-run the matrix** with A-prime and D included. The v2 labels change 17/87 runs,
      so the oracle arm's existing data no longer means what it claims.

### Cost of the re-run

Priced from the observed per-run means. Arm D never ran live, so its $0.0612/run is a
replay of the ladder over Arm A's trajectories — the same estimator predicts Arm B's real
cost to within **4%**, so the figure is trustworthy. Dropping `max_steps` 30 → 20 is nearly
free (Arm A: $10.69 → $8.57 for **1 lost resolution of 26**; C and C-oracle lose none),
since resolved runs finish by 15 steps at P90.

| design (all 87 tasks, 1 seed) | runs | model API |
|---|---|---|
| all 6 arms, `max_steps 30` | 522 | $50.54 |
| **all 6 arms, `max_steps 20`** | 522 | **$42.90** |
| drop C-oracle (A, A′, B, C, D) | 435 | $33.91 |
| core question only (A, A′, B, D) | 348 | $27.17 |
| cheapest useful (A, B, D) | 261 | $18.61 |
| *previous sweep, for reference* | 348 | $34.52 |

One seed suffices: the tail comparisons were already significant at p < 10⁻⁷ at n=87, and
the paired quality test supports a 5pp margin there. More seeds buy a tighter quality
margin (table in §2) and nothing else.

**Not included: sandbox cost.** E2B bills by the second and the previous sweep recorded no
timings, so its infrastructure spend is unrecoverable. That is now fixed — set
`e2b_usd_per_sandbox_hour` in `config/experiment.yaml` and the report breaks model vs infra
spend down per arm. This matters beyond bookkeeping: an intervention that cuts tokens but
*raises* step count can be a net loss once sandbox-seconds are priced, and no arm was
evaluated on that basis.

The sweep is also strictly sequential (`for arm / for task / for seed`), so 522 runs is
roughly 2 days unattended. Sandboxes are independent and the sweep is resumable, so
parallelising is the cheapest available speedup.

The `mock` results are for **wiring validation**, not the paper's finding: whether
matched (C) beats generic (B) is the empirical question the real-model runs answer.
The mock is deliberately not rigged to guarantee that outcome.
