# Disruption-Aware Path Repair for Lifelong AMR Routing

LLM-guided repair-strategy selection for lifelong multi-agent path finding (MAPF), with a
deterministic validator and a verified strategy cache, plus a browser 3D replay viewer.

> * Setup and commands: [`HOW_TO_RUN.md`](HOW_TO_RUN.md)
> * Viewer architecture and UI status: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and
>   [`docs/3D-viewer-redesign-change-report.pdf`](docs/3D-viewer-redesign-change-report.pdf)
> * What was broken in the pipeline, how it was fixed and tested:
>   [`INTEGRATION_REVIEW.md`](INTEGRATION_REVIEW.md)

## Quick start

```bash
uv sync                                   # or: pip install -e . (Python 3.11)
cp .env.example .env                      # add GROQ_API_KEY (or OpenAI / Ollama settings)
pytest -q                                 # 91 tests, ~2 min
python -m amr_repair.experiments.run --provider groq --require-llm --robots 100 --steps 300
```

**Always check which strategist actually ran.** Every run prints/records `llm_provider`
(`groq`, `openai`, `ollama`, `custom` or `heuristic`). `heuristic` means *no LLM was called*: the
deterministic rule-based selector made the decisions, and those decisions are counted in
`heuristic_decisions`, never in `llm_calls`. Use `--require-llm` (or `require_llm=True`) for
experiments so a missing key fails loudly instead of silently producing rule-based results.

## How the pipeline maps to the synopsis

| Stage | Module | What it does |
| --- | --- | --- |
| 1 Simulation | `sim/lifelong.py`, `sim/planning.py` | Lifelong MAPF: every robot follows a committed, time-indexed, collision-free plan from a rolling-horizon **cooperative space-time A\*** (WHCA\*, Silver 2005); new goal on arrival from a per-robot seeded task stream. |
| 2 Disruptions | `disrupt/injector.py` | Blocked cells (contiguous, optionally temporary), robot failures (permanent obstacle), robot delays (k-step stall). Isolated or overlapping (see below). |
| 3 Detection | `sentinel/detect.py` | Incident record from the committed plans: affected robots, `must_replan`, `candidate_robots`, congestion, interaction with simultaneous **and earlier still-active** disruptions, `overlap_kind`. |
| 4 Cache lookup | `cache/store.py`, `cache/negative.py` | Hard filter (types, size, congestion, isolated/interacting) + RBF similarity on incident features; strategy signatures in negative memory are skipped. |
| 5 Dry run | `loop/controller.py` | Cached strategy is solved + validated on the live state; failure = cache miss. |
| 6 Strategist | `llm/strategist.py`, `llm/prompts.py`, `llm/schema.py` | Instructor + Pydantic. The LLM sees the incident summary (never the map) and returns robots / solver / priority / horizon from a fixed menu. Robot IDs are grounded: hallucinated IDs trigger a schema re-ask. |
| 7 Path repair | `solvers/pibt.py`, `solvers/lns.py`, `solvers/windowed_pbs.py` | PIBT, MAPF-LNS, windowed PBS; all plan around every other robot's committed plan and respect delays. |
| 8 Validation | `validate/validator.py` | Single source of truth, checked against the **whole fleet**: continuity, obstacles, failed robots, delays, vertex/edge conflicts, parking safety, unrepaired affected robots, deadlock. One LLM retry with the exact violation, then the fixed fallback (windowed PBS, horizon 30), then an emergency hold. |
| 9 Execute + feedback | `loop/controller.py` | Commit validated plans; store the strategy in the cache, rejected ones in negative memory. |
| 10 Evaluation | `experiments/` | Benchmarks vs. fixed solvers, ablations, controlled overlap scenarios. |

## Overlapping disruptions

`overlap_mode` (config / `--overlap` / `--overlaps`):

* `none`: isolated disruptions only
* `same_aisle`: a second disruption of any type, 1–9 steps later, in the same area (sequential)
* `same_robot`: a second disruption 1–9 steps later involving the same robot
* `simultaneous`: a second disruption of any type in the same area at the same step
* `cascade`: blockage + failure + delay in one area within a few steps (compound)
* `mixed`: every kind in one run

Follow-up disruptions are resolved when they fire so that they interact with the first one
(affected robots or path regions overlap). Each incident is labelled
`isolated | simultaneous | sequential | compound` (`overlap_kind`), lists the earlier disruptions
it overlaps (`related_disruptions`), and both are given to the strategist. Metrics are reported
per overlap kind (`metrics.by_overlap_kind`).

Controlled scenario suite: every type alone, every pair of types both simultaneous and
sequential, and the three-type compound (16 scenarios):

```bash
python -m amr_repair.experiments.scenarios --provider groq --seeds 1 2 3      # 150 robots by default
```

## Experiments

```bash
# Full comparison: proposed framework vs Fixed PIBT/LNS/PBS, rule-based selector,
# no-cache, no-validator and threshold sweep, over all overlap modes.
python -m amr_repair.experiments.benchmark --provider groq --seeds 1 2 3
# -> runs/benchmark_summary.{csv,md,tex}, runs/benchmark_by_overlap_kind.csv

python -m amr_repair.experiments.benchmark --quick --allow-heuristic   # smoke test without an LLM
```

Every run starts with an **empty strategy cache** (`cache_persist=False`), so results do not
depend on run order. Baselines use the same validator and the same deterministic fallback, so
the comparison isolates the value of adaptive strategy selection. `RuleBased_Selector` runs the
full pipeline with the deterministic rules instead of the LLM: the key control condition.

### Metric definitions

* `repair_success_rate`: incidents whose executed plan passed validation (not an emergency hold)
* `first_attempt_success_rate`: first candidate (cache dry run, first LLM directive or baseline solver) valid
* `mean_recovery_steps`: steps until every repaired robot reached its goal or is closer to it than at detection (censored at run end, see `recovered`)
* `cost_delta_pct`: extra steps of the executed repair over each robot's single-agent shortest path on the disrupted map, as a percentage of that shortest path
* `throughput`: goals completed in the run (lifelong MAPF)
* `llm_calls` / `llm_api_requests` / `llm_tokens`: real LLM decisions / HTTP requests incl. schema re-asks / tokens summed over all requests
* `executed_vertex_collisions`, `executed_edge_swaps`: audited from the executed frames, independent of planner and validator; must be 0
* `executed_safety_violations`: invalid plans executed without validation (no-validator ablation only)

## 3D replay viewer

```bash
python scripts/make_viewer_replays.py --provider groq    # writes viewer/public/replays/<scenario>[_<mode>].json
cd viewer && npm ci && npm run dev
```

Three.js `WebGPURenderer` (falls back to WebGL2), modular scene in `viewer/src/world/`, incident
overlays in `world/incidents.ts` (see `PROJECT_CONTEXT.md`). Pick a scenario (isolated /
simultaneous / sequential / compound / mixed) and a mode (LLM or one of the three baselines; each
mode is its own run). The incident list shows every incident with its overlap kind and how it was
resolved; selecting one frames it in 3D and opens the repair-pipeline card: sentinel, cache, dry
run, strategist (provider, latency, tokens, re-asks), executed solver and validator attempts.
Replays generated without an LLM are labelled "rule-based, NO LLM". Blocked cells are shown for
their recorded lifetime (`disruptions[].active_until`). The "legacy" experiment entries are old
replays (one is synthetic) kept for reference.

## Replay JSON Schema

The browser 3D viewer reads files matching this schema (`runs/*.json` or `/public/replays/*.json`):

```json
{
  "meta": {
    "seed": 42,
    "mode": "llm", // "llm" | "baseline_pibt" | "baseline_lns" | "baseline_pbs"
    "map_name": "warehouse-10-20-10-2-1",
    "steps": 600
  },
  "map": {
    "width": 161,
    "height": 63,
    "obstacles": [[1, 2], [1, 3]]
  },
  "frames": [
    {
      "t": 0,
      "pos": [[10, 15], [20, 30]], // indexed by robot id
      "state": ["moving", "idle"]    // "moving" | "idle" | "delayed" | "failed" | "replanning"
    }
  ],
  "incidents": [
    {
      "id": "inc_017",
      "t_detected": 212,
      "disruptions": [
        {
          "type": "blocked_cell",    // "blocked_cell" | "robot_failure" | "robot_delay"
          "cells": [[52, 17]],
          "robot": 17,
          "k_steps": 5
        }
      ],
      "affected_robots": [3, 17, 41],
      "congestion": {
        "level": "medium",          // "low" | "medium" | "high"
        "region": [[48, 14], [56, 20]]
      },
      "interacts_with": ["inc_016"],
      "cache": {
        "hit": true,
        "matched": "inc_004",
        "similarity": 0.91
      },
      "dry_run": {
        "ran": true,
        "valid": false,
        "violation": "edge swap robots 3,41 t=214"
      },
      "llm": {
        "called": true,
        "latency_ms": 1430,
        "tokens": 916,
        "directive": {
          "robots": [3, 17, 41],
          "solver": "pibt",         // "pibt" | "lns" | "windowed_pbs"
          "priority": [17, 3, 41],
          "horizon": 20
        }
      },
      "validator": [
        { "attempt": 1, "valid": false, "violation": "vertex robots 3,41 cell [52,17] t=215" },
        { "attempt": 2, "valid": true }
      ],
      "fallback_used": false,
      "executed_at": 214,
      "paths": {
        "invalidated": { "3": [[50, 17], [51, 17], [52, 17]] },
        "repaired": { "3": [[50, 17], [50, 18], [51, 18], [52, 18], [53, 17]] }
      },
      "outcome": {
        "recovery_steps": 4,
        "cost_delta": 11
      }
    }
  ],
  "metrics": {
    "repair_success_rate": 0.97,
    "mean_recovery_steps": 6.2,
    "cost_delta_pct": 3.1,
    "llm_calls": 14,
    "llm_tokens": 21800,
    "cache_hits": 9
  }
}
```

Additional fields written by the current pipeline (all optional for the viewer):
`meta.llm_provider`, `meta.llm_model`, `meta.planner`; per incident `overlap_kind`,
`related_disruptions`, `must_replan`, `candidate_robots`, `failed_robots`, `decision_source`,
`executed_directive`, `executed_valid`, `llm.provider`, `llm.calls[]`, `outcome.shortest_cost`,
`outcome.recovered`, `disruptions[].active_until`; top-level `disruptions[]` (every disruption that
fired, with `t`, `active_until`, `impact`, `incident_id`, `group`, `overlap_kind`);
`metrics.by_overlap_kind` and the metrics listed above.

## Repository structure

```
amr_repair/
  sim/          grid, robots, planning primitives (reservation table, cooperative A*), lifelong world
  disrupt/      disruption injector (isolated + overlapping)
  sentinel/     incident records, detection, interaction / overlap classification
  cache/        verified strategy cache, negative memory
  llm/          directive schema (grounded), prompts, strategist (Instructor, providers)
  solvers/      PIBT, LNS, windowed PBS
  validate/     fleet-wide deterministic validator
  loop/         closed-loop controller, metrics logger, frame audit
  experiments/  benchmark, scenarios, run, baselines, ablations
  replay/       replay models and exporter
tests/          pytest suite (incl. a fake OpenAI-compatible LLM for end-to-end tests)
scripts/        viewer replay generation, 3D model builder
viewer/         Vite + TypeScript + Three.js (WebGPU) replay viewer: src/world/, src/render/, src/main.ts
docs/           viewer redesign change report
runs/legacy_invalid/   results produced before the fixes (do not cite)
```
