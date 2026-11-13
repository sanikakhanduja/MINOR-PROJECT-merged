"""Controlled disruption scenarios: every single type and every overlapping combination.

Synopsis, methodology step 2: "Inject three disruption types ... Each is tested
alone and in controlled overlapping combinations." This module builds those
scenarios explicitly instead of relying on random generation:

* isolated      blocked_cell | robot_failure | robot_delay alone
* simultaneous  every pair of types (6 pairs incl. same-type pairs) at the SAME step,
                in the same area
* sequential    every pair of types, the second 3 steps later while the first is
                still active / being repaired
* compound      blockage + failure + delay in one area within 4 steps

Follow-up disruptions are resolved so that they INTERACT with the first one in
the synopsis' sense (affected robots or path regions overlap). The summary
reports, per scenario, in how many repetitions the overlap was detected; a
miss means the follow-up genuinely affected no robot (more likely in sparse
fleets, hence the default of 150 robots).

Each scenario repeats its pattern several times per run (every 35 steps) in a
busy part of the warehouse (the first disruption targets the robot/aisle used
by most other plans; follow-ups are anchored to it), so every repetition
genuinely overlaps.

Usage
  python -m amr_repair.experiments.scenarios                       # LLM (provider from env), all scenarios
  python -m amr_repair.experiments.scenarios --modes llm baseline_pbs --provider groq
  python -m amr_repair.experiments.scenarios --allow-heuristic     # no LLM available: smoke test
"""

import argparse
import csv
import itertools
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from amr_repair.disrupt.injector import DisruptionItem  # noqa: E402
from amr_repair.loop.controller import SimulationConfig, run  # noqa: E402

TYPES = ["blocked_cell", "robot_failure", "robot_delay"]
SHORT = {"blocked_cell": "block", "robot_failure": "failure", "robot_delay": "delay"}


def _item(dtype: str, t: int, **kw) -> DisruptionItem:
    d = DisruptionItem(type=dtype, t=t, **kw)  # type: ignore[arg-type]
    if dtype == "blocked_cell":
        d.block_size = d.block_size if d.block_size > 1 else 2
        d.ahead = d.ahead or 4
        d.duration = d.duration or 40
    elif dtype == "robot_delay":
        d.k_steps = d.k_steps or 6
    return d


def build_scenarios() -> Dict[str, dict]:
    """name -> {"kind", "types", "pattern": [(dtype, dt, anchored)]}"""
    scen: Dict[str, dict] = {}
    for t in TYPES:
        scen[f"isolated_{SHORT[t]}"] = {"kind": "isolated", "types": [t], "pattern": [(t, 0, False)]}
    for a, b in itertools.combinations_with_replacement(TYPES, 2):
        name = f"{SHORT[a]}+{SHORT[b]}"
        scen[f"simultaneous_{name}"] = {"kind": "simultaneous", "types": [a, b],
                                        "pattern": [(a, 0, False), (b, 0, True)]}
        scen[f"sequential_{name}"] = {"kind": "sequential", "types": [a, b],
                                      "pattern": [(a, 0, False), (b, 3, True)]}
    scen["compound_block+failure+delay"] = {
        "kind": "compound", "types": TYPES,
        "pattern": [("blocked_cell", 0, False), ("robot_failure", 2, True), ("robot_delay", 4, True)]}
    return scen


def build_schedule(pattern, steps: int, start: int = 15, every: int = 35) -> List[DisruptionItem]:
    items: List[DisruptionItem] = []
    group = 0
    for t0 in range(start, steps - 15, every):
        group += 1
        first_idx = len(items)
        for dtype, dt, anchored in pattern:
            kw = {"group": group}
            if anchored:
                kw["anchor"] = first_idx
            else:
                kw["select"] = "busy"
            items.append(_item(dtype, t0 + dt, **kw))
    return items


def run_scenario(name: str, spec: dict, mode: str, seed: int, robots: int, steps: int,
                 provider: Optional[str], model: Optional[str], require_llm: bool, out_dir: str) -> dict:
    cfg = SimulationConfig(
        map_path="maps/warehouse-10-20-10-2-1.map", num_robots=robots, steps=steps, seed=seed, mode=mode,
        schedule=build_schedule(spec["pattern"], steps), run_id=f"scenario_{name}_{mode}_s{seed}",
        output_dir=out_dir, llm_provider=provider, llm_model=model, require_llm=require_llm and mode == "llm",
    )
    t0 = time.perf_counter()
    rep = run(cfg)
    m = rep.metrics
    groups = len(build_schedule(spec["pattern"], steps)) // len(spec["pattern"])
    kinds = Counter(i.overlap_kind for i in rep.incidents)
    # Per repetition: was the overlap detected (an overlapping incident inside its
    # time window)? For isolated scenarios: were all its incidents isolated?
    span = max(dt for _, dt, _ in spec["pattern"])
    starts = sorted({d.t for d in build_schedule(spec["pattern"], steps) if d.anchor is None})
    detected = 0
    for t0 in starts:
        inside = [i for i in rep.incidents if t0 <= i.t_detected <= t0 + span]
        if spec["kind"] == "isolated":
            detected += bool(inside) and all(i.overlap_kind == "isolated" for i in inside)
        else:
            detected += any(i.overlap_kind != "isolated" for i in inside)
    multi = sum(1 for i in rep.incidents if len(i.disruptions) + len(i.related_disruptions) >= len(spec["types"]))
    return {
        "Scenario": name, "Kind": spec["kind"], "Types": "+".join(SHORT[t] for t in spec["types"]),
        "Mode": mode, "Seed": seed, "Provider": m.llm_provider or "-", "Repetitions": groups,
        "Incidents": m.incidents_total, "DetectedKinds": dict(kinds),
        "DisruptionsWithoutImpact": m.disruptions_without_impact,
        "RepetitionsDetected": detected, "AllTypesInOneIncident": multi,
        "SuccessRatePct": round(100 * m.repair_success_rate, 1),
        "FirstAttemptSuccessPct": round(100 * m.first_attempt_success_rate, 1),
        "Fallbacks": m.fallback_invocations, "EmergencyHolds": m.emergency_holds,
        "MeanRecoverySteps": m.mean_recovery_steps, "CostDeltaPct": m.cost_delta_pct,
        "LLMCalls": m.llm_calls, "CacheHits": m.cache_hits,
        "ExecutedCollisions": m.executed_vertex_collisions + m.executed_edge_swaps,
        "Sec": round(time.perf_counter() - t0, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Controlled single / overlapping disruption scenarios")
    p.add_argument("--modes", nargs="+", default=["llm"],
                   choices=["llm", "baseline_pibt", "baseline_lns", "baseline_pbs"])
    p.add_argument("--scenarios", nargs="*", default=None, help="subset of scenario names")
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--robots", type=int, default=150,
                   help="fleet size; overlaps only matter if robots share aisles (100 robots is sparse)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--allow-heuristic", action="store_true")
    p.add_argument("--output_dir", default="runs/scenarios")
    args = p.parse_args(argv)

    scen = build_scenarios()
    names = args.scenarios or list(scen)
    rows = []
    for name in names:
        for mode in args.modes:
            for seed in args.seeds:
                row = run_scenario(name, scen[name], mode, seed, args.robots, args.steps, args.provider,
                                   args.model, not args.allow_heuristic, args.output_dir)
                rows.append(row)
                print(f"{name:30s} {mode:6s} s{seed} provider={row['Provider']:9s} "
                      f"overlap detected {row['RepetitionsDetected']}/{row['Repetitions']} "
                      f"incidents={row['Incidents']:2d} kinds={row['DetectedKinds']} "
                      f"success={row['SuccessRatePct']}% first={row['FirstAttemptSuccessPct']}% "
                      f"collisions={row['ExecutedCollisions']}")
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "scenario_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[Artifact] {path}")
    return rows


if __name__ == "__main__":
    main()
