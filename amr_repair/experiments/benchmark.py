"""Benchmark runner: proposed framework vs fixed MAPF strategies and ablations.

Conditions
  Proposed_Framework   LLM strategist + verified cache + validator
  Fixed_PIBT/LNS/PBS   one fixed solver (same validator and same fallback)
  RuleBased_Selector   same pipeline, deterministic rules instead of the LLM
  Ablation_NoCache     LLM + validator, no cache
  Ablation_NoValidator LLM + cache, plans executed without validation
  Threshold_x.xx       cache similarity threshold sweep

Every run starts with an EMPTY strategy cache, so results do not depend on
the order in which conditions are run or on earlier sessions.

Usage
  python -m amr_repair.experiments.benchmark --provider groq --seeds 1 2 3
  python -m amr_repair.experiments.benchmark --quick --allow-heuristic   # no LLM: smoke test only
"""

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from amr_repair.loop.controller import SimulationConfig, run  # noqa: E402

COLUMNS = [
    "Condition", "Mode", "Overlap", "Seed", "Provider", "Cache", "Validator", "Threshold",
    "Incidents", "Interacting", "SuccessRatePct", "FirstAttemptSuccessPct", "MeanRecoverySteps", "CostDeltaPct",
    "Throughput", "LLMCalls", "LLMApiRequests", "LLMTokens", "HeuristicDecisions", "LLMErrors",
    "MeanLLMLatencyMs", "CacheHits", "CacheLookups", "CacheHitRatePct", "VerifiedReuses",
    "MeanRepairLatencyMs", "InterceptedViolations", "Fallbacks", "EmergencyHolds",
    "UnvalidatedExecuted", "GuardInterventions", "ExecutedCollisions",
    "IsolatedIncidents", "IsolatedSuccessPct", "OverlapIncidents", "OverlapSuccessPct", "OverlapFirstAttemptPct",
    "RunDurationSec", "Status",
]
OVERLAP_KINDS = ["isolated", "simultaneous", "sequential", "compound"]


def parse_benchmark_args(argv=None):
    p = argparse.ArgumentParser(description="Lifelong MAPF disruption-repair benchmark")
    p.add_argument("--map", default="maps/warehouse-10-20-10-2-1.map")
    p.add_argument("--robots", type=int, default=100)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--overlaps", nargs="+", default=None,
                   choices=["none", "same_aisle", "same_robot", "simultaneous", "cascade", "mixed"],
                   help="default: none same_aisle same_robot simultaneous cascade (--quick: none mixed)")
    p.add_argument("--rate", type=float, default=0.12, help="Disruption events per step (generator)")
    p.add_argument("--provider", default=None, help="groq | openai | ollama | heuristic | auto (default: $LLM_PROVIDER or auto)")
    p.add_argument("--model", default=None)
    p.add_argument("--allow-heuristic", action="store_true",
                   help="Allow LLM conditions to run without an LLM (results are then NOT LLM-guided)")
    p.add_argument("--output_dir", default="runs/benchmark")
    p.add_argument("--summary_dir", default="runs")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def build_experiment_matrix(quick: bool = False) -> List[Dict[str, Any]]:
    core = [
        {"name": "Proposed_Framework", "mode": "llm", "cache": True, "val": True, "thresh": 0.85},
        {"name": "Fixed_PIBT", "mode": "baseline_pibt", "cache": False, "val": True, "thresh": 0.85},
        {"name": "Fixed_LNS", "mode": "baseline_lns", "cache": False, "val": True, "thresh": 0.85},
        {"name": "Fixed_PBS", "mode": "baseline_pbs", "cache": False, "val": True, "thresh": 0.85},
        {"name": "RuleBased_Selector", "mode": "llm", "cache": True, "val": True, "thresh": 0.85, "provider": "heuristic"},
        {"name": "Ablation_NoCache", "mode": "llm", "cache": False, "val": True, "thresh": 0.85},
    ]
    if quick:
        return core
    return core + [
        {"name": "Ablation_NoValidator", "mode": "llm", "cache": True, "val": False, "thresh": 0.85},
        *({"name": f"Threshold_{th:.2f}", "mode": "llm", "cache": True, "val": True, "thresh": th}
          for th in (0.70, 0.80, 0.90, 0.95)),
    ]


def _row(exp, overlap, seed, replay, elapsed) -> Dict[str, Any]:
    m = replay.metrics
    bk = m.by_overlap_kind
    iso = bk.get("isolated", {})
    ov = [bk[k] for k in ("simultaneous", "sequential", "compound") if k in bk]
    n_ov = sum(g["incidents"] for g in ov)

    def wavg(key):
        return round(100 * sum(g[key] * g["incidents"] for g in ov) / n_ov, 1) if n_ov else ""

    return {
        "IsolatedIncidents": int(iso.get("incidents", 0)),
        "IsolatedSuccessPct": round(100 * iso["success_rate"], 1) if iso else "",
        "OverlapIncidents": n_ov, "OverlapSuccessPct": wavg("success_rate"),
        "OverlapFirstAttemptPct": wavg("first_attempt_success_rate"),
        "Condition": exp["name"], "Mode": exp["mode"], "Overlap": overlap, "Seed": seed,
        "Provider": m.llm_provider or "-", "Cache": "On" if exp["cache"] else "Off",
        "Validator": "On" if exp["val"] else "Off", "Threshold": exp["thresh"],
        "Incidents": m.incidents_total, "Interacting": m.incidents_interacting,
        "SuccessRatePct": round(m.repair_success_rate * 100, 1),
        "FirstAttemptSuccessPct": round(m.first_attempt_success_rate * 100, 1),
        "MeanRecoverySteps": m.mean_recovery_steps, "CostDeltaPct": m.cost_delta_pct,
        "Throughput": m.throughput, "LLMCalls": m.llm_calls, "LLMApiRequests": m.llm_api_requests,
        "LLMTokens": m.llm_tokens, "HeuristicDecisions": m.heuristic_decisions, "LLMErrors": m.llm_errors,
        "MeanLLMLatencyMs": m.mean_llm_latency_ms, "CacheHits": m.cache_hits, "CacheLookups": m.cache_lookups,
        "CacheHitRatePct": m.cache_hit_rate_pct, "VerifiedReuses": m.verified_cache_reuse_count,
        "MeanRepairLatencyMs": m.mean_total_repair_latency_ms,
        "InterceptedViolations": m.safety_violations_intercepted, "Fallbacks": m.fallback_invocations,
        "EmergencyHolds": m.emergency_holds, "UnvalidatedExecuted": m.executed_safety_violations,
        "GuardInterventions": m.guard_interventions,
        "ExecutedCollisions": m.executed_vertex_collisions + m.executed_edge_swaps,
        "RunDurationSec": round(elapsed, 2), "Status": "COMPLETED",
    }


def run_benchmarks(args) -> List[Dict[str, Any]]:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)
    matrix = build_experiment_matrix(quick=args.quick)
    if args.overlaps is None:
        args.overlaps = ["none", "mixed"] if args.quick else ["none", "same_aisle", "same_robot", "simultaneous", "cascade"]
    rows: List[Dict[str, Any]] = []
    kind_rows: List[Dict[str, Any]] = []
    total = len(matrix) * len(args.seeds) * len(args.overlaps)
    print(f"=== Benchmark: {len(matrix)} conditions x {len(args.seeds)} seeds x {len(args.overlaps)} overlaps "
          f"| {args.robots} robots, {args.steps} steps ===")
    i = 0
    for exp in matrix:
        for overlap in args.overlaps:
            for seed in args.seeds:
                i += 1
                exp_id = f"{exp['name']}_s{seed}_{overlap}"
                config = SimulationConfig(
                    map_path=args.map, num_robots=args.robots, steps=args.steps, seed=seed, mode=exp["mode"],
                    disruption_rate=args.rate, overlap_mode=overlap, cache_enabled=exp["cache"],
                    validator_enabled=exp["val"], cache_threshold=exp["thresh"], run_id=exp_id,
                    output_dir=args.output_dir, cache_persist=False,
                    llm_provider=exp.get("provider", args.provider), llm_model=args.model,
                    require_llm=(exp["mode"] == "llm" and exp.get("provider") != "heuristic"
                                 and not args.allow_heuristic),
                )
                t0 = time.perf_counter()
                try:
                    replay = run(config)
                    row = _row(exp, overlap, seed, replay, time.perf_counter() - t0)
                    for kind, g in replay.metrics.by_overlap_kind.items():
                        kind_rows.append({"Condition": exp["name"], "OverlapMode": overlap, "Seed": seed,
                                          "OverlapKind": kind, **g})
                    print(f"[{i}/{total}] {exp_id}: provider={row['Provider']} incidents={row['Incidents']} "
                          f"success={row['SuccessRatePct']}% first={row['FirstAttemptSuccessPct']}% "
                          f"llm_calls={row['LLMCalls']} cache_hits={row['CacheHits']} "
                          f"collisions={row['ExecutedCollisions']}")
                except Exception as ex:  # keep the sweep going, but make failure visible
                    print(f"[{i}/{total}] ERROR in {exp_id}: {ex}")
                    row = {c: "" for c in COLUMNS}
                    row.update({"Condition": exp["name"], "Mode": exp["mode"], "Overlap": overlap,
                                "Seed": seed, "Status": f"FAILED: {ex}"})
                rows.append(row)

    csv_path = os.path.join(args.summary_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[Artifact] {csv_path}")
    kind_path = os.path.join(args.summary_dir, "benchmark_by_overlap_kind.csv")
    if kind_rows:
        with open(kind_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(kind_rows[0]))
            w.writeheader()
            w.writerows(kind_rows)
        print(f"[Artifact] {kind_path}")
    _write_tables(rows, args.summary_dir, kind_rows)
    return rows


def _write_tables(rows: List[Dict[str, Any]], out_dir: str, kind_rows: List[Dict[str, Any]] = ()) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r["Status"] == "COMPLETED":
            groups.setdefault(f"{r['Condition']}|{r['Overlap']}", []).append(r)
    keys = ["Incidents", "SuccessRatePct", "FirstAttemptSuccessPct", "MeanRecoverySteps", "CostDeltaPct",
            "Throughput", "LLMCalls", "LLMTokens", "CacheHitRatePct", "MeanRepairLatencyMs",
            "InterceptedViolations", "ExecutedCollisions"]
    agg = []
    for k, items in groups.items():
        cond, ov = k.split("|")
        a = {"Condition": cond, "Overlap": ov, "Provider": items[0]["Provider"], "n": len(items)}
        for key in keys:
            a[key] = round(sum(float(it[key]) for it in items) / len(items), 2)
        agg.append(a)
    md = os.path.join(out_dir, "benchmark_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Lifelong MAPF disruption repair - benchmark (mean over seeds)\n\n")
        f.write("| Condition | Overlap | Provider | Seeds | Incidents | Success % | 1st-attempt % | Recovery | "
                "Cost Δ % | Throughput | LLM calls | LLM tokens | Cache hit % | Repair ms | Violations caught | "
                "Executed collisions |\n")
        f.write("|" + " --- |" * 16 + "\n")
        for a in agg:
            f.write(f"| {a['Condition']} | {a['Overlap']} | {a['Provider']} | {a['n']} | {a['Incidents']} | "
                    f"{a['SuccessRatePct']} | {a['FirstAttemptSuccessPct']} | {a['MeanRecoverySteps']} | "
                    f"{a['CostDeltaPct']} | {a['Throughput']} | {a['LLMCalls']} | {a['LLMTokens']} | "
                    f"{a['CacheHitRatePct']} | {a['MeanRepairLatencyMs']} | {a['InterceptedViolations']} | "
                    f"{a['ExecutedCollisions']} |\n")
        # isolated vs overlapping disruptions (synopsis: "single versus overlapping")
        f.write("\n## Single vs. overlapping disruptions (pooled over seeds and overlap modes)\n\n")
        f.write("| Condition | Overlap kind | Incidents | Success % | 1st-attempt % | Fallback % | Recovery | Cost Δ % |\n")
        f.write("|" + " --- |" * 8 + "\n")
        pooled: Dict[tuple, List[Dict[str, Any]]] = {}
        for kr in kind_rows:
            pooled.setdefault((kr["Condition"], kr["OverlapKind"]), []).append(kr)
        for (cond, kind), items in sorted(pooled.items(), key=lambda x: (x[0][0], OVERLAP_KINDS.index(x[0][1]))):
            n = sum(i["incidents"] for i in items)

            def w(key):
                return round(sum(i[key] * i["incidents"] for i in items) / n, 3) if n else 0

            f.write(f"| {cond} | {kind} | {int(n)} | {round(100 * w('success_rate'), 1)} | "
                    f"{round(100 * w('first_attempt_success_rate'), 1)} | {round(100 * w('fallback_rate'), 1)} | "
                    f"{round(w('mean_recovery_steps'), 2)} | {round(w('cost_delta_pct'), 2)} |\n")
    tex = os.path.join(out_dir, "benchmark_summary.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n\\centering\\small\n"
                "\\caption{Disruption repair in lifelong MAPF: adaptive (LLM) selection vs. fixed strategies "
                "and ablations (mean over seeds).}\n\\label{tab:mapf_benchmark_results}\n"
                "\\begin{tabular}{llcccccccc}\n\\hline\n"
                "Condition & Overlap & Success (\\%) & 1st-attempt (\\%) & Recovery & Cost $\\Delta$ (\\%) & "
                "Throughput & LLM calls & Hit rate (\\%) & Latency (ms) \\\\\n\\hline\n")
        for a in agg:
            f.write(f"{a['Condition'].replace('_', chr(92) + '_')} & {a['Overlap'].replace('_', chr(92) + '_')} & "
                    f"{a['SuccessRatePct']} & {a['FirstAttemptSuccessPct']} & {a['MeanRecoverySteps']} & "
                    f"{a['CostDeltaPct']} & {a['Throughput']} & {a['LLMCalls']} & {a['CacheHitRatePct']} & "
                    f"{a['MeanRepairLatencyMs']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")
    print(f"[Artifact] {md}\n[Artifact] {tex}")


if __name__ == "__main__":
    run_benchmarks(parse_benchmark_args())
