"""Scale sweep on one map: fleet size x disruption rate x seeds (Step 2), with the
cache trained on separate seeds and tested on held-out ones (Step 3).

For every (robots, rate) cell:

  reference  same seeds, no disruptions          -> throughput without disruptions
  train      training seeds, cache on            -> strategy-cache files
  cold       test seeds, empty cache             -> counterfactual arms + cache probes
  warm       test seeds, cache filled from train -> counterfactual arms + cache probes

Each test run forks the world at every incident and rolls PIBT, LNS, windowed
PBS and the executed decision forward ``--window`` steps
(loop/counterfactual.py), so random choice, the single best solver and the
per-incident oracle are measured on the same incidents.

The strategist defaults to the rule-based selector (free, deterministic). An
LLM sweep uses ``--provider gemini``; its answers are recorded per run under
``<out>/llm/`` so the sweep can be repeated with ``--llm-mode replay`` at no cost.

Every finished run is written to ``<out>/runs/<name>.json`` (no frames) and is
skipped when the sweep is restarted, so an interrupted sweep resumes.
Afterwards ``analyze_sweep.py`` writes the summary tables.

    python -m amr_repair.experiments.sweep --workers 8
    python -m amr_repair.experiments.sweep --robots 100 --rates 0.1 --seeds 1 2 --train-seeds 101 --steps 100  # smoke test
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from amr_repair.loop.controller import SimulationConfig, run  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="maps/warehouse-10-20-10-2-1.map")
    p.add_argument("--robots", nargs="+", type=int, default=[100, 200, 400])
    p.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3],
                   help="disruption events per step")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)), help="test seeds")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[101, 102, 103],
                   help="seeds that fill the cache for the warm condition (never tested on)")
    p.add_argument("--overlap", default="mixed",
                   choices=["none", "same_aisle", "same_robot", "simultaneous", "cascade", "mixed"])
    p.add_argument("--window", type=int, default=50, help="counterfactual steps per arm")
    p.add_argument("--threshold", type=float, default=0.85, help="cache similarity threshold")
    p.add_argument("--provider", default="heuristic", help="heuristic (rule-based, default) | gemini | ...")
    p.add_argument("--model", default=None)
    p.add_argument("--llm-mode", default="record", choices=["record", "replay"])
    p.add_argument("--step-seconds", type=float, default=1.0,
                   help="simulated seconds per step for charging LLM thinking time (0 = do not charge)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", default="runs/sweep")
    return p.parse_args(argv)


# ----------------------------------------------------------------- jobs

def run_name(kind: str, robots: int, rate: float, seed: int) -> str:
    return f"{kind}_r{robots}_rate{rate:g}_s{seed}"


def cache_file(out: str, robots: int, rate: float, seed: Optional[int] = None) -> str:
    tail = f"train_s{seed}" if seed is not None else "warm"
    return os.path.join(out, "cache", f"r{robots}_rate{rate:g}_{tail}.json")


def plan_jobs(args) -> Dict[str, List[Dict[str, Any]]]:
    """All runs of the sweep, in two phases (the warm runs need the training caches)."""
    common = dict(map_path=args.map, steps=args.steps, overlap=args.overlap, provider=args.provider,
                  model=args.model, llm_mode=args.llm_mode, step_seconds=args.step_seconds,
                  threshold=args.threshold, out=args.out)
    first, second = [], []
    for robots in args.robots:
        for seed in args.seeds:
            first.append({**common, "kind": "reference", "robots": robots, "rate": 0.0, "seed": seed})
        for rate in args.rates:
            for seed in args.train_seeds:
                first.append({**common, "kind": "train", "robots": robots, "rate": rate, "seed": seed})
            for seed in args.seeds:
                for kind in ("cold", "warm"):
                    (first if kind == "cold" else second).append(
                        {**common, "kind": kind, "robots": robots, "rate": rate, "seed": seed,
                         "window": args.window})
    return {"first": first, "second": second}


def config_for(job: Dict[str, Any]) -> SimulationConfig:
    kind, out = job["kind"], job["out"]
    name = run_name(kind, job["robots"], job["rate"], job["seed"])
    is_llm = job["provider"] not in (None, "heuristic")
    base = dict(map_path=job["map_path"], num_robots=job["robots"], steps=job["steps"], seed=job["seed"],
                disruption_rate=job["rate"], overlap_mode=job["overlap"], run_id=name, export=False,
                cache_persist=False, cache_threshold=job["threshold"],
                decision_step_seconds=job["step_seconds"] or None)
    if kind == "reference":  # no disruptions: the solver never runs, any mode gives the same run
        return SimulationConfig(**base, mode="baseline_pibt", cache_enabled=False)
    llm = dict(mode="llm", llm_provider=job["provider"], llm_model=job["model"], require_llm=is_llm,
               llm_response_store=os.path.join(out, "llm", f"{name}.jsonl") if is_llm else None,
               llm_response_mode=job["llm_mode"])
    if kind == "train":
        return SimulationConfig(**base, **llm, cache_enabled=True,
                                cache_export_path=cache_file(out, job["robots"], job["rate"], job["seed"]))
    return SimulationConfig(**base, **llm, cache_enabled=True, cache_probe=True,
                            counterfactual_window=job["window"],
                            cache_warm_path=cache_file(out, job["robots"], job["rate"]) if kind == "warm" else None)


def compact(replay, job: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    """What the analysis needs from a run: settings, run metrics and per-incident records (no frames)."""
    incidents = []
    for inc in replay.incidents:
        llm_calls = [c for c in (inc.llm.calls if inc.llm else [])
                     if c.error is None and not c.provider.startswith("heuristic")]
        incidents.append({
            "id": inc.id, "t": inc.t_detected, "overlap_kind": inc.overlap_kind,
            "types": sorted({d.type for d in inc.disruptions}),
            "robots": len(inc.affected_robots) - len(inc.failed_robots),
            "congestion": inc.congestion.level, "density": inc.congestion.density,
            "decision_source": inc.decision_source, "fallback": inc.fallback_used,
            "solver": inc.executed_directive.solver if inc.executed_directive else None,
            "cache": inc.cache.model_dump() if inc.cache else None,
            "dry_run": inc.dry_run.model_dump() if inc.dry_run else None,
            "llm_calls": len(llm_calls),
            "llm_tokens": sum(c.tokens for c in llm_calls),
            "llm_latency_ms": round(sum(c.latency_ms for c in llm_calls), 1),
            "decision_delay_steps": inc.decision_delay_steps,
            "recovery_steps": inc.outcome.recovery_steps, "recovered": inc.outcome.recovered,
            "counterfactual": inc.counterfactual,
        })
    return {"job": {k: v for k, v in job.items() if k != "out"}, "meta": replay.meta.model_dump(),
            "metrics": replay.metrics.model_dump(), "incidents": incidents, "seconds": round(seconds, 1)}


def run_job(job: Dict[str, Any]) -> str:
    name = run_name(job["kind"], job["robots"], job["rate"], job["seed"])
    path = os.path.join(job["out"], "runs", f"{name}.json")
    if os.path.exists(path):
        return f"{name}: already done"
    t0 = time.perf_counter()
    replay = run(config_for(job))
    data = compact(replay, job, time.perf_counter() - t0)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)  # a run file exists only when the run finished
    m = replay.metrics
    return (f"{name}: {m.incidents_total} incidents, {m.throughput_per_step:.3f} goals/step, "
            f"{data['seconds']} s")


def merge_training_caches(out: str, robots: int, rate: float, seeds: List[int]) -> int:
    """Concatenate the training runs' caches into the warm-start file for this cell."""
    entries: List[dict] = []
    for seed in seeds:
        with open(cache_file(out, robots, rate, seed), encoding="utf-8") as f:
            entries.extend(json.load(f))
    path = cache_file(out, robots, rate)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    return len(entries)


def run_phase(jobs: List[Dict[str, Any]], workers: int) -> None:
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            try:
                print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)
            except Exception as e:  # keep going; the analysis reports missing runs
                print(f"[{i}/{len(jobs)}] FAILED {run_name(j['kind'], j['robots'], j['rate'], j['seed'])}: "
                      f"{type(e).__name__}: {e}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    for sub in ("runs", "cache", "llm"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    with open(os.path.join(args.out, "sweep_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)
    jobs = plan_jobs(args)
    print(f"Sweep: robots {args.robots} x rates {args.rates} x {len(args.seeds)} test seeds "
          f"(+{len(args.train_seeds)} training seeds), {args.steps} steps, window {args.window}, "
          f"{args.workers} workers -> {args.out}", flush=True)
    t0 = time.perf_counter()
    run_phase(jobs["first"], args.workers)
    for robots in args.robots:
        for rate in args.rates:
            n = merge_training_caches(args.out, robots, rate, args.train_seeds)
            print(f"warm cache r{robots} rate{rate:g}: {n} entries from seeds {args.train_seeds}", flush=True)
    run_phase(jobs["second"], args.workers)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min. Next: "
          f"python -m amr_repair.experiments.analyze_sweep {args.out}", flush=True)


if __name__ == "__main__":
    main()
