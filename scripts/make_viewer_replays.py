"""Generate the 3D viewer's replays from the REAL pipeline (not synthetic data).

For every scenario below it runs the closed loop once per mode and writes
    viewer/public/replays/<scenario>.json                 (mode = llm)
    viewer/public/replays/<scenario>_baseline_pibt.json
    viewer/public/replays/<scenario>_baseline_lns.json
    viewer/public/replays/<scenario>_baseline_pbs.json
All modes share the seed, so robots start in the same cells, get the same task
streams and face the same disruption events.

    python scripts/make_viewer_replays.py --provider groq            # LLM-guided
    python scripts/make_viewer_replays.py --allow-heuristic          # no LLM: replays are labelled rule-based
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from amr_repair.loop.controller import SimulationConfig, run  # noqa: E402

SCENARIOS = {
    "isolated": "none",
    "overlap_mixed": "mixed",
    "overlap_simultaneous": "simultaneous",
    "overlap_sequential": "same_aisle",
    "overlap_compound": "cascade",
}
MODES = ["llm", "baseline_pibt", "baseline_lns", "baseline_pbs"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="viewer/public/replays")
    p.add_argument("--robots", type=int, default=120)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--rate", type=float, default=0.12)
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--allow-heuristic", action="store_true")
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    args = p.parse_args(argv)

    for key in args.scenarios:
        for mode in MODES:
            name = key if mode == "llm" else f"{key}_{mode}"
            rep = run(SimulationConfig(
                map_path="maps/warehouse-10-20-10-2-1.map", num_robots=args.robots, steps=args.steps,
                seed=args.seed, mode=mode, disruption_rate=args.rate, overlap_mode=SCENARIOS[key],
                run_id=name, output_dir=args.out, llm_provider=args.provider, llm_model=args.model,
                require_llm=(mode == "llm" and not args.allow_heuristic),
            ))
            m = rep.metrics
            size = os.path.getsize(os.path.join(args.out, f"{name}.json")) / 1e6
            print(f"{name:40s} provider={m.llm_provider or '-':9s} incidents={m.incidents_total:2d} "
                  f"overlapping={m.incidents_interacting:2d} success={m.repair_success_rate:.2f} "
                  f"collisions={m.executed_vertex_collisions + m.executed_edge_swaps} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
