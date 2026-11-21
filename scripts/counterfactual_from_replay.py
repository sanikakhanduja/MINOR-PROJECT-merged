"""Add counterfactual per-incident analysis to a run that was recorded without it.

No LLM is called. The run is re-simulated deterministically with every incident
executing exactly the directive it executed originally (``recorded_decisions``),
and at each incident the world is forked and every fixed solver plus the
executed decision is rolled forward (``amr_repair/loop/counterfactual.py``).

The result is written only if the re-simulation is *faithful*: the same
incidents at the same steps, and every robot in exactly the same cell in every
frame. Otherwise the script stops and writes nothing.

Replays written by the current pipeline store every generator setting in
``meta``. Older replays lack the disruption rate and overlap mode, so pass them
explicitly (any value given on the command line must agree with ``meta``):

    python scripts/counterfactual_from_replay.py viewer/public/replays/overlap_mixed.json \
        --robots 120 --steps 200 --seed 3 --rate 0.12 --overlap mixed --in-place
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from amr_repair.loop.controller import SimulationConfig, run  # noqa: E402


def resolve_settings(meta: dict, given: dict) -> dict:
    """Run settings from ``meta``, with command-line values filling gaps. A given value
    that contradicts ``meta`` is an error, as is a setting found in neither."""
    from_meta = dict(meta)
    if meta.get("steps") is not None:
        from_meta["steps"] = int(meta["steps"]) - 1  # meta.steps counts frames, including t=0
    out = {}
    for key, value in given.items():
        recorded = from_meta.get(key)
        if value is not None and recorded is not None and value != recorded:
            sys.exit(f"{key}={value} contradicts the replay's recorded {key}={recorded}")
        out[key] = value if value is not None else recorded
    if out["max_block_size"] is None:
        out["max_block_size"] = 3  # SimulationConfig default; older replays never changed it
    missing = [k for k, v in out.items() if v is None]
    if missing:
        sys.exit(f"not recorded in this replay, pass them explicitly: {', '.join(missing)}")
    return out


def recorded_decisions(replay: dict) -> dict:
    """Each incident's executed directive (None = emergency hold), with the steps its robots
    waited for the decision when the run charged thinking time."""
    out = {}
    for inc in replay["incidents"]:
        d = inc.get("executed_directive")
        out[inc["id"]] = None if d is None else {**d, "decision_delay_steps": inc.get("decision_delay_steps", 0)}
    return out


def check_faithful(original: dict, resim) -> list:
    """Differences between the recorded run and its re-simulation (empty = faithful)."""
    problems = []
    a = [(i["id"], i["t_detected"]) for i in original["incidents"]]
    b = [(i.id, i.t_detected) for i in resim.incidents]
    if a != b:
        problems.append(f"incidents differ: recorded {a} vs re-simulated {b}")
    fa, fb = original["frames"], resim.frames
    if len(fa) != len(fb):
        problems.append(f"frame count {len(fa)} vs {len(fb)}")
    for x, y in zip(fa, fb):
        if [tuple(p) for p in x["pos"]] != [tuple(p) for p in y.pos]:
            problems.append(f"robot positions first differ at t={x['t']}")
            break
    return problems


def merge(original: dict, resim, settings: dict) -> dict:
    out = dict(original)
    # record the settings that reproduced the run, so the next re-simulation needs no flags
    out["meta"] = {**original["meta"], **{k: v for k, v in settings.items() if k in
                                          ("disruption_rate", "overlap_mode", "max_block_size")},
                   "counterfactual_window": resim.meta.counterfactual_window}
    by_id = {i.id: i.counterfactual for i in resim.incidents}
    out["incidents"] = [{**inc, "counterfactual": by_id.get(inc["id"])} for inc in original["incidents"]]
    summary = dict(resim.metrics.counterfactual or {})
    summary["method"] = "re-simulated from recorded decisions; frames verified identical"
    out["metrics"] = {**original["metrics"], "counterfactual": summary}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replay")
    p.add_argument("--map", default=None, help="default: maps/<meta.map_name>.map")
    p.add_argument("--robots", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--overlap", default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--in-place", action="store_true")
    args = p.parse_args(argv)

    with open(args.replay, encoding="utf-8") as f:
        original = json.load(f)
    meta = original["meta"]
    settings = resolve_settings(meta, {
        "seed": args.seed, "num_robots": args.robots, "steps": args.steps,
        "disruption_rate": args.rate, "overlap_mode": args.overlap, "max_block_size": args.block_size,
    })
    if meta.get("mode") != "llm":
        sys.exit("only llm-mode runs are re-simulated (baseline runs are the fixed arms already)")

    t0 = time.perf_counter()
    resim = run(SimulationConfig(
        map_path=args.map or os.path.join("maps", f"{meta['map_name']}.map"),
        num_robots=settings["num_robots"], steps=settings["steps"], seed=settings["seed"], mode="llm",
        disruption_rate=settings["disruption_rate"], overlap_mode=settings["overlap_mode"],
        max_block_size=settings["max_block_size"],
        cache_enabled=False, llm_provider="heuristic", export=False,
        recorded_decisions=recorded_decisions(original), counterfactual_window=args.window,
        # replays older than the decision delay never charged thinking time
        decision_step_seconds=meta.get("decision_step_seconds"),
    ))
    problems = check_faithful(original, resim)
    if problems:
        sys.exit("Re-simulation is NOT faithful, nothing written:\n  " + "\n  ".join(problems))

    out = merge(original, resim, settings)
    path = args.replay if args.in_place else (args.out or args.replay.replace(".json", "_counterfactual.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    s = out["metrics"]["counterfactual"]
    print(f"faithful re-simulation, {s['incidents']} incidents, window {s['window']} "
          f"({time.perf_counter() - t0:.1f} s) -> {path}")
    for name, a in s["arms"].items():
        print(f"  {name:13s} goals/incident {a['mean_goals']:7.2f}  recovery {a['mean_recovery_steps']:6.2f}  "
              f"first-valid {a['first_valid_rate']:.2f}  holds {a['hold_rate']:.2f}")
    print(f"  random {s['random_mean_goals']}  VBS {s['vbs_mean_goals']}  SBS={s['sbs']} {s['sbs_mean_goals']}  "
          f"matches oracle {s['executed_matches_oracle_rate']}  decision-relevant {s['distinct_rate']}  "
          f"gap closed vs random {s['gap_closed_vs_random']} (progress {s['gap_closed_vs_random_progress']})")


if __name__ == "__main__":
    main()
