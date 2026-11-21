"""Summary tables for a sweep written by ``sweep.py``.

Per cell (robots, disruption rate) and condition (cold / warm cache), over the
test seeds:

* throughput (goals per step) and throughput lost against the same seed run
  without disruptions
* repair success, fallback and emergency-hold rates
* the same-incident comparison: fleet goals and fleet progress per incident for
  PIBT, LNS, windowed PBS, the executed decision, random choice (exact
  expectation), the single best solver (SBS, chosen on the pooled incidents of
  the cell) and the per-incident oracle (VBS); share of incidents where the
  solvers differ, how often the executed decision matches the oracle, and the
  share of the oracle gap it closes
* a paired Wilcoxon signed-rank test over seeds: executed vs SBS and executed vs
  random, on fleet progress per incident
* the cache: hit rate, precision (hits whose dry run passed / matches), LLM calls
  and tokens avoided, and the threshold curve from the cache probes

Uncertainty is a 95% percentile bootstrap over seeds (10 000 resamples, fixed
seed). Statistics use numpy, plus scipy for the Wilcoxon test.

    python -m amr_repair.experiments.analyze_sweep runs/sweep
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from amr_repair.loop.counterfactual import FIXED_ARMS  # noqa: E402

BOOTSTRAP = 10_000
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
CONDITIONS = ("cold", "warm")


# --------------------------------------------------------------- statistics

def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, stat=np.mean
                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(estimate, low, high) of ``stat`` with a 95% percentile bootstrap. None when empty."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return None, None, None
    est = float(stat(x))
    if x.size == 1:
        return est, est, est
    idx = rng.integers(0, x.size, size=(BOOTSTRAP, x.size))
    boots = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(axis=1)
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ratio_ci(num: Sequence[float], den: Sequence[float], rng: np.random.Generator
             ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """sum(num) / sum(den) with a bootstrap over seeds (pairs resampled together).
    None when the denominator is not positive."""
    n, d = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if n.size == 0 or d.sum() <= 0:
        return None, None, None
    est = float(n.sum() / d.sum())
    idx = rng.integers(0, n.size, size=(BOOTSTRAP, n.size))
    ds = d[idx].sum(axis=1)
    ok = ds > 0
    boots = n[idx].sum(axis=1)[ok] / ds[ok]
    if boots.size == 0:
        return est, None, None
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Optional[Dict[str, float]]:
    """Paired two-sided Wilcoxon signed-rank test; None when every pair is equal or n < 5."""
    from scipy.stats import wilcoxon as _w
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size < 5 or np.all(diff == 0):
        return None
    res = _w(diff)
    return {"n": int(diff.size), "statistic": float(res.statistic), "p": float(res.pvalue),
            "median_diff": float(np.median(diff))}


# ------------------------------------------------------------------ loading

def load_runs(sweep_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "runs", "*.json"))):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def by_key(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, int, float], Dict[int, Dict[str, Any]]]:
    out: Dict[Tuple[str, int, float], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in runs:
        j = r["job"]
        out[(j["kind"], j["robots"], float(j["rate"]))][j["seed"]] = r
    return out


# ----------------------------------------------------------------- per cell

def _cf_incidents(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [i["counterfactual"] for i in run["incidents"] if i.get("counterfactual")]


def cell_summary(runs: Dict[int, Dict[str, Any]], refs: Dict[int, Dict[str, Any]],
                 rng: np.random.Generator) -> Dict[str, Any]:
    seeds = sorted(runs)
    m = {s: runs[s]["metrics"] for s in seeds}
    out: Dict[str, Any] = {"seeds": seeds, "n_seeds": len(seeds)}

    # fleet level
    tp = [m[s]["throughput_per_step"] for s in seeds]
    out["throughput_per_step"] = bootstrap_ci(tp, rng)
    paired = [s for s in seeds if s in refs]
    if paired:
        ref_tp = [refs[s]["metrics"]["throughput_per_step"] for s in paired]
        loss = [100.0 * (r - m[s]["throughput_per_step"]) / r for s, r in zip(paired, ref_tp) if r > 0]
        out["reference_throughput_per_step"] = bootstrap_ci(ref_tp, rng)
        out["throughput_loss_pct"] = bootstrap_ci(loss, rng)
    out["incidents_per_run"] = bootstrap_ci([m[s]["incidents_total"] for s in seeds], rng)
    out["repair_success_rate"] = bootstrap_ci([m[s]["repair_success_rate"] for s in seeds], rng)
    out["fallback_rate"] = bootstrap_ci(
        [m[s]["fallback_invocations"] / m[s]["incidents_total"] for s in seeds if m[s]["incidents_total"]], rng)
    out["emergency_holds"] = int(sum(m[s]["emergency_holds"] for s in seeds))
    out["executed_collisions"] = int(sum(m[s]["executed_vertex_collisions"] + m[s]["executed_edge_swaps"]
                                         for s in seeds))
    out["decision_sources"] = dict(sum((_counter(m[s]["decision_sources"]) for s in seeds), _counter({})))

    # same-incident comparison
    evals = {s: _cf_incidents(runs[s]) for s in seeds}
    pooled = [e for s in seeds for e in evals[s]]
    out["incidents_compared"] = len(pooled)
    if pooled:
        out["comparison"] = comparison(evals, pooled, rng)

    # cache
    out["cache"] = cache_summary(runs, rng)
    return out


class _counter(dict):
    def __add__(self, other):
        res = _counter(self)
        for k, v in other.items():
            res[k] = res.get(k, 0) + v
        return res


def comparison(evals: Dict[int, List[dict]], pooled: List[dict], rng: np.random.Generator) -> Dict[str, Any]:
    seeds = [s for s in evals if evals[s]]
    totals = {a: (sum(e["arms"][a]["goals"] for e in pooled), sum(e["arms"][a].get("progress", 0) for e in pooled))
              for a in FIXED_ARMS}
    sbs = max(FIXED_ARMS, key=lambda a: (totals[a][0], totals[a][1], -FIXED_ARMS.index(a)))
    res: Dict[str, Any] = {"sbs": sbs, "oracle_counts": {a: sum(1 for e in pooled if a in e["oracle"])
                                                         for a in FIXED_ARMS}}

    def per_seed(fn) -> List[float]:
        return [float(np.mean([fn(e) for e in evals[s]])) for s in seeds]

    for metric in ("goals", "progress"):
        def val(arm):
            return arm.get(metric, 0)
        rows = {
            **{a: per_seed(lambda e, a=a: val(e["arms"][a])) for a in FIXED_ARMS},
            "executed": per_seed(lambda e: val(e["arms"]["executed"])),
            "random": per_seed(lambda e: np.mean([val(e["arms"][a]) for a in FIXED_ARMS])),
            "sbs": per_seed(lambda e: val(e["arms"][sbs])),
            "oracle": per_seed(lambda e: val(e["arms"][e["oracle"][0]])),
        }
        res[metric] = {k: bootstrap_ci(v, rng) for k, v in rows.items()}
        # share of the oracle gap closed, per seed totals resampled together
        for ref in ("random", "sbs"):
            num = [sum(val(e["arms"]["executed"]) for e in evals[s]) - _ref_total(evals[s], ref, sbs, val)
                   for s in seeds]
            den = [sum(val(e["arms"][e["oracle"][0]]) for e in evals[s]) - _ref_total(evals[s], ref, sbs, val)
                   for s in seeds]
            res[metric][f"gap_closed_vs_{ref}"] = ratio_ci(num, den, rng)
        res[metric]["wilcoxon_executed_vs_sbs"] = wilcoxon(rows["executed"], rows["sbs"])
        res[metric]["wilcoxon_executed_vs_random"] = wilcoxon(rows["executed"], rows["random"])
    res["distinct_rate"] = bootstrap_ci(per_seed(lambda e: 1.0 if e.get("distinct") else 0.0), rng)
    res["matches_oracle_rate"] = bootstrap_ci(per_seed(lambda e: 1.0 if e["matches_oracle"] else 0.0), rng)
    res["reproduced_rate"] = float(np.mean([1.0 if e["reproduced"] else 0.0 for e in pooled]))
    res["mean_decision_delay"] = float(np.mean([e.get("decision_delay", 0) for e in pooled]))
    return res


def _ref_total(evals: List[dict], ref: str, sbs: str, val) -> float:
    if ref == "sbs":
        return float(sum(val(e["arms"][sbs]) for e in evals))
    return float(sum(np.mean([val(e["arms"][a]) for a in FIXED_ARMS]) for e in evals))


def cache_summary(runs: Dict[int, Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
    seeds = sorted(runs)
    m = {s: runs[s]["metrics"] for s in seeds}
    lookups = sum(m[s]["cache_lookups"] for s in seeds)
    matches = sum(m[s].get("cache_matches", 0) for s in seeds)
    hits = sum(m[s]["cache_hits"] for s in seeds)
    incs = [i for s in seeds for i in runs[s]["incidents"]]
    llm_decisions = [i for i in incs if i["llm_calls"] > 0]
    out = {
        "warm_entries": runs[seeds[0]]["meta"].get("cache_warm_entries"),
        "lookups": lookups, "matches": matches, "hits": hits,
        "hit_rate": bootstrap_ci([m[s]["cache_hits"] / m[s]["cache_lookups"] for s in seeds
                                  if m[s]["cache_lookups"]], rng),
        "precision": round(hits / matches, 3) if matches else None,
        "decisions_avoided": hits,
        # tokens and seconds a hit saves = the mean cost of an LLM decision in the same runs
        "mean_tokens_per_llm_decision": (float(np.mean([i["llm_tokens"] for i in llm_decisions]))
                                         if llm_decisions else None),
        "mean_llm_latency_ms": (float(np.mean([i["llm_latency_ms"] for i in llm_decisions]))
                                if llm_decisions else None),
    }
    if out["mean_tokens_per_llm_decision"] is not None:
        out["tokens_saved"] = round(hits * out["mean_tokens_per_llm_decision"])
        out["seconds_saved"] = round(hits * out["mean_llm_latency_ms"] / 1000.0, 1)
    out["threshold_curve"] = threshold_curve(incs)
    return out


def threshold_curve(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """From the cache probes: at each threshold, the share of incidents that would match
    (coverage), the share of those matches whose dry run fails (wrong-hit rate), and the share
    whose solver is one of the incident's oracle solvers."""
    looked = [i for i in incidents if i.get("cache") is not None]
    probes = [(i["cache"]["probe"], i.get("counterfactual")) for i in looked if i["cache"].get("probe")]
    rows = []
    for th in THRESHOLDS:
        sel = [(p, c) for p, c in probes if p["similarity"] >= th]
        n = len(sel)
        rows.append({
            "threshold": th,
            "coverage": round(n / len(looked), 4) if looked else None,
            "matches": n,
            "wrong_hit_rate": round(sum(1 for p, _ in sel if not p["valid"]) / n, 4) if n else None,
            "oracle_agreement": (round(sum(1 for p, c in sel if c and p["solver"] in c["oracle"]) / n, 4)
                                 if n else None),
        })
    return rows


# ------------------------------------------------------------------ output

def analyze(sweep_dir: str) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    runs = load_runs(sweep_dir)
    keyed = by_key(runs)
    cells = sorted({(k[1], k[2]) for k in keyed if k[0] in CONDITIONS})
    summary: Dict[str, Any] = {"sweep_dir": sweep_dir, "cells": []}
    cfg_path = os.path.join(sweep_dir, "sweep_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            summary["config"] = json.load(f)
    for robots, rate in cells:
        refs = keyed.get(("reference", robots, 0.0), {})
        cell = {"robots": robots, "rate": rate}
        for cond in CONDITIONS:
            if keyed.get((cond, robots, rate)):
                cell[cond] = cell_summary(keyed[(cond, robots, rate)], refs, rng)
        summary["cells"].append(cell)
    return summary


def _f(x, nd=3, pct=False) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.{max(0, nd - 2)}f}%" if pct else f"{x:.{nd}f}"


def _ci(t, nd=3, pct=False) -> str:
    if t is None or t[0] is None:
        return "—"
    est, lo, hi = t
    if lo is None:
        return _f(est, nd, pct)
    return f"{_f(est, nd, pct)} [{_f(lo, nd, pct)}, {_f(hi, nd, pct)}]"


def _p(w) -> str:
    return "— (all equal)" if w is None else f"p={w['p']:.3g} (median Δ {w['median_diff']:+.2f})"


def write_markdown(summary: Dict[str, Any], path: str) -> None:
    cfg = summary.get("config", {})
    L = ["# Scale sweep: results", ""]
    if cfg:
        L += [f"Map `{cfg.get('map')}`, {cfg.get('steps')} steps, overlap mode `{cfg.get('overlap')}`, "
              f"strategist `{cfg.get('provider')}`, counterfactual window {cfg.get('window')} steps, "
              f"cache threshold {cfg.get('threshold')}. Test seeds {cfg.get('seeds')}; "
              f"cache training seeds {cfg.get('train_seeds')}.", ""]
    L += ["Values are the mean over test seeds with a 95% bootstrap interval in brackets.", ""]

    L += ["## Fleet throughput", "",
          "| Robots | Rate | Cache | Goals/step | No-disruption goals/step | Throughput lost | Incidents/run | "
          "Repair success | Fallback rate | Holds | Collisions |", "|" + " --- |" * 11]
    for c in summary["cells"]:
        for cond in CONDITIONS:
            s = c.get(cond)
            if not s:
                continue
            L.append(f"| {c['robots']} | {c['rate']:g} | {cond} | {_ci(s['throughput_per_step'])} | "
                     f"{_ci(s.get('reference_throughput_per_step'))} | {_ci(s.get('throughput_loss_pct'), 2)}% | "
                     f"{_ci(s['incidents_per_run'], 1)} | {_ci(s['repair_success_rate'], 3)} | "
                     f"{_ci(s['fallback_rate'], 3)} | {s['emergency_holds']} | {s['executed_collisions']} |")

    for metric, unit in (("progress", "net steps toward goals, whole fleet"), ("goals", "goals, whole fleet")):
        L += ["", f"## Same-incident comparison: {metric} per incident ({unit})", "",
              "| Robots | Rate | Cache | Incidents | Solvers differ | PIBT | LNS | Windowed PBS | Random | SBS | "
              "Oracle | Executed | Gap closed vs random | Gap closed vs SBS | Executed vs SBS | Executed vs random |",
              "|" + " --- |" * 16]
        for c in summary["cells"]:
            for cond in CONDITIONS:
                s = c.get(cond)
                if not s or "comparison" not in s:
                    continue
                k = s["comparison"]
                r = k[metric]
                nd = 1
                L.append(f"| {c['robots']} | {c['rate']:g} | {cond} | {s['incidents_compared']} | "
                         f"{_ci(k['distinct_rate'], 3, True)} | {_ci(r['pibt'], nd)} | {_ci(r['lns'], nd)} | "
                         f"{_ci(r['windowed_pbs'], nd)} | {_ci(r['random'], nd)} | {k['sbs']}: {_ci(r['sbs'], nd)} | "
                         f"{_ci(r['oracle'], nd)} | {_ci(r['executed'], nd)} | {_ci(r['gap_closed_vs_random'], 2)} | "
                         f"{_ci(r['gap_closed_vs_sbs'], 2)} | {_p(r['wilcoxon_executed_vs_sbs'])} | "
                         f"{_p(r['wilcoxon_executed_vs_random'])} |")

    L += ["", "## Oracle choices and decisions", "",
          "| Robots | Rate | Cache | Oracle solver counts | Executed matches oracle | Decision sources |",
          "|" + " --- |" * 6]
    for c in summary["cells"]:
        for cond in CONDITIONS:
            s = c.get(cond)
            if not s or "comparison" not in s:
                continue
            k = s["comparison"]
            L.append(f"| {c['robots']} | {c['rate']:g} | {cond} | "
                     f"{', '.join(f'{a} {n}' for a, n in k['oracle_counts'].items())} | "
                     f"{_ci(k['matches_oracle_rate'], 3, True)} | "
                     f"{', '.join(f'{a} {n}' for a, n in sorted(s['decision_sources'].items()))} |")

    L += ["", "## Strategy cache (cold = empty at start, warm = filled on the training seeds)", "",
          "| Robots | Rate | Cache | Entries at start | Lookups | Matches | Hits (dry run passed) | Hit rate | "
          "Precision | Decisions avoided | Tokens saved |", "|" + " --- |" * 11]
    for c in summary["cells"]:
        for cond in CONDITIONS:
            s = c.get(cond)
            if not s:
                continue
            k = s["cache"]
            tok = k.get("tokens_saved")
            L.append(f"| {c['robots']} | {c['rate']:g} | {cond} | {k['warm_entries'] or 0} | {k['lookups']} | "
                     f"{k['matches']} | {k['hits']} | {_ci(k['hit_rate'], 3, True)} | {_f(k['precision'])} | "
                     f"{k['decisions_avoided']} | {tok if tok is not None else '— (no LLM in this sweep)'} |")

    L += ["", "## Cache threshold curve (warm cache, from the probes)", "",
          "For each threshold: share of incidents a stored strategy would match, share of those matches that fail "
          "their dry run, and share whose solver is one of the incident's oracle solvers.", ""]
    for c in summary["cells"]:
        s = c.get("warm")
        if not s:
            continue
        L += [f"**{c['robots']} robots, rate {c['rate']:g}**", "",
              "| Threshold | Coverage | Matches | Wrong-hit rate | Oracle agreement |", "|" + " --- |" * 5]
        for row in s["cache"]["threshold_curve"]:
            L.append(f"| {row['threshold']} | {_f(row['coverage'], 3, True)} | {row['matches']} | "
                     f"{_f(row['wrong_hit_rate'], 3, True)} | {_f(row['oracle_agreement'], 3, True)} |")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", nargs="?", default="runs/sweep")
    args = p.parse_args(argv)
    summary = analyze(args.sweep_dir)
    js = os.path.join(args.sweep_dir, "summary.json")
    md = os.path.join(args.sweep_dir, "summary.md")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    write_markdown(summary, md)
    print(f"[Artifact] {js}\n[Artifact] {md}")


if __name__ == "__main__":
    main()
