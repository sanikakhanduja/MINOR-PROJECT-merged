"""Counterfactual per-incident evaluation (branching from the detection state).

Why this exists
---------------
In a closed-loop lifelong run, two methods diverge after their first different
repair, so incident k of one run is not incident k of another run. Pairing
incidents across separate runs therefore compares different situations. To
compare repair strategies on the *same* incident, the controller forks the
world at detection time and evaluates every arm from that identical state.

Arms
----
``pibt`` / ``lns`` / ``windowed_pbs``
    The fixed baselines, exactly as the baseline modes run them: all plannable
    affected robots, nearest-to-goal first, horizon 20.
``executed``
    The directive the run actually executed (LLM, cache, rule-based or
    fallback). No LLM is called again.

Every arm goes through the same chain as the controller: directive ->
validator -> (if invalid) fixed fallback (windowed PBS, horizon 30) -> (if
still invalid) emergency hold. Each branch is then rolled forward ``window``
steps with the world's normal lifelong planner and **no new disruptions**
(temporary blockages that expire inside the window still expire), so the
measured differences are caused by this incident's repair alone.

Measured per arm
----------------
* ``first_valid``: the arm's own directive passed the validator
* ``source``: ``direct`` | ``fallback`` | ``hold``
* ``goals``: goals completed by the whole fleet in the window (lifelong throughput)
* ``progress``: net steps the whole fleet moved toward its goals in the window
  (+1 for a step closer, -1 for a step away, 0 for a wait). Goals take tens of
  steps, so two repairs can finish the same number of goals inside a short
  window while one of them has left the fleet much closer to the next ones;
  progress separates them.
* ``recovery_steps`` / ``recovered``: same definition as the main run
  (every affected robot failed, completed a goal, or is closer to its goal than
  at detection; censored at ``window``)
* ``detour``: extra steps of the repair over single-agent shortest paths
* ``solver_ms``: wall-clock time of the solver call(s)

Oracle
------
The virtual best solver (VBS) is the best *fixed* arm per incident under
``rank_key``: no emergency hold, then most fleet goals, then most fleet
progress, then fastest recovery, then smallest detour. Ties are kept as a set.
An incident is *decision-relevant* when the fixed arms do not all tie: only
there can a choice of solver matter.

Decision delay
--------------
When the run charged the LLM's thinking time to the robots (controller
``decision_step_seconds``), the executed arm applies the same delay before its
repair, so it reproduces the executed plan. The fixed arms have no delay: they
are what a fixed solver would have done without asking anyone.
"""

import copy
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

from amr_repair.llm.schema import Directive
from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.sim.lifelong import World
from amr_repair.validate.validator import validate

Cell = Tuple[int, int]
Paths = Dict[int, List[Cell]]
SolverFn = Callable[[World, List[int], List[int], int], Paths]

FIXED_ARMS: Tuple[str, ...] = ("pibt", "lns", "windowed_pbs")
BASELINE_HORIZON = 20
UNREACHABLE = 10_000


# ---------------------------------------------------------------- shared rules
# These are the single definitions used by both the controller and the branches.

def fixed_directive(world: World, required: List[int], solver: str, horizon: int, why: str) -> Directive:
    robots = {r.id: r for r in world.robots}
    prio = sorted(required, key=lambda r: (world.map.distance(robots[r].pos, robots[r].goal), r))
    return Directive(robots=sorted(required), solver=solver, priority=prio, horizon=horizon, rationale=why)


def sanitize(world: World, d: Directive, incident: IncidentRecord) -> Directive:
    """Never let a directive touch failed or non-existent robots."""
    alive = {r.id for r in world.robots if r.state != "failed"}
    allowed = alive & (set(incident.candidate_robots) | set(incident.plannable_affected()))
    robots = [r for r in d.robots if r in allowed] or sorted(set(incident.plannable_affected()) & alive)
    prio = [r for r in d.priority if r in robots] + [r for r in robots if r not in d.priority]
    if robots == d.robots and prio == d.priority:
        return d
    return Directive(robots=robots, solver=d.solver, priority=prio, horizon=d.horizon, rationale=d.rationale)


def detour_cost(world: World, paths: Paths) -> Tuple[int, int]:
    """(extra steps over single-agent shortest paths, sum of those shortest paths)."""
    robots = {r.id: r for r in world.robots}
    extra = shortest = 0
    for rid, p in paths.items():
        r = robots[rid]
        d0 = world.map.distance(r.pos, r.goal)
        if d0 >= UNREACHABLE:
            continue
        used = (p.index(r.goal) + 1) if r.goal in p else len(p) + world.map.distance(p[-1], r.goal)
        extra += max(0, used - d0)
        shortest += d0
    return int(extra), int(shortest)


def recovery_baseline(world: World, required: List[int]) -> Dict[int, Tuple[int, int]]:
    """Per robot: (distance to goal, goals completed) at detection."""
    robots = {r.id: r for r in world.robots}
    return {rid: (world.map.distance(robots[rid].pos, robots[rid].goal), robots[rid].goals_completed)
            for rid in required}


def progress_resumed(world: World, baseline: Dict[int, Tuple[int, int]]) -> bool:
    """Every tracked robot failed, completed a goal, or is closer to its goal than at detection."""
    robots = {r.id: r for r in world.robots}
    for rid, (d0, gc0) in baseline.items():
        r = robots[rid]
        if r.state == "failed" or r.goals_completed > gc0:
            continue
        if world.map.distance(r.pos, r.goal) >= d0:
            return False
    return True


def apply_paths(world: World, paths: Paths, t: int) -> None:
    robots = {r.id: r for r in world.robots}
    for rid, p in paths.items():
        robots[rid].path = list(p)
        robots[rid].repair_until = t + len(p)


def charge_delay(world: World, robots: List[int], steps: int) -> None:
    """The robots wait ``steps`` steps for a decision before their repair starts."""
    if steps <= 0:
        return
    by_id = {r.id: r for r in world.robots}
    for rid in robots:
        r = by_id[rid]
        if r.state != "failed":
            r.delay_remaining = max(r.delay_remaining, steps)


def emergency_hold(world: World, required: List[int]) -> None:
    robots = {r.id: r for r in world.robots}
    for rid in required:
        r = robots[rid]
        r.path = [r.pos] * r.delay_remaining


# ------------------------------------------------------------------- forking

def fork_world(world: World) -> World:
    """Independent deep copy of the world without its frame history.

    The copy shares no mutable state with the original, so a branch can never
    change the main run (including its RNG streams). The one exception is the
    distance cache: its BFS tables are never modified after they are built and
    are keyed by map version, so the clone gets a new cache container holding
    the same tables (a branch that blocks or unblocks cells moves to a new map
    version and computes its own)."""
    frames, cache = world.frames, world.map._dist_cache
    world.frames, world.map._dist_cache = [], OrderedDict()
    try:
        clone = copy.deepcopy(world)
    finally:
        world.frames, world.map._dist_cache = frames, cache
    clone.map._dist_cache = OrderedDict(cache)
    return clone


# ------------------------------------------------------------------ one branch

def run_arm(snapshot: World, incident: IncidentRecord, required: List[int], directive: Optional[Directive],
            solvers: Dict[str, SolverFn], unblocks: Dict[int, List[Cell]], window: int,
            fallback_horizon: int = 30, decision_delay: int = 0) -> Tuple[dict, Paths]:
    """Evaluate one arm on a fresh fork of ``snapshot``. ``directive=None`` means emergency hold."""
    world = fork_world(snapshot)
    t = world.t
    solver_ms = 0.0
    if directive is not None:
        charge_delay(world, required, decision_delay)

    def attempt(d: Directive) -> Tuple[Directive, Paths, bool]:
        nonlocal solver_ms
        d = sanitize(world, d, incident)
        t0 = time.perf_counter()
        paths = solvers[d.solver](world, d.robots, d.priority, d.horizon)
        solver_ms += (time.perf_counter() - t0) * 1000
        return d, paths, validate(paths, world, t, required=required).valid

    source, paths, first_valid = "hold", {}, False
    if directive is not None:
        _, paths, first_valid = attempt(directive)
        source = "direct"
        if not first_valid:
            fb = fixed_directive(world, required, "windowed_pbs", fallback_horizon, "fallback")
            _, paths, ok = attempt(fb)
            source = "fallback" if ok else "hold"
            if not ok:
                paths = {}

    detour, shortest = detour_cost(world, paths if paths else {r: [world.robots[r].pos] for r in required})
    baseline = recovery_baseline(world, required)
    if paths:
        apply_paths(world, paths, t)
    else:
        emergency_hold(world, required)

    goals0 = world.goals_completed
    recovery: Optional[int] = None
    progress = 0
    dist = world.map.distance
    for _ in range(window):
        before = [(r, dist(r.pos, r.goal), r.goals_completed) for r in world.robots if r.state != "failed"]
        world.step()
        for r, d0, g0 in before:
            progress += d0 if r.goals_completed > g0 else d0 - dist(r.pos, r.goal)
        if recovery is None and progress_resumed(world, baseline):
            recovery = world.t - t
        cells = unblocks.get(world.t)
        if cells:
            world.map.unblock(cells)

    return {
        "solver": directive.solver if directive is not None else None,
        "horizon": directive.horizon if directive is not None else None,
        "first_valid": bool(first_valid),
        "source": source,
        "goals": int(world.goals_completed - goals0),
        "progress": int(progress),
        "recovery_steps": int(recovery if recovery is not None else window),
        "recovered": recovery is not None,
        "detour": int(detour),
        "shortest": int(shortest),
        "solver_ms": round(solver_ms, 2),
    }, paths


def rank_key(arm: dict) -> Tuple[int, int, int, int, int]:
    """Smaller is better: no hold, more goals, more progress, faster recovery, smaller detour.
    (Arms recorded before ``progress`` existed rank as if it were equal.)"""
    return (1 if arm["source"] == "hold" else 0, -arm["goals"], -arm.get("progress", 0),
            arm["recovery_steps"], arm["detour"])


def evaluate_incident(snapshot: World, incident: IncidentRecord, required: List[int],
                      executed: Optional[Directive], executed_paths: Paths, solvers: Dict[str, SolverFn],
                      unblocks: Dict[int, List[Cell]], window: int, fallback_horizon: int = 30,
                      decision_delay: int = 0) -> dict:
    """All arms for one incident, plus the oracle comparison for the executed decision."""
    arms: Dict[str, dict] = {}
    for name in FIXED_ARMS:
        d = fixed_directive(snapshot, required, name, BASELINE_HORIZON, "Fixed baseline")
        arms[name], _ = run_arm(snapshot, incident, required, d, solvers, unblocks, window, fallback_horizon)
    arms["executed"], replayed = run_arm(snapshot, incident, required, executed, solvers, unblocks, window,
                                         fallback_horizon, decision_delay)

    best = min(rank_key(arms[n]) for n in FIXED_ARMS)
    oracle = [n for n in FIXED_ARMS if rank_key(arms[n]) == best]
    ex = arms["executed"]
    best_goals = max(arms[n]["goals"] for n in FIXED_ARMS)
    return {
        "window": window,
        "decision_delay": int(decision_delay),
        "arms": arms,
        "oracle": oracle,
        # the fixed arms do not all tie, so the choice of solver mattered on this incident
        "distinct": len(oracle) < len(FIXED_ARMS),
        # executed decision is at least as good as the best fixed arm
        "matches_oracle": rank_key(ex) <= best,
        # fleet goals the executed decision lost against the best fixed arm (negative = it did better)
        "goal_regret": int(best_goals - ex["goals"]),
        # the branch re-produced exactly the plan the run executed (False can happen when a
        # solver hits its wall-clock budget at a different point)
        "reproduced": _same_paths(replayed, executed_paths),
    }


def _same_paths(a: Paths, b: Paths) -> bool:
    return {int(k): [tuple(c) for c in v] for k, v in a.items()} == \
           {int(k): [tuple(c) for c in v] for k, v in b.items()}


# --------------------------------------------------------------- aggregation

def summarize(evals: List[dict]) -> dict:
    """Run-level comparison on identical incidents.

    * per arm: mean fleet goals and progress in the window, mean recovery,
      valid-first-try rate, hold rate
    * ``random``: exact expectation of picking a fixed arm uniformly at random
    * ``vbs``: virtual best solver (per-incident oracle under ``rank_key``)
    * ``sbs``: single best solver (the fixed arm with the most total goals, then progress)
    * ``gap_closed_vs_*``: (executed - ref) / (vbs - ref), on goals and on progress.
      1 = as good as the oracle, 0 = no better than the reference, None = the oracle
      is no better than the reference (nothing to close).
    * ``distinct_rate``: share of incidents where the fixed arms do not all tie
    """
    n = len(evals)
    if n == 0:
        return {"incidents": 0}

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    names = list(FIXED_ARMS) + ["executed"]
    per_arm = {}
    for name in names:
        rows = [e["arms"][name] for e in evals]
        per_arm[name] = {
            "mean_goals": mean([r["goals"] for r in rows]),
            "mean_progress": mean([r.get("progress", 0) for r in rows]),
            "mean_recovery_steps": mean([r["recovery_steps"] for r in rows]),
            "recovered_rate": mean([1.0 if r["recovered"] else 0.0 for r in rows]),
            "first_valid_rate": mean([1.0 if r["first_valid"] else 0.0 for r in rows]),
            "hold_rate": mean([1.0 if r["source"] == "hold" else 0.0 for r in rows]),
            "mean_detour": mean([r["detour"] for r in rows]),
            "total_goals": int(sum(r["goals"] for r in rows)),
            "total_progress": int(sum(r.get("progress", 0) for r in rows)),
        }

    def vbs_arm(e) -> dict:  # one oracle arm (all oracle arms share the rank key)
        return e["arms"][e["oracle"][0]]

    oracle_counts = {a: sum(1 for e in evals if a in e["oracle"]) for a in FIXED_ARMS}
    # single best solver: best total over all incidents (goals, then progress), as in algorithm selection
    sbs_name = max(FIXED_ARMS, key=lambda a: (per_arm[a]["total_goals"], per_arm[a]["total_progress"],
                                              -FIXED_ARMS.index(a)))
    out = {
        "incidents": n,
        "window": evals[0]["window"],
        "arms": per_arm,
        "sbs": sbs_name,
        "oracle_counts": oracle_counts,
        "distinct_rate": mean([1.0 if e.get("distinct", len(e["oracle"]) < len(FIXED_ARMS)) else 0.0
                               for e in evals]),
        "executed_matches_oracle_rate": mean([1.0 if e["matches_oracle"] else 0.0 for e in evals]),
        "mean_goal_regret": mean([e["goal_regret"] for e in evals]),
        "reproduced_rate": mean([1.0 if e["reproduced"] else 0.0 for e in evals]),
        "mean_decision_delay": mean([e.get("decision_delay", 0) for e in evals]),
    }
    for metric, label in (("goals", "goals"), ("progress", "progress")):
        def val(arm):
            return arm.get(metric, 0)
        vbs = [val(vbs_arm(e)) for e in evals]
        rnd = [sum(val(e["arms"][a]) for a in FIXED_ARMS) / len(FIXED_ARMS) for e in evals]
        sbs = [val(e["arms"][sbs_name]) for e in evals]
        ex = [val(e["arms"]["executed"]) for e in evals]

        def gap(ref: List[float]) -> Optional[float]:
            denom = sum(vbs) - sum(ref)
            return round((sum(ex) - sum(ref)) / denom, 3) if denom > 0 else None

        out[f"random_mean_{label}"] = mean(rnd)
        out[f"vbs_mean_{label}"] = mean(vbs)
        out[f"sbs_mean_{label}"] = mean(sbs)
        out[f"gap_closed_vs_random{'' if label == 'goals' else '_progress'}"] = gap(rnd)
        out[f"gap_closed_vs_sbs{'' if label == 'goals' else '_progress'}"] = gap(sbs)
    return out
