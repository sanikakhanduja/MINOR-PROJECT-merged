"""Large Neighbourhood Search (MAPF-LNS, Li et al. 2021) for local repair.

1. Initial solution: prioritized planning (cooperative space-time A*) in the
   directive's priority order; if that fails, a few deterministic random
   restarts, then PIBT as a last resort.
2. Improvement: repeatedly destroy a neighbourhood of 2-5 robots and replan
   them in random order around everyone else; accept the new solution if it
   is complete and its sum of costs (steps + true remaining distance) does not
   increase. Stops after 50 iterations. There is no wall-clock budget by
   default, so a seeded run gives the same plans on any machine.
"""

import time
from typing import Dict, List, Optional, Tuple

from amr_repair.sim.lifelong import World
from amr_repair.sim.planning import build_table, plan_path
from amr_repair.solvers import pibt
from amr_repair.solvers.common import plannable, soc, solver_rng

Cell = Tuple[int, int]


def _prioritized(world: World, order: List[int], seq: List[int], fixed: Dict[int, List[Cell]],
                 horizon: int) -> Optional[Dict[int, List[Cell]]]:
    by_id = {r.id: r for r in world.robots}
    table = build_table(world.robots, exclude=order)
    for rid, p in fixed.items():
        table.add_path(rid, by_id[rid].pos, p)
    out: Dict[int, List[Cell]] = {}
    for rid in seq:
        r = by_id[rid]
        p = plan_path(world.map, r.pos, r.goal, horizon, table, wait_until=r.delay_remaining)
        if p is None:
            return None
        if not p:  # already at goal and safe: hold one step
            p = [r.pos]
        out[rid] = p
        table.add_path(rid, r.pos, p)
    return out


def solve(world: World, robots: List[int], priority: List[int], horizon: int,
          iterations: int = 50, time_budget_s: Optional[float] = None) -> Dict[int, List[Cell]]:
    t_start = time.perf_counter()
    order = plannable(world, robots, priority)
    if not order:
        return {}
    rng = solver_rng(world, order)

    best = _prioritized(world, order, order, {}, horizon)
    restarts = 0
    while best is None and restarts < 4:
        seq = list(order)
        rng.shuffle(seq)
        best = _prioritized(world, order, seq, {}, horizon)
        restarts += 1
    if best is None:
        return pibt.solve(world, order, order, horizon)

    best_cost = soc(world, best)
    if len(order) < 2:
        return best
    for _ in range(iterations):
        if time_budget_s is not None and time.perf_counter() - t_start > time_budget_s:
            break
        size = int(rng.integers(2, min(5, len(order)) + 1))
        destroyed = list(rng.choice(order, size=size, replace=False))
        rng.shuffle(destroyed)
        fixed = {rid: p for rid, p in best.items() if rid not in destroyed}
        repaired = _prioritized(world, order, [int(x) for x in destroyed], fixed, horizon)
        if repaired is None:
            continue
        candidate = {**fixed, **repaired}
        cost = soc(world, candidate)
        if cost <= best_cost:
            best, best_cost = candidate, cost
    return best
