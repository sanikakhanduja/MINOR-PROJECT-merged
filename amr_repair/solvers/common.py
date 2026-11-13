"""Helpers shared by the repair solvers.

Every solver has the signature ``solve(world, robots, priority, horizon)`` and
returns ``{robot_id: path}`` with paths in the k = 1..L convention of
:mod:`amr_repair.sim.planning`. Solvers plan *only* the given robots and treat
every other robot's committed plan as a hard space-time reservation.
Solvers never mutate the world; the controller commits a plan only after the
validator approves it.
"""

from typing import Dict, List, Tuple

import numpy as np

from amr_repair.sim.lifelong import World

Cell = Tuple[int, int]


def plannable(world: World, robots: List[int], priority: List[int]) -> List[int]:
    """Robots to plan, in priority order: existing, not failed, de-duplicated.
    Robots missing from ``priority`` are appended in ``robots`` order."""
    by_id = {r.id: r for r in world.robots}
    order: List[int] = []
    for rid in list(priority) + list(robots):
        if rid in by_id and rid in robots and rid not in order and by_id[rid].state != "failed":
            order.append(rid)
    return order


def solver_rng(world: World, robots: List[int]) -> np.random.Generator:
    """Deterministic RNG for stochastic solvers that does NOT consume the
    world's RNG (so different solvers see identical disruption/goal streams)."""
    return np.random.default_rng([world.t, len(robots), *sorted(robots)[:8]])


def hold_paths(world: World, robots: List[int], horizon: int) -> Dict[int, List[Cell]]:
    by_id = {r.id: r for r in world.robots}
    return {rid: [by_id[rid].pos] * max(1, horizon) for rid in robots}


def soc(world: World, paths: Dict[int, List[Cell]]) -> int:
    """Sum of costs: steps used + true remaining distance to goal (lower is better)."""
    by_id = {r.id: r for r in world.robots}
    total = 0
    for rid, p in paths.items():
        r = by_id[rid]
        if not p:
            total += world.map.distance(r.pos, r.goal)
            continue
        if r.goal in p:
            total += p.index(r.goal) + 1
        else:
            total += len(p) + world.map.distance(p[-1], r.goal)
    return total
