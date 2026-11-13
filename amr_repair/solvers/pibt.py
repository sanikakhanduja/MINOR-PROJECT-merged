"""Priority Inheritance with Backtracking (PIBT), Okumura et al. 2022.

Plans the given robots one timestep at a time for ``horizon`` steps. In each
step robots choose, in priority order, the free neighbour closest to their
goal (true distance). If the chosen cell is occupied by a lower-priority
robot, that robot inherits the priority and is pushed recursively; if it
cannot move, the higher robot backtracks to its next candidate.

Robots not being replanned are hard space-time obstacles (vertex and swap).
PIBT is fast and myopic: it does not guarantee that the final cells are safe
to park in, so the validator may reject it in congested areas — which is
exactly the trade-off the strategist has to reason about.
"""

from typing import Dict, List, Optional, Tuple

from amr_repair.sim.lifelong import World
from amr_repair.sim.planning import build_table
from amr_repair.solvers.common import plannable

Cell = Tuple[int, int]


def solve(world: World, robots: List[int], priority: List[int], horizon: int) -> Dict[int, List[Cell]]:
    order = plannable(world, robots, priority)
    if not order:
        return {}
    by_id = {r.id: r for r in world.robots}
    others = build_table(world.robots, exclude=order)
    dist = {rid: world.map.distance_map(by_id[rid].goal) for rid in order}

    cur: Dict[int, Cell] = {rid: by_id[rid].pos for rid in order}
    paths: Dict[int, List[Cell]] = {rid: [] for rid in order}

    for k in range(max(1, horizon)):
        nxt: Dict[int, Cell] = {}
        reserved_by: Dict[Cell, int] = {}
        cur_occ = {c: rid for rid, c in cur.items()}

        def pibt(r: int, parent: Optional[int]) -> bool:
            c = cur[r]
            if k < by_id[r].delay_remaining:
                cands = [c]
            else:
                d = dist[r]
                cands = world.map.neighbors(c) + [c]
                cands.sort(key=lambda v: (d.get(v, 10**6), v == c))
            for v in cands:
                if v in reserved_by:
                    continue
                if others.occupant(v, k + 1) is not None:
                    continue
                if v != c and others.swap_partner(c, v, k) is not None:
                    continue
                if parent is not None and v == cur[parent]:
                    continue  # would swap with the robot pushing us
                nxt[r] = v
                reserved_by[v] = r
                j = cur_occ.get(v)
                if j is not None and j != r and j not in nxt:
                    if pibt(j, r):
                        return True
                    # j could not move away: backtrack
                    del nxt[r]
                    if reserved_by.get(v) == r:
                        del reserved_by[v]
                    continue
                return True
            # No feasible move: stay (may collide; the validator will catch it)
            nxt[r] = c
            reserved_by.setdefault(c, r)
            return False

        for rid in order:
            if rid not in nxt:
                pibt(rid, None)
        for rid in order:
            cur[rid] = nxt.get(rid, cur[rid])
            paths[rid].append(cur[rid])

    return paths
