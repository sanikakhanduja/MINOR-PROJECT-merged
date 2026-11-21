"""Windowed Priority-Based Search (PBS, Ma et al. 2019) for local repair.

High level: depth-first search over *partial priority orderings*. The root
plans every robot independently (around the rest of the fleet). At each node
the earliest conflict (a, b) between replanned robots is found and two
children are generated, adding a < b or b < a; in each child the
lower-priority robot and everything below it is replanned with cooperative
space-time A* around all higher-priority robots. The child consistent with
the directive's priority order is explored first. Planning is windowed to
``horizon`` steps. Bounded by 300 expanded nodes (no wall-clock budget by
default, so a seeded run gives the same plans on any machine); if no
conflict-free node is found the best node is returned and the validator will
reject it.
"""

import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from amr_repair.sim.lifelong import World
from amr_repair.sim.planning import build_table, future_cell, plan_path
from amr_repair.solvers.common import plannable

Cell = Tuple[int, int]
Edge = Tuple[int, int]  # (higher, lower)


def _first_conflict(world: World, order: List[int], paths: Dict[int, List[Cell]]) -> Optional[Tuple[int, int]]:
    by_id = {r.id: r for r in world.robots}
    T = max((len(p) for p in paths.values()), default=0) + 1
    for k in range(1, T + 1):
        seen: Dict[Cell, int] = {}
        for rid in order:
            c = future_cell(by_id[rid].pos, paths[rid], k)
            if c in seen:
                return seen[c], rid
            seen[c] = rid
        prev = {rid: future_cell(by_id[rid].pos, paths[rid], k - 1) for rid in order}
        now = {rid: future_cell(by_id[rid].pos, paths[rid], k) for rid in order}
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if prev[a] == now[b] and prev[b] == now[a] and prev[a] != prev[b]:
                    return a, b
    return None


def _ancestors(edges: FrozenSet[Edge], node: int) -> Set[int]:
    out: Set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        for hi, lo in edges:
            if lo == n and hi not in out:
                out.add(hi)
                stack.append(hi)
    return out


def _descendants(edges: FrozenSet[Edge], node: int) -> Set[int]:
    out: Set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        for hi, lo in edges:
            if hi == n and lo not in out:
                out.add(lo)
                stack.append(lo)
    return out


def solve(world: World, robots: List[int], priority: List[int], horizon: int,
          time_budget_s: Optional[float] = None, max_nodes: int = 300) -> Dict[int, List[Cell]]:
    t_start = time.perf_counter()
    order = plannable(world, robots, priority)
    if not order:
        return {}
    by_id = {r.id: r for r in world.robots}
    rank = {rid: i for i, rid in enumerate(order)}

    def plan_one(rid: int, higher: Set[int], paths: Dict[int, List[Cell]]) -> Optional[List[Cell]]:
        table = build_table(world.robots, exclude=order)
        for h in higher:
            table.add_path(h, by_id[h].pos, paths[h])
        r = by_id[rid]
        p = plan_path(world.map, r.pos, r.goal, horizon, table, wait_until=r.delay_remaining)
        if p is not None and not p:
            p = [r.pos]
        return p

    root: Dict[int, List[Cell]] = {}
    for rid in order:
        p = plan_one(rid, set(), root)
        root[rid] = p if p is not None else [by_id[rid].pos] * max(1, horizon)

    stack: List[Tuple[FrozenSet[Edge], Dict[int, List[Cell]]]] = [(frozenset(), root)]
    best_paths, nodes = root, 0
    while stack and nodes < max_nodes and (time_budget_s is None or time.perf_counter() - t_start < time_budget_s):
        edges, paths = stack.pop()
        nodes += 1
        conflict = _first_conflict(world, order, paths)
        if conflict is None:
            return paths
        best_paths = paths
        a, b = conflict
        children = []
        for hi, lo in ((a, b), (b, a)):
            if hi in _descendants(edges, lo) or (lo, hi) in edges:
                continue  # would create a priority cycle
            new_edges = edges | {(hi, lo)}
            new_paths = dict(paths)
            # replan lo and everything below it, in topological order
            to_replan = [lo] + sorted(_descendants(new_edges, lo), key=lambda x: len(_ancestors(new_edges, x)))
            ok = True
            for rid in to_replan:
                p = plan_one(rid, _ancestors(new_edges, rid), new_paths)
                if p is None:
                    ok = False
                    break
                new_paths[rid] = p
            if ok:
                preferred = rank[hi] < rank[lo]
                children.append((preferred, new_edges, new_paths))
        # push the non-preferred child first so the preferred one is expanded next
        for _pref, e, p in sorted(children, key=lambda c: c[0]):
            stack.append((e, p))
    return best_paths
