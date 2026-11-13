"""Shared multi-agent planning primitives.

Time convention used everywhere in the package
---------------------------------------------
* ``k`` is time *relative to now*: ``k = 0`` is the current step.
* A robot's ``path`` lists the cells it will occupy at ``k = 1, 2, ...``.
* After its path ends, a robot is **parked** at the last cell (or at its
  current position if the path is empty) for all later times.

The :class:`ReservationTable` stores the committed plans of a set of robots in
this convention, and :func:`plan_path` is a cooperative space-time A*
(Silver 2005, "Cooperative pathfinding") that plans one robot around those
reservations. The lifelong simulator, all three repair solvers and the
validator use these same primitives, so "collision-free" means the same thing
everywhere.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from amr_repair.sim.agents import Robot
    from amr_repair.sim.grid import GridMap

Cell = Tuple[int, int]


def manhattan_distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def future_cell(start: Cell, path: List[Cell], k: int) -> Cell:
    """Cell occupied ``k`` steps from now by a robot at ``start`` following ``path``."""
    if k <= 0:
        return start
    if k - 1 < len(path):
        return path[k - 1]
    return path[-1] if path else start


class ReservationTable:
    """Space-time reservations of a set of robots (vertex, edge and parking)."""

    def __init__(self) -> None:
        self.vertex: Dict[int, Dict[Cell, int]] = defaultdict(dict)
        self.edges: Dict[int, Dict[Tuple[Cell, Cell], int]] = defaultdict(dict)
        self.parked: Dict[Cell, Tuple[int, int]] = {}  # cell -> (robot, from_k)
        self.last_visit: Dict[Cell, int] = {}  # cell -> latest k it is visited (excl. parking)
        self.robots: Set[int] = set()

    def add_path(self, rid: int, start: Cell, path: List[Cell]) -> None:
        positions = [start] + list(path)
        for k, c in enumerate(positions):
            self.vertex[k][c] = rid
            if self.last_visit.get(c, -1) < k:
                self.last_visit[c] = k
        for k in range(len(positions) - 1):
            a, b = positions[k], positions[k + 1]
            if a != b:
                self.edges[k][(a, b)] = rid
        self.parked[positions[-1]] = (rid, len(positions) - 1)
        self.robots.add(rid)

    def occupant(self, cell: Cell, k: int) -> Optional[int]:
        rid = self.vertex.get(k, {}).get(cell)
        if rid is not None:
            return rid
        p = self.parked.get(cell)
        if p is not None and k >= p[1]:
            return p[0]
        return None

    def swap_partner(self, a: Cell, b: Cell, k: int) -> Optional[int]:
        """Robot that moves b -> a between k and k+1 (would swap with a -> b)."""
        return self.edges.get(k, {}).get((b, a))

    def safe_to_park(self, cell: Cell, k: int) -> bool:
        """True if nobody in the table occupies ``cell`` at any time >= k."""
        if cell in self.parked:
            return False
        return self.last_visit.get(cell, -1) < k

    def first_conflict(self, rid: int, start: Cell, path: List[Cell]) -> Optional[str]:
        """Describe the first conflict of ``path`` against this table, or None."""
        positions = [start] + list(path)
        for k in range(1, len(positions)):
            c = positions[k]
            other = self.occupant(c, k)
            if other is not None:
                return f"vertex collision robots {min(rid, other)},{max(rid, other)} cell [{c[0]},{c[1]}] k={k}"
            prev = positions[k - 1]
            if prev != c:
                other = self.swap_partner(prev, c, k - 1)
                if other is not None:
                    return (f"edge swap robots {min(rid, other)},{max(rid, other)} "
                            f"cells [{prev[0]},{prev[1]}]->[{c[0]},{c[1]}] k={k - 1}")
        end = positions[-1]
        end_k = len(positions) - 1
        if not self.safe_to_park(end, end_k + 1):
            other = self.parked.get(end, (None,))[0]
            if other is None:
                # find who visits later
                for kk in range(end_k + 1, self.last_visit.get(end, end_k) + 1):
                    other = self.vertex.get(kk, {}).get(end)
                    if other is not None:
                        break
            return (f"parking robot {rid} parks at [{end[0]},{end[1]}] from k={end_k} "
                    f"but robot {other} uses that cell later")
        return None


def build_table(robots: Iterable["Robot"], exclude: Iterable[int] = ()) -> ReservationTable:
    """Reservation table of every robot not in ``exclude`` (failed robots are parked)."""
    ex = set(exclude)
    table = ReservationTable()
    for r in robots:
        if r.id in ex:
            continue
        table.add_path(r.id, r.pos, [] if r.state == "failed" else r.path)
    return table


def plan_path(
    grid: "GridMap",
    start: Cell,
    goal: Cell,
    horizon: int,
    table: ReservationTable,
    wait_until: int = 0,
    max_expansions: int = 40_000,
    stop_at_goal: bool = True,
) -> Optional[List[Cell]]:
    """Cooperative space-time A* for one robot.

    Returns a list of cells for k = 1..L (L <= horizon) that

    * never enters a blocked cell or a cell reserved in ``table`` (vertex),
    * never swaps with a reserved robot (edge),
    * waits in place for the first ``wait_until`` steps (robot delay),
    * ends in a cell that is safe to park in (nobody reserved uses it later).

    The search stops early when the goal is reached (if ``stop_at_goal``),
    otherwise at ``horizon`` in the reachable cell closest to the goal
    (rolling-horizon / windowed planning). Returns ``None`` if no safe plan
    exists within the expansion budget.
    """
    dist = grid.distance_map(goal)

    def h(c: Cell) -> int:
        d = dist.get(c)
        return d if d is not None else 10_000 + manhattan_distance(c, goal)

    horizon = max(1, int(horizon))
    start_state = (start, 0)
    parent: Dict[Tuple[Cell, int], Tuple[Cell, int]] = {}
    g_seen: Set[Tuple[Cell, int]] = set()
    counter = 0
    open_heap = [(h(start), h(start), counter, start, 0)]
    expansions = 0

    def reconstruct(state: Tuple[Cell, int]) -> List[Cell]:
        out = []
        while state != start_state:
            out.append(state[0])
            state = parent[state]
        out.reverse()
        return out

    while open_heap:
        _f, hv, _c, cell, k = heapq.heappop(open_heap)
        state = (cell, k)
        if state in g_seen:
            continue
        g_seen.add(state)
        expansions += 1
        if expansions > max_expansions:
            break

        if k >= wait_until and table.safe_to_park(cell, k + 1):
            if stop_at_goal and cell == goal:
                return reconstruct(state)
            if k == horizon:
                return reconstruct(state)
        if k == horizon:
            continue

        nk = k + 1
        if k < wait_until:
            candidates = [cell]
        else:
            candidates = grid.neighbors(cell) + [cell]
        for nxt in candidates:
            if not grid.is_free(nxt):
                continue
            if table.occupant(nxt, nk) is not None:
                continue
            if nxt != cell and table.swap_partner(cell, nxt, k) is not None:
                continue
            ns = (nxt, nk)
            if ns in g_seen:
                continue
            if ns not in parent:
                parent[ns] = state
            else:
                # keep the first parent found (A* pops states in f order, so
                # any later parent cannot be strictly better for the same (cell, k))
                pass
            nh = h(nxt)
            counter += 1
            heapq.heappush(open_heap, (nk + nh, nh, counter, nxt, nk))
    return None
