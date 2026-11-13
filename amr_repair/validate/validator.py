"""Deterministic plan validator — the single source of truth for "valid plan".

A candidate repair is a dict ``{robot_id: path}`` where ``path`` lists the
cells for k = 1, 2, ... (see :mod:`amr_repair.sim.planning`). It is checked
**together with the committed plans of every other robot in the fleet**.

Checks, in order (the first violation is returned, with absolute time t):

1. structure      - unknown robot, failed robot, empty path
2. continuity     - every step is a 4-neighbour move or a wait ("teleport")
3. failed_robot   - entering a failed robot's cell
4. blocked_cell   - entering a static/temporary obstacle
5. delay          - a delayed robot moves before its delay has elapsed
6. vertex / edge  - collisions among the candidate robots and against every
                    other robot's committed plan (incl. parked robots)
7. parking        - a robot's final cell is used by someone after it parks
8. unrepaired     - a ``required`` robot that was not replanned still has an
                    invalid plan (e.g. through the blocked aisle)
9. deadlock       - no robot in the set moves for 10 consecutive steps while
                    none of them is at its goal or delayed
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from amr_repair.sim.lifelong import World
from amr_repair.sim.planning import ReservationTable, build_table, future_cell

Cell = Tuple[int, int]
DEADLOCK_WINDOW = 10


@dataclass
class Verdict:
    valid: bool
    violation: Optional[str] = None


def _fmt(c: Cell) -> str:
    return f"[{c[0]},{c[1]}]"


def _absolute_time(msg: str, t0: int) -> str:
    """Convert relative 'k=' markers from the planning module to absolute 't='."""
    out = []
    for tok in msg.split(" "):
        if tok.startswith("k=") and tok[2:].isdigit():
            out.append(f"t={t0 + int(tok[2:])}")
        else:
            out.append(tok)
    return " ".join(out)


def validate(
    paths: Dict[int, List[Cell]],
    world: World,
    t0: Optional[int] = None,
    required: Optional[Iterable[int]] = None,
) -> Verdict:
    t0 = world.t if t0 is None else t0
    robots = {r.id: r for r in world.robots}
    required_set = set(required or ())

    combined: Dict[int, List[Cell]] = {}
    for rid, p in paths.items():
        if rid not in robots:
            return Verdict(False, f"unknown robot {rid}")
        if robots[rid].state == "failed":
            return Verdict(False, f"failed_robot robot {rid} is failed and cannot be replanned")
        if not p:
            return Verdict(False, f"empty path for robot {rid}")
        combined[rid] = list(p)
    unrepaired = sorted(
        r for r in required_set - set(combined) if r in robots and robots[r].state != "failed"
    )
    for rid in unrepaired:
        combined[rid] = list(robots[rid].path)

    if not combined:
        return Verdict(True)

    failed_cells = {r.pos for r in world.robots if r.state == "failed"}

    # 2-5: per-robot checks
    for rid in sorted(combined):
        r = robots[rid]
        tag = "unrepaired robot" if rid in unrepaired else "robot"
        prev = r.pos
        for k, c in enumerate(combined[rid], start=1):
            t = t0 + k
            if abs(c[0] - prev[0]) + abs(c[1] - prev[1]) > 1:
                return Verdict(False, f"teleport {tag} {rid} {_fmt(prev)}->{_fmt(c)} t={t}")
            if c in failed_cells:
                return Verdict(False, f"failed_robot {tag} {rid} cell {_fmt(c)} t={t}")
            if not world.map.is_free(c):
                return Verdict(False, f"blocked_cell {tag} {rid} cell {_fmt(c)} t={t}")
            if k <= r.delay_remaining and c != r.pos:
                return Verdict(False, f"delay robot {rid} must wait {r.delay_remaining} steps but moves at t={t}")
            prev = c

    # 6-7: collisions/parking among the set and against the rest of the fleet
    others = build_table(world.robots, exclude=combined.keys())
    table = ReservationTable()
    for rid in sorted(combined):
        r = robots[rid]
        for tbl in (others, table):
            conflict = tbl.first_conflict(rid, r.pos, combined[rid])
            if conflict is not None:
                suffix = " (robot was not replanned)" if rid in unrepaired else ""
                return Verdict(False, _absolute_time(conflict, t0) + suffix)
        table.add_path(rid, r.pos, combined[rid])

    # 9: deadlock
    max_len = max(len(p) for p in combined.values())
    if max_len >= DEADLOCK_WINDOW:
        for s in range(0, max_len - DEADLOCK_WINDOW + 1):
            stuck = True
            for rid, p in combined.items():
                r = robots[rid]
                window = [future_cell(r.pos, p, k) for k in range(s, s + DEADLOCK_WINDOW + 1)]
                if len(set(window)) > 1 or window[-1] == r.goal or s < r.delay_remaining:
                    stuck = False
                    break
            if stuck:
                return Verdict(False, f"deadlock robots {sorted(combined)} make no move during "
                                      f"t={t0 + s}..{t0 + s + DEADLOCK_WINDOW}")

    return Verdict(True)
