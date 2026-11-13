"""Disruption injection engine for lifelong MAPF simulation.

Three disruption types: blocked cells (contiguous aisle blockages, optionally
temporary), robot failures (robot becomes a permanent static obstacle) and
robot delays (robot stalls for k steps).

The schedule is generated once from a seeded RNG, so every mode faces the same
sequence of disruption *events*. With ``targeting="active"`` (default) a
blockage is placed a few steps *ahead of a scheduled robot* on its current
plan - an aisle that is actually in use.

Overlapping disruptions (``overlap``):

* ``none``         isolated disruptions only
* ``same_aisle``   a second disruption of any allowed type, 1-9 steps later, in the
                   same aisle/region as the first (sequential overlap)
* ``same_robot``   a second disruption 1-9 steps later involving the same robot
                   (or, if that robot has failed, the nearest robot to it)
* ``simultaneous`` a second disruption of any allowed type in the same region at
                   the SAME step
* ``cascade``      three disruptions of different types in one region within a
                   few steps (compound overlap)
* ``mixed``        each event randomly becomes isolated / same_aisle / same_robot /
                   simultaneous / cascade - one run covers every kind

Follow-up disruptions are *anchored* to the first one and resolved at fire
time: an anchored blockage is placed on the plan of a robot moving through the
anchor's region; an anchored failure/delay hits the robot nearest to the anchor.
This guarantees the disruptions actually overlap in space and time instead of
landing on cells nobody uses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from amr_repair.sim.lifelong import World

Cell = Tuple[int, int]
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class DisruptionItem:
    type: Literal["blocked_cell", "robot_failure", "robot_delay"]
    t: int
    cells: List[Cell] = field(default_factory=list)
    robot: Optional[int] = None
    k_steps: Optional[int] = None
    duration: Optional[int] = None      # temporary blockage: unblocked at t + duration
    # Placement resolved at fire time (generated schedules only)
    target_robot: Optional[int] = None  # block `ahead` steps ahead on this robot's plan
    ahead: int = 0
    anchor: Optional[int] = None        # index of an earlier disruption to overlap with
    offset: Tuple[int, int] = (0, 0)
    block_size: int = 1
    direction: Tuple[int, int] = (1, 0)
    select: Optional[str] = None        # "busy": robot whose cell is on most other plans
    group: Optional[int] = None         # overlap group id (disruptions generated together)
    overlap_kind: str = "isolated"      # isolated | same_aisle | same_robot | simultaneous | cascade


Disruption = DisruptionItem


class Injector:
    def __init__(self, rng: np.random.Generator,
                 schedule: Optional[Union[List[Union[Dict[str, Any], DisruptionItem]], Dict[str, Any]]] = None,
                 world: Optional[World] = None):
        self.rng: np.random.Generator = rng
        self.raw_schedule = schedule
        self.disruptions: List[DisruptionItem] = []
        self._unblock_schedule: Dict[int, List[Cell]] = {}
        self._applied_cells: Dict[int, List[Cell]] = {}  # disruption index -> cells actually blocked

        if schedule is None:
            if world is not None:
                self._generate_schedule({"rate": 0.05, "types": ["blocked_cell", "robot_failure", "robot_delay"],
                                         "overlap": "same_aisle", "max_steps": 300, "max_block_size": 3}, world)
        elif isinstance(schedule, list):
            for item in schedule:
                if isinstance(item, DisruptionItem):
                    self.disruptions.append(item)
                elif isinstance(item, dict):
                    kw = dict(item)
                    kw["cells"] = [tuple(c) for c in kw.get("cells", [])]
                    for key in ("offset", "direction"):
                        if key in kw:
                            kw[key] = tuple(kw[key])
                    self.disruptions.append(DisruptionItem(**kw))
        elif isinstance(schedule, dict) and world is not None:
            self._generate_schedule(schedule, world)
        # stable sort keeps anchors pointing at earlier items
        order = sorted(range(len(self.disruptions)), key=lambda i: self.disruptions[i].t)
        remap = {old: new for new, old in enumerate(order)}
        self.disruptions = [self.disruptions[i] for i in order]
        for d in self.disruptions:
            if d.anchor is not None:
                d.anchor = remap[d.anchor]

    # ------------------------------------------------------------ generation
    def _generate_schedule(self, config: Dict[str, Any], world: World) -> None:
        rate = config.get("rate", 0.05)
        allowed = list(config.get("types", ["blocked_cell", "robot_failure", "robot_delay"]))
        overlap = config.get("overlap", "none")
        max_steps = config.get("max_steps", 200)
        max_block = config.get("max_block_size", 3)
        targeting = config.get("targeting", "active")
        kinds = ["none", "same_aisle", "same_robot", "simultaneous", "cascade"]
        if overlap not in kinds + ["mixed"]:
            raise ValueError(f"unknown overlap mode {overlap!r}; choose from {kinds + ['mixed']}")

        group = 0
        t = 10
        while t < max_steps:
            if self.rng.random() >= rate:
                t += 1
                continue
            kind = overlap if overlap != "mixed" else kinds[int(self.rng.integers(0, len(kinds)))]
            group += 1
            dtype = str(self.rng.choice(allowed))
            first = self._create_disruption(dtype, t, world, world.free_cells, max_block, targeting)
            first.group = group
            first.overlap_kind = "isolated" if kind == "none" else kind
            self.disruptions.append(first)
            first_idx = len(self.disruptions) - 1

            if kind in ("same_aisle", "simultaneous"):
                dt = 0 if kind == "simultaneous" else int(self.rng.integers(1, 10))
                second_type = str(self.rng.choice(allowed))
                self.disruptions.append(self._anchored(second_type, t + dt, first_idx, group, kind, max_block))
            elif kind == "same_robot":
                dt = int(self.rng.integers(1, 10))
                second_type = str(self.rng.choice(allowed))
                d2 = self._anchored(second_type, t + dt, first_idx, group, kind, max_block)
                rid = first.robot if first.robot is not None else first.target_robot
                if second_type == "blocked_cell":
                    d2.target_robot, d2.anchor = rid, None  # block the aisle ahead of the same robot
                    if d2.target_robot is None:
                        d2.anchor = first_idx
                else:
                    d2.robot = rid  # same robot (nearest one if it has failed meanwhile)
                self.disruptions.append(d2)
            elif kind == "cascade":
                others = [x for x in allowed if x != dtype]
                self.rng.shuffle(others)
                extra_types = (others + [dtype, dtype])[:2]  # all three types when allowed
                t_next = t
                for extra_type in extra_types:
                    t_next += int(self.rng.integers(0, 5))
                    self.disruptions.append(self._anchored(str(extra_type), t_next, first_idx, group, kind, max_block))
            t += 15

    def _anchored(self, dtype: str, t: int, anchor: int, group: int, kind: str, max_block: int) -> DisruptionItem:
        """A follow-up disruption that overlaps (in space) with disruption ``anchor``."""
        d = DisruptionItem(type=dtype, t=t, anchor=anchor, group=group, overlap_kind=kind,  # type: ignore[arg-type]
                           offset=(int(self.rng.integers(-4, 5)), int(self.rng.integers(-2, 3))))
        if dtype == "blocked_cell":
            d.block_size = int(self.rng.integers(1, max(2, max_block + 1)))
            d.direction = _DIRS[int(self.rng.integers(0, 4))]
            d.duration = int(self.rng.integers(15, 45))
        elif dtype == "robot_delay":
            d.k_steps = int(self.rng.integers(3, 8))
        return d

    def _create_disruption(self, dtype: str, t: int, world: World, free_cells: List[Cell],
                           max_block_size: int = 3, targeting: str = "active") -> DisruptionItem:
        if dtype == "blocked_cell":
            duration = int(self.rng.integers(15, 45)) if self.rng.random() < 0.5 else None
            block_size = int(self.rng.integers(1, max(2, max_block_size + 1)))
            direction = _DIRS[int(self.rng.integers(0, 4))]
            if targeting == "active":
                return DisruptionItem(type="blocked_cell", t=t, duration=duration,
                                      target_robot=int(self.rng.integers(0, len(world.robots))),
                                      ahead=int(self.rng.integers(2, 9)), block_size=block_size,
                                      direction=direction)
            seed_cell = free_cells[int(self.rng.integers(0, len(free_cells)))]
            return DisruptionItem(type="blocked_cell", t=t, duration=duration,
                                  cells=self._extend(world, seed_cell, block_size, direction))
        robot = world.robots[int(self.rng.integers(0, len(world.robots)))]
        if dtype == "robot_failure":
            return DisruptionItem(type="robot_failure", t=t, robot=robot.id)
        return DisruptionItem(type="robot_delay", t=t, robot=robot.id, k_steps=int(self.rng.integers(3, 8)))

    @staticmethod
    def _extend(world: World, seed: Cell, size: int, direction: Tuple[int, int]) -> List[Cell]:
        cells = [seed]
        cur = seed
        for _ in range(size - 1):
            nxt = (cur[0] + direction[0], cur[1] + direction[1])
            if not world.map.is_free(nxt) or nxt in cells:
                break
            cells.append(nxt)
            cur = nxt
        return cells

    def _anchor_cells(self, d: DisruptionItem) -> List[Cell]:
        return self._applied_cells.get(d.anchor, []) if d.anchor is not None else []

    def _resolve_cells(self, idx: int, d: DisruptionItem, world: World) -> List[Cell]:
        """Concrete cells of a blockage at fire time."""
        if d.cells:
            return [tuple(c) for c in d.cells]
        if d.select == "busy" and d.target_robot is None and d.anchor is None:
            # block the aisle ahead of the robot whose next cells are shared by most
            # other plans; use the first cell (>= `ahead` steps out) nobody stands on
            from collections import Counter
            use = Counter(c for r in world.robots for c in r.path)
            occupied = {r.pos for r in world.robots}
            ranked = sorted((r for r in world.robots if r.state != "failed" and len(r.path) >= 2),
                            key=lambda r: (-sum(use[c] for c in r.path[:6]), r.id))
            for r in ranked:
                for c in r.path[max(d.ahead, 2) - 1:]:
                    if c not in occupied and world.map.is_free(c):
                        return self._extend(world, c, d.block_size, d.direction)
            return []
        if d.target_robot is not None and d.target_robot < len(world.robots):
            r = world.robots[d.target_robot]
            if r.state == "failed" or not r.path:
                return []
            seed = r.path[min(max(d.ahead, 1), len(r.path)) - 1]
            return self._extend(world, seed, d.block_size, d.direction)
        base = self._anchor_cells(d)
        if d.anchor is not None and base:
            ax, ay = base[0]
            occupied = {r.pos for r in world.robots}
            # Prefer a cell 2+ steps ahead on a plan that passes near the anchor,
            # so the two disruptions really overlap on someone's route.
            best = None
            for r in world.robots:
                if r.state == "failed":
                    continue
                for k, c in enumerate(r.path[1:12], start=2):
                    dist = abs(c[0] - ax) + abs(c[1] - ay)
                    if 1 <= dist <= 6 and c not in occupied and c not in base and world.map.is_free(c):
                        key = (dist, k, r.id)
                        if best is None or key < best[0]:
                            best = (key, c)
                        break
            if best is not None:
                return self._extend(world, best[1], d.block_size, d.direction)
            c = (ax + d.offset[0], ay + d.offset[1])
            if not world.map.is_free(c) or c in occupied:
                c = min([n for n in world.map.neighbors(base[0]) if n not in occupied] or [base[0]],
                        key=lambda n: abs(n[0] - c[0]) + abs(n[1] - c[1]))
            return [c]
        return []

    @staticmethod
    def _invalidated(world: World, robot, d: DisruptionItem) -> set:
        """Robots whose committed plans would become invalid if ``robot`` failed
        (its cell becomes an obstacle) or stalled for k steps."""
        from amr_repair.sim.planning import ReservationTable
        others = [r for r in world.robots if r.id != robot.id and r.state != "failed"]
        if d.type == "robot_failure":
            return {r.id for r in others if robot.pos in r.path}
        table = ReservationTable()
        table.add_path(robot.id, robot.pos, [robot.pos] * int(d.k_steps or 5) + list(robot.path))
        return {r.id for r in others if table.first_conflict(r.id, r.pos, r.path) is not None}

    def _impact(self, world: World, robot, d: DisruptionItem) -> int:
        return len(self._invalidated(world, robot, d))

    def _resolve_robot(self, d: DisruptionItem, world: World) -> Optional[int]:
        """Robot hit by a failure/delay at fire time."""
        alive = [r for r in world.robots if r.state != "failed"]
        if not alive:
            return None
        if d.robot is not None and d.robot < len(world.robots) and world.robots[d.robot].state != "failed":
            return d.robot
        if d.select == "busy":
            return max(alive, key=lambda r: (self._impact(world, r, d), -r.id)).id
        anchor = self._anchor_cells(d) or ([world.robots[d.robot].pos] if d.robot is not None
                                            and d.robot < len(world.robots) else [])
        if anchor:
            ax, ay = anchor[0]
            # An anchored failure/delay hits a *different* robot (same-robot
            # overlaps set d.robot explicitly) chosen so the two disruptions
            # INTERACT in the synopsis' sense - their affected robots or path
            # regions overlap - and the second one actually invalidates plans:
            #   1. robots within 6 cells (regions overlap) with impact > 0
            #   2. robots within 15 cells whose failure/stall invalidates the plan of a
            #      robot that is in, or routed through, the first disruption's area
            #   3. otherwise the nearest robot (realistic: may have no impact)
            cands = [r for r in alive if r.pos not in anchor]
            if not cands:
                return None

            def dist(r):
                return abs(r.pos[0] - ax) + abs(r.pos[1] - ay)

            def in_area(c):
                return abs(c[0] - ax) <= 4 and abs(c[1] - ay) <= 4

            area_robots = {r.id for r in alive if in_area(r.pos) or any(in_area(c) for c in r.path)}
            scored = []
            for r in cands:
                if dist(r) > 15:
                    continue
                inv = self._invalidated(world, r, d)
                if not inv:
                    continue
                overlap = (dist(r) <= 6) or bool(inv & area_robots)
                if overlap:
                    scored.append(((dist(r) <= 6), len(inv & area_robots), len(inv), -dist(r), -r.id, r.id))
            if scored:
                return max(scored)[-1]
            return min(cands, key=lambda r: (dist(r), r.id)).id
        return None

    def apply(self, world: World, t: int) -> List[DisruptionItem]:
        """Apply every disruption scheduled for step ``t`` to the world.

        Returns the disruptions that actually took effect, with ``cells``
        reflecting the *current* situation (e.g. a robot's position when it
        fails, not where it was when the schedule was generated).
        """
        # Temporary blockages expiring now
        if t in self._unblock_schedule:
            world.map.unblock(self._unblock_schedule.pop(t))

        fired: List[DisruptionItem] = []
        robot_cells = {r.pos for r in world.robots}
        for idx, d in enumerate(self.disruptions):
            if d.t != t:
                continue
            if d.type == "blocked_cell":
                # A blockage never appears underneath a robot.
                cells = [c for c in self._resolve_cells(idx, d, world) if c not in robot_cells]
                applied = world.map.block(cells)
                if not applied:
                    continue
                self._applied_cells[idx] = applied
                if d.duration:
                    self._unblock_schedule.setdefault(t + d.duration, []).extend(applied)
                world.reassign_blocked_goals()
                fired.append(DisruptionItem(type="blocked_cell", t=t, cells=applied, duration=d.duration,
                                            group=d.group, overlap_kind=d.overlap_kind))

            elif d.type == "robot_failure":
                rid = self._resolve_robot(d, world)
                r = world.robots[rid] if rid is not None else None
                if r is None or r.state == "failed":
                    continue
                r.state = "failed"
                r.path = []
                r.delay_remaining = 0
                world.map.block([r.pos])  # failed robot is a static obstacle
                self._applied_cells[idx] = [r.pos]
                world.reassign_blocked_goals()
                fired.append(DisruptionItem(type="robot_failure", t=t, cells=[r.pos], robot=r.id,
                                            group=d.group, overlap_kind=d.overlap_kind))

            elif d.type == "robot_delay":
                rid = self._resolve_robot(d, world)
                r = world.robots[rid] if rid is not None else None
                if r is None or r.state == "failed":
                    continue
                k = int(d.k_steps or 5)
                # The robot stalls for k steps: its plan is shifted by k waits,
                # which may now conflict with other robots' plans.
                r.path = [r.pos] * k + list(r.path)
                r.delay_remaining += k
                r.state = "delayed"
                self._applied_cells[idx] = [r.pos]
                fired.append(DisruptionItem(type="robot_delay", t=t, cells=[r.pos], robot=r.id, k_steps=k,
                                            duration=k, group=d.group, overlap_kind=d.overlap_kind))

        return fired
