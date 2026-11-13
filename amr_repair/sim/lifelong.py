"""Lifelong MAPF simulation world.

Robots follow **committed, collision-free, time-indexed plans** produced by a
rolling-horizon cooperative planner (windowed cooperative A*, Silver 2005).
Every step:

1. robots that reached their goal receive the next goal from their own task
   stream; robots without a plan are planned around everyone else's
   reservations (priority = order of planning);
2. a consistency check finds any plans that conflict with each other or with
   the map (e.g. after an emergency hold) and replans them;
3. all robots execute one step of their plan simultaneously. An execution
   guard (the robot's low-level safety stop) holds any robot whose move would
   still collide; such interventions are counted and should be zero.

Disruption *repair* (the research contribution) is done by the controller
before ``step()``; the background planner here only handles normal lifelong
operation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from amr_repair.replay.models import Frame
from amr_repair.sim.agents import Robot
from amr_repair.sim.grid import GridMap
from amr_repair.sim.planning import (
    ReservationTable, build_table, future_cell, manhattan_distance, plan_path,
)

Cell = Tuple[int, int]

__all__ = ["World", "manhattan_distance"]


class World:
    def __init__(
        self,
        grid_map: GridMap,
        robots: List[Robot],
        rng: np.random.Generator,
        planning_window: int = 20,
        goal_seed: Optional[int] = None,
    ):
        self.map: GridMap = grid_map
        self.robots: List[Robot] = robots
        self.rng: np.random.Generator = rng
        self.free_cells: List[Cell] = self.map.get_free_cells()
        self.planning_window = planning_window
        self.t: int = 0
        self.frames: List[Frame] = []

        # Per-robot task streams: robot r's k-th goal is identical across runs
        # with the same seed, independent of which planner/repair mode is used.
        if goal_seed is None:
            goal_seed = int(rng.integers(0, 2**31 - 1))
        self._goal_rngs = {r.id: np.random.default_rng([goal_seed, r.id]) for r in robots}

        # Telemetry
        self.goals_completed: int = 0
        self.background_replans: int = 0
        self.conflict_replans: int = 0
        self.guard_interventions: int = 0
        self.planning_failures: int = 0

        self._moved_last_step: Set[int] = set()
        self._plan_all([r.id for r in self.robots if not r.path])
        self._update_states()
        self._record_frame()

    # ------------------------------------------------------------------ goals
    def _assign_new_goal(self, robot: Robot) -> None:
        rng = self._goal_rngs.setdefault(robot.id, np.random.default_rng([0, robot.id]))
        for _ in range(100):
            candidate = self.free_cells[int(rng.integers(0, len(self.free_cells)))]
            if candidate != robot.pos and self.map.is_free(candidate):
                robot.goal = candidate
                return
        robot.goal = robot.pos

    def reassign_blocked_goals(self) -> List[int]:
        """Give a new goal to robots whose goal cell became an obstacle."""
        changed = []
        for r in self.robots:
            if r.state != "failed" and not self.map.is_free(r.goal):
                self._assign_new_goal(r)
                changed.append(r.id)
        return changed

    # --------------------------------------------------------------- planning
    def active_robots(self) -> List[Robot]:
        return [r for r in self.robots if r.state != "failed"]

    def plan_robot(self, robot: Robot, table: ReservationTable, horizon: Optional[int] = None) -> bool:
        path = plan_path(
            self.map, robot.pos, robot.goal, horizon or self.planning_window, table,
            wait_until=robot.delay_remaining,
        )
        if path is None:
            self.planning_failures += 1
            # Park in place; consistency check will move others around it.
            robot.path = [robot.pos] * robot.delay_remaining
            table.add_path(robot.id, robot.pos, robot.path)
            return False
        robot.path = path
        table.add_path(robot.id, robot.pos, path)
        return True

    def _plan_all(self, robot_ids: List[int], restarts: int = 3) -> None:
        """Prioritized planning of a batch of robots around everyone else.
        If a robot finds no safe plan, it is moved to the front of the order and
        the batch is replanned (standard prioritized-planning restart)."""
        order = [rid for rid in robot_ids if self.robots[rid].state != "failed"]
        if not order:
            return
        for r in (self.robots[rid] for rid in order):
            if r.pos == r.goal:
                self._assign_new_goal(r)
        for attempt in range(restarts + 1):
            table = build_table(self.robots, exclude=order)
            failed = [rid for rid in order if not self.plan_robot(self.robots[rid], table)]
            if not failed or attempt == restarts:
                return
            order = failed + [rid for rid in order if rid not in failed]

    def find_conflicting_robots(self) -> Set[int]:
        """Robots whose committed plans are invalid: they hit an obstacle or
        collide with another robot's plan (vertex, swap or parking)."""
        bad: Set[int] = set()
        table = ReservationTable()
        for r in sorted(self.robots, key=lambda x: x.id):
            if r.unvalidated_until > self.t:
                continue  # executing an unvalidated plan: left to the execution guard
            path = [] if r.state == "failed" else r.path
            if any(not self.map.is_free(c) for c in path):
                bad.add(r.id)
            conflict = table.first_conflict(r.id, r.pos, path)
            if conflict is not None:
                bad.add(r.id)
                # also mark the robot it collides with
                for tok in conflict.replace(",", " ").split():
                    if tok.isdigit() and int(tok) in table.robots:
                        bad.add(int(tok))
            table.add_path(r.id, r.pos, path)
        return {rid for rid in bad if self.robots[rid].state != "failed"}

    def _enforce_consistency(self) -> None:
        bad = self.find_conflicting_robots()
        if not bad:
            return
        # Prefer to keep validated repair plans; replan ordinary robots first.
        repairing = {rid for rid in bad if self.robots[rid].repair_until > self.t}
        to_replan = sorted(bad - repairing) or sorted(bad)
        self.conflict_replans += len(to_replan)
        for rid in to_replan:
            self.robots[rid].path = []
        self._plan_all(to_replan)
        # One more pass in case parking in place created new conflicts.
        bad = self.find_conflicting_robots()
        if bad:
            for rid in bad:
                self.robots[rid].path = []
            self.conflict_replans += len(bad)
            self._plan_all(sorted(bad))

    # -------------------------------------------------------------- execution
    def step(self) -> None:
        # 1. lifelong goal assignment + planning for robots without a plan
        needs_plan = []
        for r in self.robots:
            if r.state == "failed":
                continue
            if r.pos == r.goal and not r.path:
                self._assign_new_goal(r)
            if not r.path:
                needs_plan.append(r.id)
        self.background_replans += len(needs_plan)
        self._plan_all(needs_plan)

        # 2. make sure the joint plan is collision-free before executing it
        self._enforce_consistency()

        # 3. execute one step with the safety guard
        desired: Dict[int, Cell] = {}
        for r in self.robots:
            if r.state == "failed" or r.delay_remaining > 0 or not r.path:
                desired[r.id] = r.pos
            else:
                desired[r.id] = r.path[0]
        held: Set[int] = set()
        changed = True
        while changed:
            changed = False
            occupied: Dict[Cell, int] = {}
            for r in self.robots:
                c = desired[r.id]
                if c != r.pos and not self.map.is_free(c):
                    desired[r.id] = r.pos
                    held.add(r.id)
                    changed = True
            for r in self.robots:
                c = desired[r.id]
                if c in occupied:
                    other = occupied[c]
                    # hold whichever of the two is actually moving
                    loser = r.id if desired[r.id] != r.pos else other
                    if desired[loser] != self.robots[loser].pos:
                        desired[loser] = self.robots[loser].pos
                        held.add(loser)
                        changed = True
                        break
                occupied[c] = r.id
            if changed:
                continue
            pos_to_robot = {r.pos: r.id for r in self.robots}
            for r in self.robots:
                c = desired[r.id]
                if c == r.pos:
                    continue
                other = pos_to_robot.get(c)
                if other is not None and other != r.id and desired[other] == r.pos:
                    desired[r.id] = r.pos
                    desired[other] = self.robots[other].pos
                    held.update((r.id, other))
                    changed = True
                    break
        self.guard_interventions += len(held)

        moved: Set[int] = set()
        for r in self.robots:
            if r.state == "failed":
                continue
            if r.delay_remaining > 0:
                r.delay_remaining -= 1
                if r.path and r.path[0] == r.pos:
                    r.path.pop(0)
                continue
            if r.id in held:
                continue  # plan not consumed; consistency check fixes timing next step
            if r.path:
                nxt = r.path.pop(0)
                if nxt != r.pos:
                    moved.add(r.id)
                r.pos = nxt

        self.t += 1
        self._moved_last_step = moved

        # 4. goal completion (lifelong: next goal is given on arrival)
        for r in self.robots:
            if r.state != "failed" and r.pos == r.goal:
                r.goals_completed += 1
                self.goals_completed += 1
                self._assign_new_goal(r)
                if r.repair_until <= self.t:
                    r.path = []  # replan towards the new goal next step

        self._update_states()
        self._record_frame()

    def _update_states(self) -> None:
        for r in self.robots:
            if r.state == "failed":
                continue
            if r.delay_remaining > 0:
                r.state = "delayed"
            elif r.repair_until >= self.t:
                r.state = "replanning"
            elif r.id in self._moved_last_step or (r.path and r.path[0] != r.pos):
                r.state = "moving"
            else:
                r.state = "idle"

    def _record_frame(self) -> None:
        self.frames.append(Frame(
            t=self.t,
            pos=[r.pos for r in self.robots],
            state=[r.state for r in self.robots],
            goals=[r.goal for r in self.robots],
        ))

    # ------------------------------------------------------------ utilities
    def robot_cell_at(self, robot: Robot, k: int) -> Cell:
        return future_cell(robot.pos, [] if robot.state == "failed" else robot.path, k)

    def to_replay(self, seed: int = 42, mode: str = "baseline_pibt",
                  map_name: str = "warehouse-10-20-10-2-1") -> Dict[str, Any]:
        return {
            "meta": {"seed": seed, "mode": mode, "map_name": map_name, "steps": len(self.frames)},
            "map": {"width": self.map.width, "height": self.map.height,
                    "obstacles": self.map.get_obstacle_cells()},
            "frames": [f.model_dump() for f in self.frames],
            "incidents": [],
            "metrics": {},
        }
