"""Sentinel: turns fired disruptions into IncidentRecords.

Affected robots are found from the robots' committed space-time plans:

* blocked_cell  - robots whose plan enters a blocked cell (must be replanned)
* robot_failure - robots whose plan enters the failed robot's cell (must be
                  replanned); the failed robot itself is recorded but can never
                  be replanned
* robot_delay   - the delayed robot plus every robot whose plan now conflicts
                  (vertex / swap / parking) with the delayed robot's shifted plan

Two disruptions *interact* if their affected robots or regions overlap. This is
checked both among disruptions that fire in the same step (they are merged
into one incident and repaired jointly) and against incidents that are still
being recovered from (robots still executing an earlier repair plan); in the
latter case those robots are added to the candidate set so they can be
repaired together.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from amr_repair.disrupt.injector import DisruptionItem
from amr_repair.sentinel.incident import (
    CongestionData, CongestionLevel, DisruptionData, IncidentRecord, RelatedDisruption, RobotContext, SizeBucket,
)
from amr_repair.sim.lifelong import World
from amr_repair.sim.planning import ReservationTable

Region = Tuple[Tuple[int, int], Tuple[int, int]]
MAX_NEARBY = 4


def _level(density: float) -> CongestionLevel:
    return "low" if density < 0.15 else ("medium" if density <= 0.35 else "high")


def _size(count: int) -> SizeBucket:
    return "small" if count <= 2 else ("medium" if count <= 6 else "large")


def _regions_intersect(a: Region, b: Region) -> bool:
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)


class IncidentDetector:
    def __init__(self, interaction_window: int = 30):
        self._incident_counter = 0
        self.interaction_window = interaction_window
        # Active disruptions/incidents that later disruptions may overlap with:
        # id -> {"t", "robots", "region", "expires", "disruptions"}
        self._active: Dict[str, dict] = {}

    # ------------------------------------------------------------- public API
    def register_repair(self, incident: IncidentRecord, robots: List[int], until_t: int) -> None:
        """Extend an incident's active window while its repair plan executes."""
        entry = self._active.setdefault(incident.id, {
            "t": incident.t_detected, "robots": set(), "region": incident.congestion.region,
            "expires": incident.t_detected, "disruptions": list(incident.disruptions)})
        entry["robots"] |= set(robots) | set(incident.affected_robots)
        entry["expires"] = max(entry["expires"], until_t)

    def active_ids(self, t: int) -> List[str]:
        return sorted(k for k, v in self._active.items() if v["expires"] > t)

    def detect(self, world: World, fired: List[DisruptionItem], t: int) -> List[IncidentRecord]:
        if not fired:
            return []
        raw = [self._build_record(world, d, t) for d in fired]
        expiry = {rec.id: t + self._expiry(d) for rec, d in zip(raw, fired)}
        merged = self._merge_same_step(world, raw)
        self._attach_residual_conflicts(world, merged)
        self._link_active(world, merged, t)
        for inc in merged:
            self._finalise(world, inc)
        # Register every disruption of this step (even those without immediate
        # impact: a blocked aisle keeps constraining the area) for later overlaps.
        for inc in merged:
            exp = max(expiry.get(i, t) for i in ([inc.id] + inc.interacts_with) if i in expiry)
            self._active[inc.id] = {"t": t, "robots": set(inc.affected_robots), "region": inc.congestion.region,
                                    "expires": exp, "disruptions": list(inc.disruptions)}
        return merged

    def _expiry(self, d: DisruptionItem) -> int:
        if d.type == "robot_delay":
            return int(d.k_steps or 5) + 10
        if d.type == "blocked_cell" and d.duration:
            return min(int(d.duration), self.interaction_window)
        return self.interaction_window

    # ---------------------------------------------------------------- helpers
    def _next_id(self) -> str:
        self._incident_counter += 1
        return f"inc_{self._incident_counter:03d}"

    def _region_and_density(self, world: World, cells: List[Tuple[int, int]]) -> Tuple[Region, float]:
        if cells:
            x0 = max(0, min(c[0] for c in cells) - 3)
            x1 = min(world.map.width - 1, max(c[0] for c in cells) + 3)
            y0 = max(0, min(c[1] for c in cells) - 3)
            y1 = min(world.map.height - 1, max(c[1] for c in cells) + 3)
        else:
            x0, y0, x1, y1 = 0, 0, 6, 6
        free = int((~world.map.blocked[y0:y1 + 1, x0:x1 + 1]).sum())
        n_rob = sum(1 for r in world.robots if x0 <= r.pos[0] <= x1 and y0 <= r.pos[1] <= y1)
        density = float(np.clip(n_rob / max(1, free), 0.0, 1.0))
        return ((x0, y0), (x1, y1)), density

    def _build_record(self, world: World, d: DisruptionItem, t: int) -> IncidentRecord:
        cells = [tuple(c) for c in d.cells]
        robots = {r.id: r for r in world.robots}
        if not cells and d.robot is not None and d.robot in robots:
            cells = [robots[d.robot].pos]
        affected: Set[int] = set()
        must: Set[int] = set()
        failed: List[int] = []
        delayed: Dict[int, int] = {}

        if d.type in ("blocked_cell", "robot_failure"):
            cell_set = set(cells)
            for r in world.robots:
                if r.state == "failed" or r.id == d.robot:
                    continue
                if any(c in cell_set for c in r.path):
                    affected.add(r.id)
                    must.add(r.id)
            if d.type == "robot_failure" and d.robot is not None:
                affected.add(d.robot)
                failed.append(d.robot)
        else:  # robot_delay
            dr = robots.get(d.robot)
            if dr is not None and dr.state != "failed":
                delayed[dr.id] = dr.delay_remaining
                table = ReservationTable()
                table.add_path(dr.id, dr.pos, dr.path)
                for r in world.robots:
                    if r.id == dr.id or r.state == "failed":
                        continue
                    if table.first_conflict(r.id, r.pos, r.path) is not None:
                        affected.add(r.id)
                if affected:  # the delay only needs repair if it creates conflicts
                    affected.add(dr.id)

        region, density = self._region_and_density(world, cells)
        return IncidentRecord(
            id=self._next_id(),
            t_detected=t,
            disruptions=[DisruptionData(type=d.type, cells=cells, robot=d.robot, k_steps=d.k_steps,
                                        duration=d.duration)],
            affected_robots=sorted(affected),
            size_bucket=_size(max(len(cells), len(affected))),
            congestion=CongestionData(level=_level(density), region=region, density=density),
            must_replan=sorted(must),
            failed_robots=failed,
            delayed_robots=delayed,
        )

    def _merge_same_step(self, world: World, incidents: List[IncidentRecord]) -> List[IncidentRecord]:
        if len(incidents) <= 1:
            return incidents
        merged: List[IncidentRecord] = []
        used: Set[int] = set()
        for i, cur in enumerate(incidents):
            if i in used:
                continue
            ids = [cur.id]
            changed = True
            while changed:  # transitive closure of interactions
                changed = False
                for j in range(len(incidents)):
                    if j == i or j in used:
                        continue
                    other = incidents[j]
                    if (set(cur.affected_robots) & set(other.affected_robots)
                            or _regions_intersect(cur.congestion.region, other.congestion.region)):
                        used.add(j)
                        ids.append(other.id)
                        (ax0, ay0), (ax1, ay1) = cur.congestion.region
                        (bx0, by0), (bx1, by1) = other.congestion.region
                        density = (cur.congestion.density + other.congestion.density) / 2.0
                        aff = sorted(set(cur.affected_robots) | set(other.affected_robots))
                        dis = cur.disruptions + other.disruptions
                        cur = IncidentRecord(
                            id=cur.id, t_detected=cur.t_detected, disruptions=dis, affected_robots=aff,
                            size_bucket=_size(max(len(dis), len(aff))),
                            congestion=CongestionData(
                                level=_level(density),
                                region=((min(ax0, bx0), min(ay0, by0)), (max(ax1, bx1), max(ay1, by1))),
                                density=density),
                            must_replan=sorted(set(cur.must_replan) | set(other.must_replan)),
                            failed_robots=sorted(set(cur.failed_robots) | set(other.failed_robots)),
                            delayed_robots={**cur.delayed_robots, **other.delayed_robots},
                        )
                        changed = True
            used.add(i)
            if len(ids) > 1:
                cur.interacts_with = sorted(set(ids))
            merged.append(cur)
        return merged

    def _attach_residual_conflicts(self, world: World, incidents: List[IncidentRecord]) -> None:
        """Conflicts caused by the *combination* of disruptions (not attributable
        to a single one) are attached to the nearest incident of this step."""
        if not incidents:
            return
        attributed = set().union(*(set(i.affected_robots) for i in incidents))
        residual = world.find_conflicting_robots() - attributed
        for rid in sorted(residual):
            pos = world.robots[rid].pos

            def dist(inc: IncidentRecord) -> int:
                (x0, y0), (x1, y1) = inc.congestion.region
                return max(0, x0 - pos[0], pos[0] - x1) + max(0, y0 - pos[1], pos[1] - y1)

            target = min(incidents, key=dist)
            if dist(target) <= 10:
                target.affected_robots = sorted(set(target.affected_robots) | {rid})

    def _link_active(self, world: World, incidents: List[IncidentRecord], t: int) -> None:
        """Sequential overlap: link incidents to earlier disruptions that are still active."""
        self._active = {k: v for k, v in self._active.items() if v["expires"] > t}
        for inc in incidents:
            simultaneous = len(inc.disruptions) > 1
            linked = False
            for aid, entry in sorted(self._active.items()):
                if entry["t"] >= t:
                    continue  # same-step disruptions are handled by the merge
                if not (entry["robots"] & set(inc.affected_robots)
                        or _regions_intersect(entry["region"], inc.congestion.region)
                        or self._paths_cross(world, inc.plannable_affected(), entry["region"])):
                    continue
                linked = True
                inc.interacts_with = sorted(set(inc.interacts_with) | {aid, inc.id})
                inc.related_disruptions.extend(
                    RelatedDisruption(incident_id=aid, t=entry["t"], type=d.type, cells=d.cells, robot=d.robot)
                    for d in entry["disruptions"])
                if not inc.plannable_affected():
                    continue  # no own impact: nothing to repair jointly
                still_repairing = {
                    r for r in entry["robots"]
                    if r < len(world.robots) and world.robots[r].state != "failed"
                    and world.robots[r].repair_until > t
                }
                inc.affected_robots = sorted(set(inc.affected_robots) | still_repairing)
            n_total = len(inc.disruptions) + len(inc.related_disruptions)
            if simultaneous and linked or n_total >= 3:
                inc.overlap_kind = "compound"
            elif simultaneous:
                inc.overlap_kind = "simultaneous"
            elif linked:
                inc.overlap_kind = "sequential"
            else:
                inc.overlap_kind = "isolated"

    @staticmethod
    def _paths_cross(world: World, robots: List[int], region: Region) -> bool:
        """True if any of ``robots`` is in, or plans to pass through, ``region``."""
        (x0, y0), (x1, y1) = region
        for rid in robots:
            r = world.robots[rid]
            for c in [r.pos] + r.path:
                if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
                    return True
        return False

    def _finalise(self, world: World, inc: IncidentRecord) -> None:
        robots = {r.id: r for r in world.robots}
        inc.failed_robots = sorted(set(inc.failed_robots) | {
            r for r in inc.affected_robots if robots[r].state == "failed"})
        inc.delayed_robots = {r: robots[r].delay_remaining for r in inc.affected_robots
                              if robots[r].delay_remaining > 0 and robots[r].state != "failed"}
        plannable = inc.plannable_affected()
        (x0, y0), (x1, y1) = inc.congestion.region
        nearby = [r for r in world.robots
                  if r.state != "failed" and r.id not in plannable
                  and x0 <= r.pos[0] <= x1 and y0 <= r.pos[1] <= y1]
        anchor = inc.disruptions[0].cells[0] if inc.disruptions and inc.disruptions[0].cells else (x0, y0)
        nearby.sort(key=lambda r: abs(r.pos[0] - anchor[0]) + abs(r.pos[1] - anchor[1]))
        inc.candidate_robots = sorted(set(plannable) | {r.id for r in nearby[:MAX_NEARBY]})
        inc.size_bucket = _size(max(sum(len(d.cells) for d in inc.disruptions), len(inc.affected_robots)))

        interacting = set()
        for aid in inc.interacts_with:
            if aid in self._active:
                interacting |= self._active[aid]["robots"]
        ctx = []
        for rid in inc.candidate_robots:
            r = robots[rid]
            if rid in inc.must_replan:
                role = "must_replan"
            elif rid in inc.delayed_robots:
                role = "delayed"
            elif rid in interacting and rid in plannable:
                role = "interacting"
            elif rid in plannable:
                role = "conflicting"
            else:
                role = "nearby"
            d = world.map.distance(r.pos, r.goal)
            ctx.append(RobotContext(id=rid, role=role, dist_to_goal=min(d, 9999),
                                    delay_remaining=r.delay_remaining))
        inc.robot_context = ctx
