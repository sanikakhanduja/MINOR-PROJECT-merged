"""Overlapping (simultaneous / sequential / compound) disruptions.

Synopsis: "Each is tested alone and in controlled overlapping combinations" and
"Particular focus is given to simultaneous and interacting disruptions".
These tests make sure such scenarios are generated, detected as interacting,
labelled with their overlap kind, passed to the strategist, repaired without
collisions, and reported separately from isolated incidents.
"""

import json
from collections import Counter

import numpy as np
import pytest

from amr_repair.disrupt.injector import DisruptionItem, Injector
from amr_repair.experiments.scenarios import build_scenarios, run_scenario
from amr_repair.llm.prompts import format_user_prompt
from amr_repair.loop.controller import SimulationConfig, SimulationController, run
from amr_repair.sentinel.detect import IncidentDetector
from amr_repair.sim.agents import Robot
from amr_repair.sim.grid import GridMap
from amr_repair.sim.lifelong import World

MAP = "maps/warehouse-10-20-10-2-1.map"
TYPES = {"blocked_cell", "robot_failure", "robot_delay"}


@pytest.fixture
def world():
    gmap = GridMap.from_movingai(MAP)
    rng = np.random.default_rng(4)
    free = gmap.get_free_cells()
    idx = rng.choice(len(free), 120, replace=False)
    w = World(gmap, [Robot(id=i, pos=free[j], goal=free[j]) for i, j in enumerate(idx)], rng, goal_seed=4)
    for _ in range(5):
        w.step()
    return w


def _gen(world, overlap, rate=0.3, steps=200):
    cfg = {"rate": rate, "types": sorted(TYPES), "overlap": overlap, "max_steps": steps}
    return Injector(np.random.default_rng(11), cfg, world).disruptions


# ------------------------------------------------------------------ generator
def test_simultaneous_mode_fires_pairs_in_the_same_step(world):
    ds = _gen(world, "simultaneous")
    groups = {}
    for d in ds:
        groups.setdefault(d.group, []).append(d)
    assert groups and all(len(g) == 2 for g in groups.values())
    assert all(g[0].t == g[1].t and g[1].anchor is not None for g in groups.values())
    assert len({g[1].type for g in groups.values()}) >= 2  # follow-ups of different types


def test_cascade_mode_combines_all_three_types(world):
    ds = _gen(world, "cascade")
    groups = {}
    for d in ds:
        groups.setdefault(d.group, []).append(d)
    assert groups
    for g in groups.values():
        assert {d.type for d in g} == TYPES
        assert g[-1].t - g[0].t <= 8


def test_same_aisle_and_same_robot_overlaps(world):
    aisle = _gen(world, "same_aisle")
    assert any(d.anchor is not None and d.type != "blocked_cell" for d in aisle)  # any type, not only blocks
    robot = _gen(world, "same_robot")
    by_group = {}
    for d in robot:
        by_group.setdefault(d.group, []).append(d)
    shared = [g for g in by_group.values() if len(g) == 2 and
              ({g[0].robot, g[0].target_robot} - {None}) & ({g[1].robot, g[1].target_robot} - {None})]
    assert shared


def test_mixed_mode_contains_every_overlap_kind(world):
    kinds = Counter(d.overlap_kind for d in _gen(world, "mixed", rate=0.5, steps=600) if d.anchor is None)
    assert set(kinds) >= {"isolated", "same_aisle", "same_robot", "simultaneous", "cascade"}


def test_anchored_failure_hits_a_different_robot_in_the_same_area(world):
    r = next(r for r in world.robots if len(r.path) >= 4)
    inj = Injector(np.random.default_rng(0), [
        DisruptionItem(type="blocked_cell", t=5, cells=[r.path[2]]),
        DisruptionItem(type="robot_failure", t=5, anchor=0),
    ])
    fired = inj.apply(world, 5)
    assert [d.type for d in fired] == ["blocked_cell", "robot_failure"]
    block, fail = fired
    assert fail.robot is not None and fail.cells[0] not in block.cells
    assert abs(fail.cells[0][0] - block.cells[0][0]) + abs(fail.cells[0][1] - block.cells[0][1]) <= 15


# ------------------------------------------------------------------- sentinel
def test_simultaneous_disruptions_merge_into_one_simultaneous_incident(world):
    r = next(r for r in world.robots if len(r.path) >= 6)
    inj = Injector(np.random.default_rng(0), [
        DisruptionItem(type="blocked_cell", t=5, cells=[r.path[3]]),
        DisruptionItem(type="robot_delay", t=5, anchor=0, k_steps=6),
    ])
    fired = inj.apply(world, 5)
    incidents = IncidentDetector().detect(world, fired, 5)
    merged = [i for i in incidents if len(i.disruptions) == 2]
    assert merged and merged[0].overlap_kind == "simultaneous"
    assert {d.type for d in merged[0].disruptions} == {"blocked_cell", "robot_delay"}
    assert merged[0].interacts_with


def test_sequential_disruption_is_linked_to_the_earlier_active_one(world):
    det = IncidentDetector()
    r = next(r for r in world.robots if len(r.path) >= 8)
    first = DisruptionItem(type="blocked_cell", t=5, cells=[r.path[5]])
    inc1 = det.detect(world, Injector(np.random.default_rng(0), [first]).apply(world, 5), 5)
    assert inc1[0].overlap_kind == "isolated"
    world.step(); world.step(); world.step()
    second = DisruptionItem(type="blocked_cell", t=8, cells=[r.path[1]] if len(r.path) > 1 else [r.pos])
    inc2 = det.detect(world, Injector(np.random.default_rng(0), [second]).apply(world, 8), 8)
    assert inc2 and inc2[0].overlap_kind in ("sequential", "compound")
    assert inc1[0].id in inc2[0].interacts_with
    assert any(rd.incident_id == inc1[0].id and rd.type == "blocked_cell" for rd in inc2[0].related_disruptions)


def test_prompt_tells_the_llm_about_overlaps(world):
    r = next(r for r in world.robots if len(r.path) >= 6)
    inj = Injector(np.random.default_rng(0), [
        DisruptionItem(type="blocked_cell", t=5, cells=[r.path[3]]),
        DisruptionItem(type="robot_failure", t=5, anchor=0),
    ])
    inc = IncidentDetector().detect(world, inj.apply(world, 5), 5)[0]
    payload = json.loads(format_user_prompt(inc).split("Incident detected:\n", 1)[1].split("\n\nSelect")[0])
    assert payload["incident"]["overlap_kind"] == inc.overlap_kind
    assert "overlapping_earlier_disruptions" in payload["incident"]


# --------------------------------------------------------------- end to end
@pytest.mark.parametrize("overlap", ["same_aisle", "same_robot", "simultaneous", "cascade", "mixed"])
def test_every_overlap_mode_produces_interacting_incidents_without_collisions(tmp_path, overlap):
    rep = run(SimulationConfig(map_path=MAP, num_robots=120, steps=200, seed=3, disruption_rate=0.15,
                               overlap_mode=overlap, run_id=overlap, output_dir=str(tmp_path),
                               llm_provider="heuristic"))
    m = rep.metrics
    kinds = Counter(i.overlap_kind for i in rep.incidents)
    assert m.incidents_interacting > 0, kinds
    assert sum(v for k, v in kinds.items() if k != "isolated") > 0
    assert (m.executed_vertex_collisions, m.executed_edge_swaps, m.guard_interventions) == (0, 0, 0)
    assert set(m.by_overlap_kind) == set(kinds)
    assert sum(g["incidents"] for g in m.by_overlap_kind.values()) == m.incidents_total


SCENARIOS = build_scenarios()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_controlled_scenario_covers_its_combination(tmp_path, name):
    row = run_scenario(name, SCENARIOS[name], "llm", seed=3, robots=150, steps=125,
                       provider="heuristic", model=None, require_llm=False, out_dir=str(tmp_path))
    assert row["Incidents"] > 0
    assert row["RepetitionsDetected"] >= 1, row
    assert row["ExecutedCollisions"] == 0
    if SCENARIOS[name]["kind"] != "isolated":
        assert any(k != "isolated" for k in row["DetectedKinds"]), row


def test_scenario_suite_covers_every_type_pair():
    pairs = Counter((s["kind"], tuple(sorted(s["types"]))) for s in SCENARIOS.values())
    type_pairs = {tuple(sorted((a, b))) for a in TYPES for b in TYPES}
    for kind in ("simultaneous", "sequential"):
        assert {p for (k, p) in pairs if k == kind} == type_pairs
    assert ("compound", tuple(sorted(TYPES))) in pairs
    assert {p for (k, p) in pairs if k == "isolated"} == {(t,) for t in TYPES}
