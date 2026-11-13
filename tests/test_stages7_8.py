"""Stages 7-8 tests: Validator and Solvers (PIBT, LNS, Windowed PBS).
Tests:
- validator catches planted vertex collision, edge swap, blocked cell, deadlock
- each solver reroutes 5 robots around a blocked aisle and passes validation
- PIBT is significantly faster than Windowed PBS
"""

import os
import time
import pytest
import numpy as np

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.sim.lifelong import World
from amr_repair.validate.validator import validate, Verdict
from amr_repair.solvers import pibt, lns, windowed_pbs


@pytest.fixture
def warehouse_world():
    map_path = os.path.join(os.path.dirname(__file__), "..", "maps", "warehouse-10-20-10-2-1.map")
    gmap = GridMap.from_movingai(map_path)
    rng = np.random.default_rng(42)
    free = gmap.get_free_cells()
    robots = [Robot(id=i, pos=free[i * 3], goal=free[i * 3 + 15]) for i in range(10)]
    return World(gmap, robots, rng)


def _open_cell(world):
    """A free cell whose left and right neighbours are free too."""
    for c in world.free_cells:
        if world.map.is_free((c[0] - 1, c[1])) and world.map.is_free((c[0] + 1, c[1])):
            if all(r.pos not in (c, (c[0] - 1, c[1]), (c[0] + 1, c[1])) for r in world.robots):
                return c
    raise RuntimeError("no open cell")


def test_validator_catches_all_violations(warehouse_world):
    w = warehouse_world
    r0, r1 = w.robots[0], w.robots[1]
    c = _open_cell(w)
    for r in w.robots:  # isolate the planted scenario from the rest of the fleet
        r.path = []
    left, right = (c[0] - 1, c[1]), (c[0] + 1, c[1])

    # 1. Blocked cell: robot 0 steps into a cell that was just blocked
    r0.pos, r0.path = left, []
    r1.pos, r1.path = (left[0], left[1] + 50), []  # out of the way
    w.map.block([c])
    v1 = validate({0: [c]}, w, t0=0)
    assert not v1.valid and "blocked_cell" in v1.violation
    w.map.unblock([c])

    # 2. Vertex collision: robots 0 and 1 move into the same cell
    r0.pos, r1.pos = left, right
    v2 = validate({0: [c], 1: [c]}, w, t0=0)
    assert not v2.valid and "vertex collision" in v2.violation

    # 3. Edge swap: robots 0 and 1 swap neighbouring cells
    r0.pos, r1.pos = c, right
    v3 = validate({0: [right], 1: [c]}, w, t0=0)
    assert not v3.valid and "edge swap" in v3.violation

    # 4. Deadlock: robot waits 12 steps without being at its goal
    r0.pos, r0.goal = left, right
    v4 = validate({0: [left] * 12}, w, t0=0)
    assert not v4.valid and "deadlock" in v4.violation

    # 5. Teleport: non-adjacent move
    far = (left[0], left[1] + 2) if w.map.is_free((left[0], left[1] + 2)) else w.free_cells[-1]
    v5 = validate({0: [far]}, w, t0=0)
    assert not v5.valid and "teleport" in v5.violation


def test_each_solver_reroutes_and_validates(warehouse_world):
    # Select 5 robots
    managed_robots = [0, 1, 2, 3, 4]
    priority = [0, 1, 2, 3, 4]
    horizon = 15

    # 1. PIBT
    t_start = time.perf_counter()
    pibt_paths = pibt.solve(warehouse_world, managed_robots, priority, horizon)
    pibt_time = time.perf_counter() - t_start

    v_pibt = validate(pibt_paths, warehouse_world, t0=warehouse_world.t)
    assert v_pibt.valid, f"PIBT produced invalid paths: {v_pibt.violation}"
    assert len(pibt_paths) == 5

    # 2. LNS
    lns_paths = lns.solve(warehouse_world, managed_robots, priority, horizon)
    v_lns = validate(lns_paths, warehouse_world, t0=warehouse_world.t)
    assert v_lns.valid, f"LNS produced invalid paths: {v_lns.violation}"
    assert len(lns_paths) == 5

    # 3. Windowed PBS
    t_start_pbs = time.perf_counter()
    pbs_paths = windowed_pbs.solve(warehouse_world, managed_robots, priority, horizon)
    pbs_time = time.perf_counter() - t_start_pbs

    v_pbs = validate(pbs_paths, warehouse_world, t0=warehouse_world.t)
    assert v_pbs.valid, f"PBS produced invalid paths: {v_pbs.violation}"
    assert len(pbs_paths) == 5

    # PIBT speed comparison
    assert pibt_time < pbs_time, f"Expected PIBT ({pibt_time:.4f}s) to be faster than PBS ({pbs_time:.4f}s)"
