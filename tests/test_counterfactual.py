"""Counterfactual per-incident evaluation (amr_repair/loop/counterfactual.py) and
re-simulation from recorded decisions (scripts/counterfactual_from_replay.py)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import counterfactual_from_replay as cfr  # noqa: E402

from amr_repair.loop import counterfactual as cf  # noqa: E402
from amr_repair.loop.controller import SimulationConfig, SimulationController, run  # noqa: E402
from amr_repair.sim.agents import Robot  # noqa: E402
from amr_repair.sim.grid import load_map  # noqa: E402
from amr_repair.sim.lifelong import World  # noqa: E402

MAP = "maps/warehouse-10-20-10-2-1.map"


def cfg(tmp_path, **kw):
    base = dict(map_path=MAP, num_robots=60, steps=120, seed=3, disruption_rate=0.12, overlap_mode="mixed",
                run_id="cf", output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"),
                llm_provider="heuristic", export=False)
    base.update(kw)
    return SimulationConfig(**base)


def positions(replay):
    return [[tuple(p) for p in f.pos] for f in replay.frames]


@pytest.fixture(scope="module")
def with_cf(tmp_path_factory):
    return run(cfg(tmp_path_factory.mktemp("cf"), counterfactual_window=30))


# ------------------------------------------------------------------ branches

def test_branches_never_change_the_main_run(tmp_path, with_cf):
    plain = run(cfg(tmp_path))
    assert with_cf.incidents, "scenario must produce incidents"
    assert positions(plain) == positions(with_cf)
    assert [(i.id, i.t_detected) for i in plain.incidents] == [(i.id, i.t_detected) for i in with_cf.incidents]
    assert plain.metrics.throughput == with_cf.metrics.throughput


def test_every_incident_has_all_arms_and_reproduces_the_executed_plan(with_cf):
    for inc in with_cf.incidents:
        ev = inc.counterfactual
        assert set(ev["arms"]) == {"pibt", "lns", "windowed_pbs", "executed"}
        assert ev["window"] == 30
        assert ev["reproduced"], f"{inc.id}: branch did not re-produce the executed plan"
        assert ev["oracle"] and set(ev["oracle"]) <= set(cf.FIXED_ARMS)
        for arm in ev["arms"].values():
            assert 0 <= arm["recovery_steps"] <= 30
            assert arm["source"] in ("direct", "fallback", "hold")
    s = with_cf.metrics.counterfactual
    assert s["incidents"] == len(with_cf.incidents)
    assert s["reproduced_rate"] == 1.0


def test_baseline_run_executed_arm_equals_its_fixed_arm(tmp_path):
    """In a fixed-PIBT run the executed decision IS the PIBT arm, so both branches must agree exactly."""
    rep = run(cfg(tmp_path, mode="baseline_pibt", cache_enabled=False, counterfactual_window=30))
    assert rep.incidents
    for inc in rep.incidents:
        a = {k: v for k, v in inc.counterfactual["arms"]["executed"].items() if k != "solver_ms"}
        b = {k: v for k, v in inc.counterfactual["arms"]["pibt"].items() if k != "solver_ms"}
        if inc.decision_source == "baseline":  # fallback runs carry the fallback directive
            assert a == b


def test_counterfactual_requires_the_validator(tmp_path):
    with pytest.raises(ValueError):
        SimulationController(cfg(tmp_path, validator_enabled=False, counterfactual_window=10))


def test_fork_is_independent_of_the_original():
    rng = np.random.default_rng(0)
    m = load_map(MAP)
    free = m.get_free_cells()
    world = World(m, [Robot(id=i, pos=free[i * 7], goal=free[i * 7]) for i in range(10)], rng, goal_seed=0)
    fork = cf.fork_world(world)
    cell = free[500]
    fork.map.block([cell])
    fork.robots[0].path = [fork.robots[0].pos] * 5
    fork.step()
    assert world.map.is_free(cell)
    assert world.t == 0 and fork.t == 1
    assert world.robots[0].path != fork.robots[0].path
    assert len(world.frames) == 1  # history kept on the original, not copied


# ---------------------------------------------------------------- aggregation

def _arm(goals, recovery=5, source="direct", detour=0):
    return {"goals": goals, "recovery_steps": recovery, "recovered": True, "first_valid": source == "direct",
            "source": source, "detour": detour, "shortest": 10, "solver_ms": 1.0}


def _ev(pibt, lns, pbs, executed):
    arms = {"pibt": pibt, "lns": lns, "windowed_pbs": pbs, "executed": executed}
    best = min(cf.rank_key(arms[a]) for a in cf.FIXED_ARMS)
    return {"window": 50, "arms": arms, "oracle": [a for a in cf.FIXED_ARMS if cf.rank_key(arms[a]) == best],
            "matches_oracle": cf.rank_key(executed) <= best,
            "goal_regret": max(arms[a]["goals"] for a in cf.FIXED_ARMS) - executed["goals"], "reproduced": True}


def test_rank_prefers_no_hold_then_goals_then_recovery_then_detour():
    assert cf.rank_key(_arm(5, source="hold")) > cf.rank_key(_arm(1))
    assert cf.rank_key(_arm(6)) < cf.rank_key(_arm(5))
    assert cf.rank_key(_arm(5, recovery=3)) < cf.rank_key(_arm(5, recovery=4))
    assert cf.rank_key(_arm(5, detour=1)) < cf.rank_key(_arm(5, detour=2))


def test_summary_random_vbs_sbs_and_gap_closed():
    evals = [
        _ev(_arm(10), _arm(12), _arm(11), _arm(12)),  # executed picks the best
        _ev(_arm(10), _arm(8), _arm(9), _arm(8)),     # executed picks the worst
    ]
    s = cf.summarize(evals)
    assert s["vbs_mean_goals"] == 11.0                # (12 + 10) / 2
    assert s["random_mean_goals"] == 10.0             # (11 + 9) / 2, exact expectation
    assert s["sbs"] == "pibt" and s["sbs_mean_goals"] == 10.0  # pibt 20 total, lns 20 total -> first in order
    assert s["executed_matches_oracle_rate"] == 0.5
    assert s["mean_goal_regret"] == 1.0
    # executed 20 total, random 20, vbs 22 -> closes 0 % of the gap
    assert s["gap_closed_vs_random"] == 0.0
    assert s["oracle_counts"] == {"pibt": 1, "lns": 1, "windowed_pbs": 0}


def test_summary_gap_is_none_when_all_arms_tie():
    s = cf.summarize([_ev(_arm(5), _arm(5), _arm(5), _arm(5))])
    assert s["gap_closed_vs_random"] is None and s["gap_closed_vs_sbs"] is None
    assert cf.summarize([]) == {"incidents": 0}


# ------------------------------------------------- re-simulation from a replay

def test_resimulation_from_recorded_decisions_is_faithful(tmp_path):
    original = json.loads(run(cfg(tmp_path)).model_dump_json())
    resim = run(cfg(tmp_path, cache_enabled=False, recorded_decisions=cfr.recorded_decisions(original),
                    counterfactual_window=20))
    assert cfr.check_faithful(original, resim) == []
    merged = cfr.merge(original, resim, {"disruption_rate": 0.12, "overlap_mode": "mixed", "max_block_size": 3})
    assert all(i["counterfactual"] for i in merged["incidents"])
    # the settings that reproduced the run are written into meta for next time
    assert merged["meta"]["overlap_mode"] == "mixed" and merged["meta"]["counterfactual_window"] == 20
    assert merged["metrics"]["counterfactual"]["method"].startswith("re-simulated")
    # original telemetry is kept untouched
    assert [i["decision_source"] for i in merged["incidents"]] == [i["decision_source"] for i in original["incidents"]]


def test_resimulation_with_the_wrong_config_is_rejected(tmp_path):
    """A wrong seed must be caught: either an unknown incident stops the run, or the frame check fails."""
    original = json.loads(run(cfg(tmp_path)).model_dump_json())
    try:
        resim = run(cfg(tmp_path, seed=4, recorded_decisions=cfr.recorded_decisions(original)))
    except RuntimeError as e:
        assert "diverged" in str(e)
        return
    assert cfr.check_faithful(original, resim) != []


def test_settings_come_from_meta_and_contradictions_are_rejected():
    meta = {"seed": 3, "num_robots": 120, "steps": 201, "disruption_rate": 0.12, "overlap_mode": "mixed",
            "max_block_size": 3}
    blank = dict(seed=None, num_robots=None, steps=None, disruption_rate=None, overlap_mode=None, max_block_size=None)
    s = cfr.resolve_settings(meta, blank)
    assert s["steps"] == 200 and s["overlap_mode"] == "mixed" and s["disruption_rate"] == 0.12
    with pytest.raises(SystemExit):
        cfr.resolve_settings(meta, {**blank, "num_robots": 50})
    with pytest.raises(SystemExit):  # old replay without generator settings and none given
        cfr.resolve_settings({"seed": 3, "num_robots": 120, "steps": 201}, blank)


def test_new_replays_record_their_generator_settings(with_cf):
    m = with_cf.meta
    assert (m.disruption_rate, m.overlap_mode, m.max_block_size, m.counterfactual_window) == (0.12, "mixed", 3, 30)
