"""End-to-end integration tests for the issues found in the integration review.

Each test pins one property the synopsis requires:
* robots move only along committed, collision-free MAPF plans
* failed robots stay failed, delayed robots wait
* the LLM is actually called (through openai SDK + Instructor) and its usage is
  accounted honestly; heuristic decisions are never counted as LLM calls
* the validator checks candidate plans against the whole fleet
* cache / negative memory behave as described
"""

import json
import os
import sys

import httpx
import numpy as np
import openai
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from mock_llm import FakeLLM  # noqa: E402

from amr_repair.cache.negative import NegativeMemory, get_hard_filter_key  # noqa: E402
from amr_repair.cache.store import StrategyCache  # noqa: E402
from amr_repair.disrupt.injector import Disruption  # noqa: E402
from amr_repair.llm.schema import Directive  # noqa: E402
from amr_repair.llm.strategist import LLMUnavailableError, Strategist  # noqa: E402
from amr_repair.loop import controller as controller_mod  # noqa: E402
from amr_repair.loop.audit import audit_frames  # noqa: E402
from amr_repair.loop.controller import SimulationConfig, SimulationController, run  # noqa: E402
from amr_repair.sentinel.incident import CongestionData, DisruptionData, IncidentRecord  # noqa: E402
from amr_repair.sim.agents import Robot  # noqa: E402
from amr_repair.sim.grid import GridMap  # noqa: E402
from amr_repair.sim.lifelong import World  # noqa: E402
from amr_repair.solvers import lns, pibt, windowed_pbs  # noqa: E402
from amr_repair.validate.validator import validate  # noqa: E402

MAP = "maps/warehouse-10-20-10-2-1.map"


def cfg(tmp_path, **kw):
    base = dict(map_path=MAP, num_robots=60, steps=120, seed=2, disruption_rate=0.15,
                run_id="t", output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"), llm_provider="heuristic")
    base.update(kw)
    return SimulationConfig(**base)


def make_incident(inc_id="inc_01", robots=(1, 2), density=0.1, interacts=False, types=("blocked_cell",)):
    return IncidentRecord(
        id=inc_id, t_detected=10,
        disruptions=[DisruptionData(type=t, cells=[(5, 5)]) for t in types],
        affected_robots=list(robots), size_bucket="small",
        congestion=CongestionData(level="low", region=((3, 3), (7, 7)), density=density),
        interacts_with=["x", inc_id] if interacts else [],
    )


# ----------------------------------------------------------------- simulator
class _PlanCheckingWorld(World):
    """Records, after all (re)planning of a step, the cell each robot's
    committed plan says it will occupy next."""

    def _enforce_consistency(self):
        super()._enforce_consistency()
        self.expected = {r.id: (r.path[0] if r.path else r.pos) for r in self.robots}


def test_robots_execute_their_committed_mapf_plans():
    """Every move a robot makes is the next cell of its committed,
    collision-free plan: robots do not walk independent/random routes."""
    gmap = GridMap.from_movingai(MAP)
    rng = np.random.default_rng(5)
    free = gmap.get_free_cells()
    idx = rng.choice(len(free), 80, replace=False)
    world = _PlanCheckingWorld(gmap, [Robot(id=i, pos=free[j], goal=free[j]) for i, j in enumerate(idx)],
                               rng, goal_seed=5)
    deviations = 0
    for _ in range(120):
        world.step()
        deviations += sum(1 for r in world.robots if r.pos != world.expected[r.id])
    assert deviations == 0
    assert audit_frames(world.frames) == (0, 0, 0)
    assert world.guard_interventions == 0
    assert world.goals_completed > 50  # lifelong: robots keep receiving and completing tasks


@pytest.mark.parametrize("mode", ["llm", "baseline_pibt", "baseline_lns", "baseline_pbs"])
def test_closed_loop_is_collision_free_in_every_mode(tmp_path, mode):
    rep = run(cfg(tmp_path, mode=mode))
    m = rep.metrics
    assert m.incidents_total > 0
    assert (m.executed_vertex_collisions, m.executed_edge_swaps) == (0, 0)
    assert audit_frames(rep.frames) == (0, 0, 0)
    assert m.guard_interventions == 0


def test_same_seed_gives_same_tasks_across_modes(tmp_path):
    a = SimulationController(cfg(tmp_path, mode="llm"))
    b = SimulationController(cfg(tmp_path, mode="baseline_pbs"))
    assert [r.pos for r in a.world.robots] == [r.pos for r in b.world.robots]
    assert [r.goal for r in a.world.robots] == [r.goal for r in b.world.robots]
    assert [(d.type, d.t, d.robot, d.target_robot) for d in a.injector.disruptions] == \
           [(d.type, d.t, d.robot, d.target_robot) for d in b.injector.disruptions]


# --------------------------------------------------------------- disruptions
def _first_planned_cell_robot(controller, min_ahead=3):
    for r in controller.world.robots:
        if len(r.path) >= min_ahead:
            return r
    raise RuntimeError


def test_failed_robot_is_never_resurrected_by_repair(tmp_path):
    probe = SimulationController(cfg(tmp_path, schedule=[]))
    victim = probe.world.robots[3].id
    schedule = [Disruption(type="robot_failure", t=5, robot=victim)]
    c = SimulationController(cfg(tmp_path, schedule=schedule, steps=60))
    rep = c.run()
    after = rep.frames[6:]
    assert {f.state[victim] for f in after} == {"failed"}
    assert len({tuple(f.pos[victim]) for f in after}) == 1
    for inc in rep.incidents:
        if inc.executed_directive:
            assert victim not in inc.executed_directive.robots


def test_delayed_robot_waits_for_k_steps_even_when_repaired(tmp_path):
    c = SimulationController(cfg(tmp_path, schedule=[], steps=1))
    rid = _first_planned_cell_robot(c).id
    k = 6
    rep = run(cfg(tmp_path, schedule=[Disruption(type="robot_delay", t=4, robot=rid, k_steps=k)], steps=30))
    positions = {tuple(f.pos[rid]) for f in rep.frames[4:4 + k + 1]}
    assert len(positions) == 1
    assert all(f.state[rid] == "delayed" for f in rep.frames[5:4 + k])


def test_blocked_aisle_ahead_of_robot_creates_must_replan_incident(tmp_path):
    rep = run(cfg(tmp_path, schedule={"rate": 0.3, "types": ["blocked_cell"], "overlap": "none",
                                     "max_steps": 80, "targeting": "active"}, steps=80))
    assert rep.metrics.incidents_total > 0
    assert any(inc.must_replan for inc in rep.incidents)
    for inc in rep.incidents:
        if inc.executed_valid:
            # repaired robots never enter a blocked cell of that incident
            blocked = {tuple(c) for d in inc.disruptions for c in d.cells}
            for path in inc.paths.repaired.values():
                assert not blocked & {tuple(c) for c in path}


# ----------------------------------------------------------------------- LLM
def test_llm_is_really_called_through_instructor(tmp_path):
    fake = FakeLLM()
    rep = run(cfg(tmp_path, llm_provider=None, llm_client=fake.client(), llm_model="fake"))
    m = rep.metrics
    assert m.llm_provider == "custom"
    assert len(fake.requests) > 0
    assert m.llm_api_requests == len(fake.requests)
    assert m.llm_tokens == 120 * len(fake.requests)
    assert m.llm_calls > 0 and m.heuristic_decisions == 0
    llm_incidents = [i for i in rep.incidents if i.llm]
    assert llm_incidents and all(i.llm.called for i in llm_incidents)
    # request carries the system prompt, JSON mode, and never the map
    body = fake.requests[0]
    assert body["response_format"] == {"type": "json_object"}
    user = [msg for msg in body["messages"] if msg["role"] == "user"][0]["content"]
    assert "candidate_robots" in user and "obstacles" not in user


def test_hallucinated_robot_id_is_rejected_and_reasked(tmp_path):
    fake = FakeLLM(hallucinate_first=True)
    rep = run(cfg(tmp_path, llm_provider=None, llm_client=fake.client(), llm_model="fake"))
    first = next(i for i in rep.incidents if i.llm)
    call = first.llm.calls[0]
    assert call.api_requests == 2  # original + Instructor re-ask
    assert 99999 not in call.directive.robots
    reask = [m for m in fake.requests[1]["messages"] if m["role"] == "user"][-1]["content"]
    assert "not candidates" in reask


def test_llm_failure_is_not_counted_as_llm_call(tmp_path):
    def boom(request):
        return httpx.Response(500, json={"error": {"message": "server down"}})
    client = openai.OpenAI(api_key="x", base_url="http://down/v1", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(boom)))
    rep = run(cfg(tmp_path, llm_provider=None, llm_client=client, llm_model="fake", cache_enabled=False))
    m = rep.metrics
    assert m.incidents_total > 0
    assert m.llm_calls == 0 and m.llm_errors > 0 and m.heuristic_decisions > 0
    for inc in rep.incidents:
        if inc.llm:
            assert inc.llm.called is False
            assert inc.llm.calls[0].error is not None


def test_heuristic_is_labelled_and_never_counted_as_llm(tmp_path):
    rep = run(cfg(tmp_path, llm_provider="heuristic"))
    m = rep.metrics
    assert m.llm_provider == "heuristic" and rep.meta.llm_provider == "heuristic"
    assert m.llm_calls == 0 and m.llm_tokens == 0 and m.heuristic_decisions > 0
    assert all(not i.llm.called for i in rep.incidents if i.llm)


def test_require_llm_fails_loudly_without_provider(monkeypatch):
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(LLMUnavailableError):
        Strategist(allow_heuristic=False)
    with pytest.raises(LLMUnavailableError):
        Strategist(provider="groq")


# ----------------------------------------------------------------- validator
def _quiet_world(n=4):
    gmap = GridMap.from_movingai(MAP)
    for c in gmap.get_free_cells():
        if all(gmap.is_free((c[0] + dx, c[1])) for dx in range(-3, 4)):
            cells = [(c[0] + dx, c[1]) for dx in range(-3, 4)]
            break
    robots = [Robot(id=0, pos=cells[0], goal=cells[6]), Robot(id=1, pos=cells[6], goal=cells[0])]
    w = World(gmap, robots, np.random.default_rng(0), goal_seed=0)
    for r in w.robots:
        r.path = []
    return w, cells


def test_validator_checks_against_robots_that_are_not_replanned():
    w, cells = _quiet_world()
    # robot 1 is NOT being replanned; it is committed to walk left and park at cells[4]
    w.robots[1].path = [cells[5], cells[4]]
    ok = validate({0: [cells[1], cells[2]]}, w, t0=0)
    assert ok.valid, ok.violation
    bad = validate({0: [cells[1], cells[2], cells[3], cells[4]]}, w, t0=0)
    assert not bad.valid and "vertex collision robots 0,1" in bad.violation


def test_validator_flags_unrepaired_required_robot():
    w, cells = _quiet_world()
    w.robots[1].path = [cells[5], cells[4]]
    w.map.block([cells[4]])
    v = validate({0: [cells[0]]}, w, t0=0, required=[0, 1])
    assert not v.valid and "unrepaired robot 1" in v.violation


def test_validator_rejects_moving_a_delayed_robot():
    w, cells = _quiet_world()
    w.robots[0].delay_remaining = 3
    v = validate({0: [cells[1], cells[2], cells[3]]}, w, t0=0)
    assert not v.valid and "delay" in v.violation


@pytest.mark.parametrize("solver", [pibt, lns, windowed_pbs])
def test_solvers_respect_delays_and_fleet(solver):
    gmap = GridMap.from_movingai(MAP)
    rng = np.random.default_rng(3)
    free = gmap.get_free_cells()
    idx = rng.choice(len(free), 40, replace=False)
    w = World(gmap, [Robot(id=i, pos=free[j], goal=free[j]) for i, j in enumerate(idx)], rng, goal_seed=3)
    for _ in range(5):
        w.step()
    w.robots[0].delay_remaining = 4
    w.robots[0].path = [w.robots[0].pos] * 4 + w.robots[0].path
    paths = solver.solve(w, [0, 1, 2], [0, 1, 2], 20)
    assert set(paths) == {0, 1, 2}
    assert all(c == w.robots[0].pos for c in paths[0][:4])
    if solver is not pibt:  # PIBT is myopic and may be rejected; the others must be valid here
        v = validate(paths, w, t0=w.t, required=[0, 1, 2])
        assert v.valid, v.violation


# --------------------------------------------------------- cache & negatives
def test_cache_does_not_mix_isolated_and_interacting_incidents():
    cache = StrategyCache(threshold=0.85)
    cache.store(make_incident("a", robots=(1,)), Directive(robots=[1], solver="pibt", priority=[1], horizon=15))
    assert cache.lookup(make_incident("b", robots=(2, 3), interacts=True)) is None


def test_cache_threshold_discriminates():
    # 2 and 3 robots share the "2-3" category; similarity then decides
    iso1 = make_incident("a", robots=(1, 2))
    iso3 = make_incident("b", robots=(1, 2, 3))
    d = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)
    sim = StrategyCache.similarity(iso1.soft_features(), iso3.soft_features())
    assert 0.3 < sim < 0.95
    loose, strict = StrategyCache(threshold=0.3), StrategyCache(threshold=0.95)
    loose.store(iso1, d)
    strict.store(iso1, d)
    assert loose.lookup(iso3) is not None
    assert strict.lookup(iso3) is None


def test_cache_category_is_disruption_types_and_robot_count():
    one, three = make_incident("a", robots=(1,)), make_incident("b", robots=(1, 2, 3))
    cache = StrategyCache(threshold=0.0)
    cache.store(one, Directive(robots=[1], solver="pibt", priority=[1], horizon=15))
    assert cache.lookup(three) is None          # 1 robot vs 2-3 robots: different category
    assert cache.probe(three) is None


def test_negative_memory_uses_strategy_signature_and_clears_on_success():
    neg = NegativeMemory()
    cache = StrategyCache(threshold=0.5, negative_memory=neg)
    inc_a, inc_b = make_incident("a", robots=(1, 2)), make_incident("b", robots=(7, 8))
    d = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)
    cache.store(inc_a, d)
    neg.add(inc_a, d, "vertex collision robots 1,2")
    # different robot IDs, same category + same signature -> still excluded
    assert cache.lookup(inc_b) is None
    assert neg.rejected_for(inc_b)[0]["solver"] == "pibt"
    neg.clear(inc_b, Directive(robots=[7, 8], solver="pibt", priority=[7, 8], horizon=15))
    assert cache.lookup(inc_b) is not None


def test_cache_entries_are_unique_across_runs(tmp_path):
    path = str(tmp_path / "cache.json")
    d = Directive(robots=[1, 2], solver="lns", priority=[1, 2], horizon=20)
    StrategyCache(filepath=path, namespace="run1").store(make_incident("inc_001"), d)
    c2 = StrategyCache(filepath=path, namespace="run2")
    c2.store(make_incident("inc_001"), d)
    assert sorted(e.incident_id for e in c2.entries) == ["run1:inc_001", "run2:inc_001"]


def test_runs_do_not_touch_repo_cache_by_default(tmp_path):
    run(cfg(tmp_path, steps=40))
    assert not os.path.exists(tmp_path / "cache" / "strategy_cache.json")


# -------------------------------------------------- validator gate / ablation
def _bad_solver(world, robots, priority, horizon):
    by_id = {r.id: r for r in world.robots}
    return {rid: [by_id[rid].pos] * 12 for rid in robots}  # deadlock: never valid


def test_validator_gate_vs_no_validator_ablation(tmp_path, monkeypatch):
    for name in ("pibt", "lns", "windowed_pbs"):  # every candidate plan (incl. fallback) is invalid
        monkeypatch.setitem(controller_mod.SOLVER_REGISTRY, name, _bad_solver)
    gated = run(cfg(tmp_path, cache_enabled=False, validator_enabled=True, run_id="g"))
    for inc in gated.incidents:
        assert inc.decision_source in ("llm", "llm_retry", "heuristic", "heuristic_retry", "fallback", "emergency_hold")
        if inc.validator and not inc.validator[0].valid:
            assert inc.executed_valid in (True, False)
            assert inc.decision_source != "unvalidated"
    assert gated.metrics.incidents_total > 0
    assert gated.metrics.executed_safety_violations == 0
    assert gated.metrics.safety_violations_intercepted > 0
    assert all(i.decision_source == "emergency_hold" for i in gated.incidents)

    ungated = run(cfg(tmp_path, cache_enabled=False, validator_enabled=False, run_id="u"))
    assert ungated.metrics.executed_safety_violations > 0
    assert any(i.decision_source == "unvalidated" for i in ungated.incidents)
