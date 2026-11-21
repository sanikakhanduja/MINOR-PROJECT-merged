"""Step 2-4 additions: LLM response store, decision delay (thinking time charged to the
robots), cache warm start / probe / precision, counterfactual progress, and the sweep
plus its analysis."""

import json
import math
import os

import numpy as np
import pytest

from amr_repair.cache.store import StrategyCache
from amr_repair.experiments import analyze_sweep as an
from amr_repair.experiments import sweep
from amr_repair.llm.response_store import ResponseNotRecorded, ResponseStore, request_key
from amr_repair.loop import counterfactual as cf
from amr_repair.loop.controller import SimulationConfig, run
from tests.mock_llm import FakeLLM

MAP = "maps/warehouse-10-20-10-2-1.map"


def cfg(tmp_path, **kw):
    base = dict(map_path=MAP, num_robots=60, steps=100, seed=3, disruption_rate=0.12, overlap_mode="mixed",
                run_id="t", output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"),
                llm_provider="heuristic", export=False)
    base.update(kw)
    return SimulationConfig(**base)


def positions(replay):
    return [[tuple(p) for p in f.pos] for f in replay.frames]


# ------------------------------------------------------------ response store

def test_response_store_roundtrip_and_replay_mode(tmp_path):
    path = str(tmp_path / "llm.jsonl")
    key = request_key("gemini", "m", 0.2, "sys", "user")
    assert key != request_key("gemini", "m", 0.2, "sys", "user2")
    rec = ResponseStore(path, "record")
    assert rec.get(key) is None
    rec.put(key, {"directive": {"robots": [1]}, "latency_ms": 10})
    again = ResponseStore(path, "replay")
    assert again.get(key)["latency_ms"] == 10
    with pytest.raises(ResponseNotRecorded):
        again.get("missing")
    with pytest.raises(ValueError):
        ResponseStore(path, "sometimes")


def test_recorded_llm_run_replays_identically_without_api_calls(tmp_path):
    store = str(tmp_path / "llm.jsonl")
    fake = FakeLLM()
    common = dict(llm_provider=None, llm_model="custom", decision_step_seconds=None, cache_enabled=False,
                  llm_response_store=store)
    first = run(cfg(tmp_path, llm_client=fake.client(), llm_response_mode="record", **common))
    n_requests = len(fake.requests)
    assert first.metrics.llm_calls > 0 and n_requests > 0
    assert first.metrics.llm_replayed_calls == 0
    calls = [c for i in first.incidents if i.llm for c in i.llm.calls]
    assert all(c.returned_model == "fake" for c in calls)

    # replay: no client at all, same provider/model -> same run, nothing sent to the API
    second = run(cfg(tmp_path, llm_provider="custom", llm_response_mode="replay", **{
        k: v for k, v in common.items() if k != "llm_provider"}))
    assert len(fake.requests) == n_requests
    assert second.metrics.llm_replayed_calls == second.metrics.llm_calls == first.metrics.llm_calls
    assert positions(second) == positions(first)
    assert second.meta.llm_response_store == "replay"


# ------------------------------------------------------------ decision delay

def test_thinking_time_is_charged_to_the_replanned_robots(tmp_path):
    step_s = 0.001  # 1 ms per step: the fake LLM's few milliseconds become a few steps of waiting
    replay = run(cfg(tmp_path, llm_provider=None, llm_client=FakeLLM().client(), cache_enabled=False,
                     decision_step_seconds=step_s, counterfactual_window=20))
    llm_incidents = [i for i in replay.incidents if i.llm and i.llm.calls]
    assert llm_incidents
    for inc in llm_incidents:
        expected = math.floor(sum(c.latency_ms for c in inc.llm.calls) / (step_s * 1000))
        # the delay is set before each attempt, so it covers at least the first call's latency
        first = math.floor(inc.llm.calls[0].latency_ms / (step_s * 1000))
        assert first <= inc.decision_delay_steps <= expected
        assert inc.counterfactual["decision_delay"] == inc.decision_delay_steps
        if inc.decision_delay_steps and inc.executed_directive is not None:
            # every robot the incident required waits in place for the decision
            for rid in inc.executed_directive.robots:
                p = inc.paths.repaired.get(str(rid))
                if p and rid in inc.must_replan:
                    t0 = inc.t_detected
                    start = replay.frames[t0].pos[rid]
                    assert all(tuple(c) == tuple(start) for c in p[:inc.decision_delay_steps])
    assert replay.metrics.decision_delay_steps == sum(i.decision_delay_steps for i in replay.incidents)
    # the executed branch applies the same delay, so it reproduces the executed plan
    assert replay.metrics.counterfactual["reproduced_rate"] == 1.0
    assert replay.meta.decision_step_seconds == step_s


def test_rule_based_decisions_cost_no_steps(tmp_path):
    replay = run(cfg(tmp_path))
    assert replay.metrics.decision_delay_steps == 0


# --------------------------------------------------------------------- cache

def test_cache_export_warm_start_and_probe(tmp_path):
    export = str(tmp_path / "train_cache.json")
    train = run(cfg(tmp_path, seed=101, run_id="train", cache_export_path=export))
    assert os.path.exists(export)
    with open(export) as f:
        stored = json.load(f)
    assert len(stored) == train.metrics.cache_insertions > 0

    warm = run(cfg(tmp_path, seed=3, run_id="warm", cache_warm_path=export, cache_probe=True))
    assert warm.meta.cache_warm_entries == len(stored)
    m = warm.metrics
    assert m.cache_hits <= m.cache_matches <= m.cache_lookups
    assert m.cache_misses == m.cache_lookups - m.cache_hits
    if m.cache_matches:
        assert m.cache_precision == round(m.cache_hits / m.cache_matches, 3)
    # hits are matches whose dry run passed
    assert m.cache_hits == sum(1 for i in warm.incidents if i.dry_run and i.dry_run.ran and i.dry_run.valid)
    probes = [i.cache.probe for i in warm.incidents if i.cache and i.cache.probe]
    assert probes, "a warm cache should have candidates to probe"
    for inc in warm.incidents:
        if inc.cache and inc.cache.hit:  # the probe sees the same best candidate as the lookup
            assert inc.cache.probe["similarity"] >= inc.cache.similarity - 1e-3


def test_load_entries_skips_old_format_and_respects_capacity(tmp_path):
    old = [{"incident_id": "x", "hard_filter_key": "a|b|c|d", "features": [0] * 6, "directive": {}}]
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    cache = StrategyCache(max_capacity=2)
    assert cache.load_entries(str(p)) == 0


# ------------------------------------------------------------ counterfactual

def test_arms_record_fleet_progress_and_distinct_flag(tmp_path):
    replay = run(cfg(tmp_path, counterfactual_window=20))
    evs = [i.counterfactual for i in replay.incidents if i.counterfactual]
    assert evs
    for ev in evs:
        for arm in ev["arms"].values():
            assert isinstance(arm["progress"], int)
        assert ev["distinct"] == (len(ev["oracle"]) < len(cf.FIXED_ARMS))
    s = replay.metrics.counterfactual
    assert 0.0 <= s["distinct_rate"] <= 1.0
    assert "gap_closed_vs_random_progress" in s and "vbs_mean_progress" in s


def test_rank_key_uses_progress_after_goals():
    a = dict(source="direct", goals=5, progress=100, recovery_steps=3, detour=0)
    b = dict(source="direct", goals=5, progress=120, recovery_steps=9, detour=9)
    c = dict(source="direct", goals=6, progress=10, recovery_steps=9, detour=9)
    assert cf.rank_key(c) < cf.rank_key(b) < cf.rank_key(a)
    old = dict(source="direct", goals=5, recovery_steps=3, detour=0)  # recorded before progress existed
    assert cf.rank_key(old) == (0, -5, 0, 3, 0)


# ------------------------------------------------------------------ analysis

def test_bootstrap_and_ratio_intervals():
    rng = np.random.default_rng(0)
    est, lo, hi = an.bootstrap_ci([1, 2, 3, 4, 5], rng)
    assert est == 3 and lo < 3 < hi
    assert an.bootstrap_ci([], rng) == (None, None, None)
    assert an.bootstrap_ci([7], rng) == (7, 7, 7)
    est, lo, hi = an.ratio_ci([1, 1], [2, 2], rng)
    assert est == 0.5 and lo == hi == 0.5
    assert an.ratio_ci([1], [0], rng) == (None, None, None)


def test_wilcoxon_needs_differences():
    assert an.wilcoxon([1] * 6, [1] * 6) is None
    assert an.wilcoxon([1, 2], [0, 0]) is None  # fewer than 5 pairs
    w = an.wilcoxon([5, 6, 7, 8, 9, 10], [1, 1, 1, 1, 1, 1])
    assert w["n"] == 6 and w["p"] < 0.05 and w["median_diff"] > 0


def test_threshold_curve_counts():
    def inc(sim, valid, solver, oracle):
        return {"cache": {"probe": {"similarity": sim, "valid": valid, "solver": solver}},
                "counterfactual": {"oracle": oracle}}
    incs = [inc(0.95, True, "lns", ["lns"]), inc(0.8, False, "pibt", ["lns"]), {"cache": {"probe": None}}]
    rows = {r["threshold"]: r for r in an.threshold_curve(incs)}
    assert rows[0.9]["matches"] == 1 and rows[0.9]["wrong_hit_rate"] == 0 and rows[0.9]["oracle_agreement"] == 1
    assert rows[0.75]["matches"] == 2 and rows[0.75]["wrong_hit_rate"] == 0.5
    assert rows[0.75]["coverage"] == round(2 / 3, 4)
    assert rows[0.99]["matches"] == 0 and rows[0.99]["wrong_hit_rate"] is None


# --------------------------------------------------------------------- sweep

def test_sweep_plan_separates_training_and_test_seeds():
    args = sweep.parse_args(["--robots", "50", "--rates", "0.1", "0.2", "--seeds", "1", "2",
                             "--train-seeds", "9", "--out", "x"])
    jobs = sweep.plan_jobs(args)
    first, second = jobs["first"], jobs["second"]
    assert {j["kind"] for j in first} == {"reference", "train", "cold"}
    assert {j["kind"] for j in second} == {"warm"}
    assert {j["seed"] for j in first if j["kind"] == "train"} == {9}
    assert {j["seed"] for j in first + second if j["kind"] in ("cold", "warm")} == {1, 2}
    assert all(j["rate"] == 0.0 for j in first if j["kind"] == "reference")
    assert len(second) == 2 * 2  # rates x test seeds


def test_tiny_sweep_end_to_end(tmp_path):
    out = str(tmp_path / "sw")
    sweep.main(["--robots", "40", "--rates", "0.2", "--seeds", "1", "2", "--train-seeds", "5",
                "--steps", "60", "--window", "10", "--workers", "1", "--out", out])
    names = sorted(os.listdir(os.path.join(out, "runs")))
    assert names == sorted(["reference_r40_rate0_s1.json", "reference_r40_rate0_s2.json",
                            "train_r40_rate0.2_s5.json", "cold_r40_rate0.2_s1.json", "cold_r40_rate0.2_s2.json",
                            "warm_r40_rate0.2_s1.json", "warm_r40_rate0.2_s2.json"])
    with open(os.path.join(out, "runs", "warm_r40_rate0.2_s1.json")) as f:
        warm = json.load(f)
    assert "frames" not in warm and warm["meta"]["cache_warm_entries"] is not None
    sweep.main(["--robots", "40", "--rates", "0.2", "--seeds", "1", "2", "--train-seeds", "5",
                "--steps", "60", "--window", "10", "--workers", "1", "--out", out])  # resumes: nothing re-run
    summary = an.analyze(out)
    cell = summary["cells"][0]
    assert (cell["robots"], cell["rate"]) == (40, 0.2)
    assert cell["cold"]["n_seeds"] == cell["warm"]["n_seeds"] == 2
    assert cell["cold"]["cache"]["warm_entries"] == 0
    an.write_markdown(summary, str(tmp_path / "s.md"))
    assert "Same-incident comparison" in (tmp_path / "s.md").read_text()
