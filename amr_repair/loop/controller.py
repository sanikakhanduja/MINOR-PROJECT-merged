"""Closed-loop lifelong MAPF controller (stages 1-10 of the synopsis).

Per simulation step:

    1  simulator           World (rolling-horizon cooperative MAPF, collision-free)
    2  disruption injector Injector.apply
    3  sentinel            IncidentDetector.detect -> IncidentRecord
    4  cache lookup        StrategyCache.lookup (hard filter + similarity, minus negative memory)
    5  dry run             solver + validator on the live state; failure = cache miss
    6  LLM strategist      Strategist.choose -> Directive (Instructor + Pydantic).
                           Thinking time is charged to the robots: the robots being
                           replanned wait floor(latency / decision_step_seconds) steps
                           before their repair starts (decision_delay_steps).
    7  path repair         PIBT | LNS | windowed PBS on the chosen robots
    8  validator           validate(); one LLM retry with the exact violation,
                           then the fixed fallback (windowed PBS, horizon 30),
                           then an emergency hold
    9  execute + feedback  commit paths, write outcome / rejected strategies to cache
   10  metrics             logged per incident and aggregated per run

Baseline modes replace stages 4-6 with one fixed solver (same validator and,
by default, the same deterministic fallback, so the comparison isolates the
value of adaptive strategy selection).
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import amr_repair.solvers.lns as lns_solver
import amr_repair.solvers.pibt as pibt_solver
import amr_repair.solvers.windowed_pbs as pbs_solver
from amr_repair.cache.negative import NegativeMemory
from amr_repair.cache.store import StrategyCache
from amr_repair.disrupt.injector import Disruption, Injector
from amr_repair.llm.response_store import ResponseStore
from amr_repair.llm.schema import Directive
from amr_repair.llm.strategist import Strategist
from amr_repair.loop import counterfactual as cf
from amr_repair.loop.audit import audit_frames
from amr_repair.loop.logger import SimulationLogger
from amr_repair.replay.exporter import export_replay
from amr_repair.replay.models import (
    CacheResult, Congestion as ReplayCongestion, Disruption as ReplayDisruption, DisruptionEvent, DryRunResult,
    Incident, IncidentOutcome, LLMCallLog, LLMDirective, LLMResult, MapData, PathsData, Replay,
    ReplayMeta, ValidatorAttempt,
)
from amr_repair.sentinel.detect import IncidentDetector
from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.sim.agents import Robot
from amr_repair.sim.grid import load_map
from amr_repair.sim.lifelong import World
from amr_repair.validate.validator import Verdict, validate

logger = logging.getLogger(__name__)
Cell = Tuple[int, int]

SOLVER_REGISTRY = {
    "pibt": pibt_solver.solve,
    "lns": lns_solver.solve,
    "windowed_pbs": pbs_solver.solve,
}
BASELINE_SOLVER = {"baseline_pibt": "pibt", "baseline_lns": "lns", "baseline_pbs": "windowed_pbs"}


@dataclass
class SimulationConfig:
    map_path: str
    num_robots: int = 100
    steps: int = 300
    seed: int = 42
    mode: str = "llm"  # 'llm', 'baseline_pibt', 'baseline_lns', 'baseline_pbs'
    schedule: Optional[Union[List[Disruption], Dict[str, Any]]] = None
    disruption_rate: float = 0.05
    disruption_types: List[str] = field(default_factory=lambda: ["blocked_cell", "robot_failure", "robot_delay"])
    overlap_mode: str = "same_aisle"  # "none", "same_aisle", "same_robot"
    max_block_size: int = 3
    cache_enabled: bool = True
    cache_capacity: int = 500
    validator_enabled: bool = True
    cache_threshold: float = 0.85
    run_id: str = "demo"
    output_dir: str = "runs"
    cache_dir: str = "runs/cache"
    # Keep the cache on disk across runs (False = every run starts with an empty
    # cache, which is what reproducible experiments need).
    cache_persist: bool = False
    # LLM: provider None -> $LLM_PROVIDER or "auto"; require_llm makes a missing
    # LLM an error instead of silently using the heuristic.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    require_llm: bool = False
    llm_client: Any = None
    planning_window: int = 20
    baseline_fallback: bool = True
    fallback_horizon: int = 30
    export: bool = True
    # Counterfactual per-incident evaluation (loop/counterfactual.py): at every
    # incident, fork the world and roll each fixed solver and the executed
    # decision forward this many steps. 0 = off.
    counterfactual_window: int = 0
    # Re-simulate a finished run: {incident id: executed directive dict, or None for an
    # emergency hold}. Each incident executes exactly its recorded directive; the cache and
    # strategist are bypassed, so no LLM is called. Used to add counterfactual analysis to
    # runs that were recorded without it (scripts/counterfactual_from_replay.py).
    recorded_decisions: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    # Simulated seconds per step for charging the strategist's thinking time to the robots
    # (decision delay = floor(latency / step)): a decision faster than one step is free, as
    # in the League of Robot Runners' 1 s per step planning budget. None = not charged.
    decision_step_seconds: Optional[float] = 1.0
    # Warm start: load strategy-cache entries from this file before the run (entries filled
    # on separate training seeds). Export: write the final cache to this file.
    cache_warm_path: Optional[str] = None
    cache_export_path: Optional[str] = None
    # Threshold curve: at every incident, dry-run the most similar stored strategy whatever
    # its similarity, and record the result in incident.cache.probe. Costs one extra solver
    # call per incident; not included in the repair latency.
    cache_probe: bool = False
    # Record LLM answers to this JSON-lines file and reuse them (mode "record"), or only reuse
    # them and never call the API (mode "replay"). See llm/response_store.py.
    llm_response_store: Optional[str] = None
    llm_response_mode: str = "record"


class SimulationController:
    def __init__(self, config: SimulationConfig):
        if config.mode not in ("llm", *BASELINE_SOLVER):
            raise ValueError(f"Unknown mode {config.mode}")
        if config.counterfactual_window and not config.validator_enabled:
            raise ValueError("counterfactual_window needs the validator: every arm is validated the same way")
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # 1. Map, robots and world. Robot k always starts at the same cell and
        #    receives the same task stream for a given seed, in every mode.
        self.grid_map = load_map(config.map_path)
        self._static_obstacles = self.grid_map.get_obstacle_cells()  # shelves only, before disruptions
        free_cells = self.grid_map.get_free_cells()
        n_robots = min(config.num_robots, len(free_cells))
        starts = self.rng.choice(len(free_cells), size=n_robots, replace=False)
        robots = [Robot(id=i, pos=free_cells[idx], goal=free_cells[idx]) for i, idx in enumerate(starts)]
        self.world = World(self.grid_map, robots, self.rng,
                           planning_window=config.planning_window, goal_seed=config.seed)

        # 2. Injector (own RNG stream) and sentinel
        inj_rng = np.random.default_rng([config.seed, 7])
        if config.schedule is None:
            schedule = {"rate": config.disruption_rate, "types": config.disruption_types,
                        "overlap": config.overlap_mode, "max_steps": config.steps,
                        "max_block_size": config.max_block_size}
        else:
            schedule = config.schedule
        self.injector = Injector(rng=inj_rng, schedule=schedule, world=self.world)
        self.detector = IncidentDetector()

        # 3. Strategy cache + negative memory
        persist = config.cache_enabled and config.cache_persist
        self.negative_memory = NegativeMemory(
            filepath=os.path.join(config.cache_dir, "negative_memory.json") if persist else None)
        self.strategy_cache: Optional[StrategyCache] = StrategyCache(
            threshold=config.cache_threshold, max_capacity=config.cache_capacity,
            negative_memory=self.negative_memory,
            filepath=os.path.join(config.cache_dir, "strategy_cache.json") if persist else None,
            namespace=f"{config.run_id}-s{config.seed}",
        ) if config.cache_enabled else None
        self.cache_warm_entries = 0
        if self.strategy_cache is not None and config.cache_warm_path:
            self.cache_warm_entries = self.strategy_cache.load_entries(config.cache_warm_path)

        # 4. Strategist (LLM mode only) and logger
        self.strategist: Optional[Strategist] = None
        if config.mode == "llm":
            store = (ResponseStore(config.llm_response_store, config.llm_response_mode)
                     if config.llm_response_store else None)
            self.strategist = Strategist(provider=config.llm_provider, model=config.llm_model,
                                         client=config.llm_client, allow_heuristic=not config.require_llm,
                                         response_store=store)
        self.logger = SimulationLogger()
        self._pending_recovery: List[Dict[str, Any]] = []
        self._events: List[DisruptionEvent] = []
        self._counterfactuals: List[Dict[str, Any]] = []

    # ================================================================== run
    def run(self) -> Replay:
        for t in range(self.config.steps):
            fired = self.injector.apply(self.world, t)
            incidents = self.detector.detect(self.world, fired, t)
            logged_before = {i.id for i in self.logger.incidents}
            for inc in incidents:
                self._handle_incident(inc, t)
            logged = {i.id for i in self.logger.incidents} - logged_before
            self._record_events(fired, incidents, logged, t)
            self.world.step()
            self._update_recovery()
        self._finish_recovery()
        if self.strategy_cache is not None and self.config.cache_export_path:
            self.strategy_cache.export(self.config.cache_export_path)
        replay = self._build_replay()
        if self.config.export:
            export_replay(replay, os.path.join(self.config.output_dir, f"{self.config.run_id}.json"))
        return replay

    # ============================================================ incidents
    def _handle_incident(self, incident: IncidentRecord, t: int) -> None:
        cfg = self.config
        required = incident.plannable_affected()
        if not required:
            self.logger.disruptions_without_impact += 1
            return
        # Snapshot before anything touches the world, for the counterfactual branches
        snapshot = cf.fork_world(self.world) if cfg.counterfactual_window else None
        unblocks = {k: list(v) for k, v in self.injector._unblock_schedule.items()} if snapshot else {}
        t_start = time.perf_counter()
        robots = {r.id: r for r in self.world.robots}
        invalidated = {str(r): [list(c) for c in robots[r].path[:20]] for r in required}

        rec: Dict[str, Any] = dict(cache=None, dry=None, llm_calls=[], attempts=[], fallback=False,
                                   solver_ms=0.0, validator_ms=0.0, probe_ms=0.0, delay=0)

        def attempt(directive: Directive) -> Tuple[Directive, Dict[int, List[Cell]], Verdict]:
            d = self._sanitize(directive, incident)
            t0 = time.perf_counter()
            paths = SOLVER_REGISTRY[d.solver](self.world, d.robots, d.priority, d.horizon)
            rec["solver_ms"] += (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            verdict = validate(paths, self.world, t, required=required)
            rec["validator_ms"] += (time.perf_counter() - t1) * 1000
            rec["attempts"].append(ValidatorAttempt(attempt=len(rec["attempts"]) + 1,
                                                    valid=verdict.valid, violation=verdict.violation))
            return d, paths, verdict

        chosen: Optional[Tuple[str, Directive, Dict[int, List[Cell]], bool]] = None  # source, d, paths, valid

        if cfg.recorded_decisions is not None:
            if incident.id not in cfg.recorded_decisions:
                raise RuntimeError(f"Re-simulation diverged: {incident.id} at t={t} is not in the recording")
            rd = cfg.recorded_decisions[incident.id]
            if rd is None:
                chosen = None
            else:
                self._charge(required, int(rd.get("decision_delay_steps") or 0), rec)
                d, paths, v = attempt(Directive(**{k: rd[k] for k in ("robots", "solver", "priority", "horizon")},
                                                rationale=(rd.get("rationale") or "")[:200]))
                chosen = ("recorded", d, paths, v.valid)
        elif cfg.mode in BASELINE_SOLVER:
            d, paths, v = attempt(self._fixed_directive(required, BASELINE_SOLVER[cfg.mode], 20, "Fixed baseline"))
            if v.valid:
                chosen = ("baseline", d, paths, True)
            elif not cfg.validator_enabled:
                chosen = ("unvalidated", d, paths, False)
            elif not cfg.baseline_fallback:
                chosen = None
            else:
                chosen = self._fallback(required, attempt, rec, incident)
        else:
            chosen = self._llm_pipeline(incident, required, attempt, rec)

        # ---------------------------------------------------------- execute
        if chosen is None:
            source, directive, paths, valid = "emergency_hold", None, {}, False
            for rid in required:
                r = robots[rid]
                r.path = [r.pos] * r.delay_remaining  # stop in place; world replans around it
        else:
            source, directive, paths, valid = chosen
            horizon_end = t + max((len(p) for p in paths.values()), default=0)
            for rid, p in paths.items():
                r = robots[rid]
                r.path = list(p)
                r.repair_until = t + len(p)
                if not valid:
                    r.unvalidated_until = t + len(p)
            self.detector.register_repair(incident, list(paths), horizon_end)

        cost, shortest = cf.detour_cost(self.world, paths if paths else {r: [robots[r].pos] for r in required})
        outcome = IncidentOutcome(recovery_steps=0, cost_delta=cost, shortest_cost=shortest, recovered=False)
        self._pending_recovery.append({
            "outcome": outcome, "t0": t, "robots": cf.recovery_baseline(self.world, required),
        })

        llm_result = self._llm_result(rec["llm_calls"])
        replay_incident = Incident(
            id=incident.id, t_detected=incident.t_detected,
            disruptions=[ReplayDisruption(type=d.type, cells=d.cells, robot=d.robot, k_steps=d.k_steps,
                                          active_until=self._active_until(d.type, t, d.duration, d.k_steps))
                         for d in incident.disruptions],
            affected_robots=incident.affected_robots,
            congestion=ReplayCongestion(level=incident.congestion.level, region=incident.congestion.region,
                                        density=incident.congestion.density),
            interacts_with=incident.interacts_with,
            overlap_kind=incident.overlap_kind,
            related_disruptions=[d.model_dump() for d in incident.related_disruptions],
            cache=rec["cache"], dry_run=rec["dry"], llm=llm_result,
            validator=rec["attempts"], fallback_used=rec["fallback"],
            decision_source=source,
            decision_delay_steps=rec["delay"],
            executed_directive=_to_llm_directive(directive) if directive else None,
            executed_valid=valid,
            must_replan=incident.must_replan, candidate_robots=incident.candidate_robots,
            failed_robots=incident.failed_robots,
            executed_at=t,
            paths=PathsData(invalidated=invalidated,
                            repaired={str(k): [list(c) for c in v] for k, v in paths.items()}),
            outcome=outcome,
        )
        self.logger.record_incident(replay_incident, telemetry={
            "solver_latency_ms": round(rec["solver_ms"], 2),
            "validator_latency_ms": round(rec["validator_ms"], 2),
            "total_repair_latency_ms": round((time.perf_counter() - t_start) * 1000.0 - rec["probe_ms"], 2),
        })

        # Counterfactual branches (not part of the repair latency above)
        if snapshot is not None:
            ev = cf.evaluate_incident(snapshot, incident, required, directive, paths, SOLVER_REGISTRY,
                                      unblocks, cfg.counterfactual_window, cfg.fallback_horizon,
                                      decision_delay=rec["delay"])
            replay_incident.counterfactual = ev
            self._counterfactuals.append(ev)

    def _active_until(self, dtype: str, t: int, duration: Optional[int], k_steps: Optional[int]) -> Optional[int]:
        if dtype == "robot_failure":
            return None
        if dtype == "robot_delay":
            return t + int(k_steps or duration or 0)
        return t + int(duration) if duration else self.config.steps  # permanent blockage

    def _record_events(self, fired, incidents, logged, t: int) -> None:
        for d in fired:
            owner = next((inc for inc in incidents for x in inc.disruptions
                          if x.type == d.type and x.robot == d.robot and list(x.cells) == list(d.cells)), None)
            self._events.append(DisruptionEvent(
                t=t, type=d.type, cells=d.cells, robot=d.robot, k_steps=d.k_steps,
                active_until=self._active_until(d.type, t, d.duration, d.k_steps),
                group=d.group, overlap_kind=d.overlap_kind,
                incident_id=owner.id if owner is not None and owner.id in logged else None,
                impact=owner is not None and owner.id in logged,
            ))

    # --------------------------------------------------------- LLM pipeline
    def _llm_pipeline(self, incident, required, attempt, rec):
        cfg = self.config
        cache = self.strategy_cache

        # Stage 4-5: cache lookup + dry run
        if cache is not None:
            probe = self._probe(incident, required, rec) if cfg.cache_probe else None
            hit = cache.lookup(incident, world=self.world)
            if hit is None:
                rec["cache"] = CacheResult(hit=False, probe=probe)
            else:
                cached, matched, sim = hit
                rec["cache"] = CacheResult(hit=True, matched=matched, similarity=round(sim, 3), probe=probe)
                if not cfg.validator_enabled:
                    d, paths, v = attempt(cached)
                    return ("cache", d, paths, v.valid) if v.valid else ("unvalidated", d, paths, False)
                d, paths, v = attempt(cached)
                rec["dry"] = DryRunResult(ran=True, valid=v.valid, violation=v.violation)
                if v.valid:
                    cache.record_verified_reuse(matched)
                    return ("cache", d, paths, True)
                cache.record_failed_reuse(matched)
                self.negative_memory.add(incident, d, v.violation)

        # Stage 6-8: LLM choice, validation, one retry with the violation
        is_llm = self.strategist.uses_llm
        rejected = self.negative_memory.rejected_for(incident)
        d1 = self.strategist.choose(incident, rejected=rejected)
        rec["llm_calls"].append(self._call_log(1))
        self._charge_thinking(required, rec)
        d1, paths1, v1 = attempt(d1)
        if not cfg.validator_enabled:
            return ("llm" if is_llm else "heuristic", d1, paths1, True) if v1.valid else ("unvalidated", d1, paths1, False)
        if v1.valid:
            self._remember_success(incident, d1, "llm" if is_llm else "heuristic")
            return ("llm" if is_llm else "heuristic", d1, paths1, True)
        self.negative_memory.add(incident, d1, v1.violation)

        d2 = self.strategist.choose(incident, violation=v1.violation,
                                    rejected=self.negative_memory.rejected_for(incident), previous=d1)
        rec["llm_calls"].append(self._call_log(2))
        self._charge_thinking(required, rec)
        d2, paths2, v2 = attempt(d2)
        if v2.valid:
            self._remember_success(incident, d2, "llm" if is_llm else "heuristic")
            return ("llm_retry" if is_llm else "heuristic_retry", d2, paths2, True)
        self.negative_memory.add(incident, d2, v2.violation)

        return self._fallback(required, attempt, rec, incident)

    def _fallback(self, required, attempt, rec, incident):
        """Fixed deterministic fallback: windowed PBS on all affected robots,
        horizon 30, nearest-to-goal first. Bounded: never loops."""
        rec["fallback"] = True
        d = self._fixed_directive(required, "windowed_pbs", self.config.fallback_horizon,
                                  "Fixed deterministic fallback after failed attempts.")
        d, paths, v = attempt(d)
        if v.valid:
            if self.config.mode == "llm":
                self._remember_success(incident, d, "fallback")
            return ("fallback", d, paths, True)
        return None

    def _remember_success(self, incident, directive, source):
        self.negative_memory.clear(incident, directive)
        if self.strategy_cache is not None:
            self.strategy_cache.store(incident, directive, source=source)

    # ------------------------------------------------------------- helpers
    def _charge(self, required: List[int], steps: int, rec: Dict[str, Any]) -> None:
        """The replanned robots wait ``steps`` steps for the decision (time passes while it is made)."""
        if steps > rec["delay"]:
            cf.charge_delay(self.world, required, steps)
            rec["delay"] = steps

    def _charge_thinking(self, required: List[int], rec: Dict[str, Any]) -> None:
        """Charge the strategist's total thinking time so far in this incident."""
        step_s = self.config.decision_step_seconds
        if not step_s:
            return
        latency_ms = sum(c.latency_ms for c in rec["llm_calls"])
        self._charge(required, int(math.floor(latency_ms / (step_s * 1000.0))), rec)

    def _probe(self, incident: IncidentRecord, required: List[int], rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dry-run the most similar stored strategy at any similarity (threshold curve)."""
        found = self.strategy_cache.probe(incident, world=self.world)
        if found is None:
            return None
        d, entry, sim = found
        t0 = time.perf_counter()
        d = self._sanitize(d, incident)
        paths = SOLVER_REGISTRY[d.solver](self.world, d.robots, d.priority, d.horizon)
        v = validate(paths, self.world, self.world.t, required=required)
        rec["probe_ms"] += (time.perf_counter() - t0) * 1000
        return {"similarity": round(sim, 4), "entry": entry, "solver": d.solver, "horizon": d.horizon,
                "valid": v.valid}

    def _fixed_directive(self, required, solver, horizon, why) -> Directive:
        return cf.fixed_directive(self.world, required, solver, horizon, why)

    def _sanitize(self, d: Directive, incident: IncidentRecord) -> Directive:
        return cf.sanitize(self.world, d, incident)

    def _call_log(self, attempt_no: int) -> LLMCallLog:
        c = self.strategist.last_call
        return LLMCallLog(attempt=attempt_no, provider=c.provider, model=c.model,
                          latency_ms=c.latency_ms, tokens=c.total_tokens,
                          prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens,
                          api_requests=c.api_requests, returned_model=c.returned_model, replayed=c.replayed,
                          error=c.error, directive=_to_llm_directive(c.directive) if c.directive else None)

    def _llm_result(self, calls: List[LLMCallLog]) -> Optional[LLMResult]:
        if not calls:
            return None
        real = [c for c in calls if c.provider not in ("heuristic",) and not c.provider.startswith("heuristic(")]
        return LLMResult(
            called=bool(real),
            latency_ms=int(round(sum(c.latency_ms for c in calls))),
            tokens=sum(c.tokens for c in calls),
            directive=calls[0].directive,
            provider=calls[0].provider,
            calls=calls,
        )

    def _update_recovery(self) -> None:
        still = []
        for item in self._pending_recovery:
            if cf.progress_resumed(self.world, item["robots"]):
                item["outcome"].recovery_steps = self.world.t - item["t0"]
                item["outcome"].recovered = True
            else:
                still.append(item)
        self._pending_recovery = still

    def _finish_recovery(self) -> None:
        for item in self._pending_recovery:
            item["outcome"].recovery_steps = self.world.t - item["t0"]
            item["outcome"].recovered = False
        self._pending_recovery = []

    def _build_replay(self) -> Replay:
        frames = list(self.world.frames)
        vertex, swaps, teleports = audit_frames(frames)
        if vertex or swaps or teleports:
            logger.error("Executed collisions detected: vertex=%d swaps=%d teleports=%d", vertex, swaps, teleports)
        s = self.strategist
        cache = self.strategy_cache
        metrics = self.logger.compute_metrics(extras={
            "throughput": self.world.goals_completed,
            "throughput_per_step": round(self.world.goals_completed / max(1, self.world.t), 4),
            "llm_replayed_calls": s.replayed_calls if s else 0,
            "decision_delay_steps": sum(i.decision_delay_steps for i in self.logger.incidents),
            "executed_vertex_collisions": vertex,
            "executed_edge_swaps": swaps,
            "guard_interventions": self.world.guard_interventions,
            "background_replans": self.world.background_replans,
            "conflict_replans": self.world.conflict_replans,
            "llm_provider": (s.provider if s else None),
            "llm_calls": s.total_calls if s else 0,
            "llm_api_requests": s.total_api_requests if s else 0,
            "llm_tokens": s.total_tokens if s else 0,
            "heuristic_decisions": s.heuristic_decisions if s else 0,
            "llm_errors": s.llm_errors if s else 0,
            "cache_lookups": cache.lookups if cache else 0,
            "cache_matches": cache.matches if cache else 0,
            "cache_hits": cache.hits if cache else 0,
            "cache_misses": cache.misses if cache else 0,
            "cache_insertions": cache.insertions if cache else 0,
            "cache_evictions": cache.evictions if cache else 0,
            "counterfactual": cf.summarize(self._counterfactuals) if self.config.counterfactual_window else None,
        })
        return Replay(
            meta=ReplayMeta(
                seed=self.config.seed, mode=self.config.mode,  # type: ignore[arg-type]
                map_name=os.path.basename(self.config.map_path).replace(".map", ""),
                steps=len(frames), run_id=self.config.run_id, num_robots=len(self.world.robots),
                planner=f"windowed cooperative A* (window={self.config.planning_window})",
                llm_provider=(s.provider if s else None), llm_model=(s.model if s else None),
                cache_enabled=self.config.cache_enabled, validator_enabled=self.config.validator_enabled,
                disruption_rate=self.config.disruption_rate, overlap_mode=self.config.overlap_mode,
                max_block_size=self.config.max_block_size, cache_threshold=self.config.cache_threshold,
                counterfactual_window=self.config.counterfactual_window or None,
                decision_step_seconds=self.config.decision_step_seconds,
                cache_warm_entries=self.cache_warm_entries if self.config.cache_enabled else None,
                llm_response_store=self.config.llm_response_mode if self.config.llm_response_store else None,
            ),
            map=MapData(width=self.grid_map.width, height=self.grid_map.height,
                        obstacles=self._static_obstacles),
            frames=frames, incidents=self.logger.incidents, metrics=metrics, disruptions=self._events,
        )


def _to_llm_directive(d: Directive) -> LLMDirective:
    return LLMDirective(robots=d.robots, solver=d.solver, priority=d.priority,
                        horizon=d.horizon, rationale=d.rationale)


def run(config: SimulationConfig) -> Replay:
    """Convenience function to run a simulation with the given configuration."""
    return SimulationController(config).run()
