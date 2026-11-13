"""Closed-loop lifelong MAPF controller (stages 1-10 of the synopsis).

Per simulation step:

    1  simulator           World (rolling-horizon cooperative MAPF, collision-free)
    2  disruption injector Injector.apply
    3  sentinel            IncidentDetector.detect -> IncidentRecord
    4  cache lookup        StrategyCache.lookup (hard filter + similarity, minus negative memory)
    5  dry run             solver + validator on the live state; failure = cache miss
    6  LLM strategist      Strategist.choose -> Directive (Instructor + Pydantic)
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
from amr_repair.llm.schema import Directive
from amr_repair.llm.strategist import Strategist
from amr_repair.loop.audit import audit_frames
from amr_repair.loop.logger import SimulationLogger
from amr_repair.replay.exporter import export_replay
from amr_repair.replay.models import (
    CacheResult, Congestion as ReplayCongestion, Disruption as ReplayDisruption, DryRunResult,
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


class SimulationController:
    def __init__(self, config: SimulationConfig):
        if config.mode not in ("llm", *BASELINE_SOLVER):
            raise ValueError(f"Unknown mode {config.mode}")
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

        # 4. Strategist (LLM mode only) and logger
        self.strategist: Optional[Strategist] = None
        if config.mode == "llm":
            self.strategist = Strategist(provider=config.llm_provider, model=config.llm_model,
                                         client=config.llm_client, allow_heuristic=not config.require_llm)
        self.logger = SimulationLogger()
        self._pending_recovery: List[Dict[str, Any]] = []

    # ================================================================== run
    def run(self) -> Replay:
        for t in range(self.config.steps):
            fired = self.injector.apply(self.world, t)
            incidents = self.detector.detect(self.world, fired, t)
            for inc in incidents:
                self._handle_incident(inc, t)
            self.world.step()
            self._update_recovery()
        self._finish_recovery()
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
        t_start = time.perf_counter()
        robots = {r.id: r for r in self.world.robots}
        invalidated = {str(r): [list(c) for c in robots[r].path[:20]] for r in required}

        rec: Dict[str, Any] = dict(cache=None, dry=None, llm_calls=[], attempts=[], fallback=False,
                                   solver_ms=0.0, validator_ms=0.0)

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

        if cfg.mode in BASELINE_SOLVER:
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

        cost, shortest = self._detour_cost(paths if paths else {r: [robots[r].pos] for r in required})
        outcome = IncidentOutcome(recovery_steps=0, cost_delta=cost, shortest_cost=shortest, recovered=False)
        self._pending_recovery.append({
            "outcome": outcome, "t0": t,
            "robots": {rid: (self.world.map.distance(robots[rid].pos, robots[rid].goal), robots[rid].goals_completed)
                       for rid in required},
        })

        llm_result = self._llm_result(rec["llm_calls"])
        replay_incident = Incident(
            id=incident.id, t_detected=incident.t_detected,
            disruptions=[ReplayDisruption(type=d.type, cells=d.cells, robot=d.robot, k_steps=d.k_steps)
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
            "total_repair_latency_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        })

    # --------------------------------------------------------- LLM pipeline
    def _llm_pipeline(self, incident, required, attempt, rec):
        cfg = self.config
        cache = self.strategy_cache

        # Stage 4-5: cache lookup + dry run
        if cache is not None:
            hit = cache.lookup(incident, world=self.world)
            if hit is None:
                rec["cache"] = CacheResult(hit=False)
            else:
                cached, matched, sim = hit
                rec["cache"] = CacheResult(hit=True, matched=matched, similarity=round(sim, 3))
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
    def _fixed_directive(self, required, solver, horizon, why) -> Directive:
        robots = {r.id: r for r in self.world.robots}
        prio = sorted(required, key=lambda r: (self.world.map.distance(robots[r].pos, robots[r].goal), r))
        return Directive(robots=sorted(required), solver=solver, priority=prio, horizon=horizon, rationale=why)

    def _sanitize(self, d: Directive, incident: IncidentRecord) -> Directive:
        """Never let a directive touch failed or non-existent robots."""
        alive = {r.id for r in self.world.robots if r.state != "failed"}
        allowed = alive & (set(incident.candidate_robots) | set(incident.plannable_affected()))
        robots = [r for r in d.robots if r in allowed] or sorted(set(incident.plannable_affected()) & alive)
        prio = [r for r in d.priority if r in robots] + [r for r in robots if r not in d.priority]
        if robots == d.robots and prio == d.priority:
            return d
        return Directive(robots=robots, solver=d.solver, priority=prio, horizon=d.horizon, rationale=d.rationale)

    def _call_log(self, attempt_no: int) -> LLMCallLog:
        c = self.strategist.last_call
        return LLMCallLog(attempt=attempt_no, provider=c.provider, model=c.model,
                          latency_ms=c.latency_ms, tokens=c.total_tokens, api_requests=c.api_requests,
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

    def _detour_cost(self, paths: Dict[int, List[Cell]]) -> Tuple[int, int]:
        robots = {r.id: r for r in self.world.robots}
        extra = shortest = 0
        for rid, p in paths.items():
            r = robots[rid]
            d0 = self.world.map.distance(r.pos, r.goal)
            if d0 >= 10_000:
                continue
            used = (p.index(r.goal) + 1) if r.goal in p else len(p) + self.world.map.distance(p[-1], r.goal)
            extra += max(0, used - d0)
            shortest += d0
        return int(extra), int(shortest)

    def _update_recovery(self) -> None:
        robots = {r.id: r for r in self.world.robots}
        still = []
        for item in self._pending_recovery:
            done = True
            for rid, (d0, gc0) in item["robots"].items():
                r = robots[rid]
                if r.state == "failed" or r.goals_completed > gc0:
                    continue
                if self.world.map.distance(r.pos, r.goal) >= d0:
                    done = False
                    break
            if done:
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
            "cache_hits": cache.hits if cache else 0,
            "cache_misses": cache.misses if cache else 0,
            "cache_insertions": cache.insertions if cache else 0,
            "cache_evictions": cache.evictions if cache else 0,
        })
        return Replay(
            meta=ReplayMeta(
                seed=self.config.seed, mode=self.config.mode,  # type: ignore[arg-type]
                map_name=os.path.basename(self.config.map_path).replace(".map", ""),
                steps=len(frames), run_id=self.config.run_id, num_robots=len(self.world.robots),
                planner=f"windowed cooperative A* (window={self.config.planning_window})",
                llm_provider=(s.provider if s else None), llm_model=(s.model if s else None),
                cache_enabled=self.config.cache_enabled, validator_enabled=self.config.validator_enabled,
            ),
            map=MapData(width=self.grid_map.width, height=self.grid_map.height,
                        obstacles=self._static_obstacles),
            frames=frames, incidents=self.logger.incidents, metrics=metrics,
        )


def _to_llm_directive(d: Directive) -> LLMDirective:
    return LLMDirective(robots=d.robots, solver=d.solver, priority=d.priority,
                        horizon=d.horizon, rationale=d.rationale)


def run(config: SimulationConfig) -> Replay:
    """Convenience function to run a simulation with the given configuration."""
    return SimulationController(config).run()
