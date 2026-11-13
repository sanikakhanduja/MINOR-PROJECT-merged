"""Per-incident log and run-level metric aggregation."""

from typing import Any, Dict, List, Optional

import numpy as np

from amr_repair.replay.models import Incident, ReplayMetrics


class SimulationLogger:
    def __init__(self):
        self.incidents: List[Incident] = []
        self.solver_latencies: List[float] = []
        self.validator_latencies: List[float] = []
        self.total_repair_latencies: List[float] = []
        self.disruptions_without_impact: int = 0

    def record_incident(self, incident: Incident, telemetry: Optional[Dict[str, Any]] = None) -> None:
        self.incidents.append(incident)
        telemetry = telemetry or {}
        if "solver_latency_ms" in telemetry:
            self.solver_latencies.append(float(telemetry["solver_latency_ms"]))
        if "validator_latency_ms" in telemetry:
            self.validator_latencies.append(float(telemetry["validator_latency_ms"]))
        if "total_repair_latency_ms" in telemetry:
            self.total_repair_latencies.append(float(telemetry["total_repair_latency_ms"]))

    def compute_metrics(self, extras: Optional[Dict[str, Any]] = None) -> ReplayMetrics:
        extras = dict(extras or {})
        incs = self.incidents
        n = len(incs)

        def mean(xs):
            return round(float(np.mean(xs)), 2) if xs else 0.0

        by_type: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        solvers: Dict[str, int] = {}
        for inc in incs:
            for d in inc.disruptions:
                by_type[d.type] = by_type.get(d.type, 0) + 1
            src = inc.decision_source or "unknown"
            sources[src] = sources.get(src, 0) + 1
            if inc.executed_directive is not None:
                solvers[inc.executed_directive.solver] = solvers.get(inc.executed_directive.solver, 0) + 1

        def ok(i: Incident) -> bool:
            if i.executed_valid is not None:
                return bool(i.executed_valid) and i.decision_source != "emergency_hold"
            return any(v.valid for v in i.validator)  # incidents logged without decision info

        success = sum(1 for i in incs if ok(i))
        first_ok = sum(1 for i in incs if i.validator and i.validator[0].valid)
        recovered = sum(1 for i in incs if i.outcome.recovered)
        cost = sum(i.outcome.cost_delta for i in incs)
        shortest = sum(i.outcome.shortest_cost for i in incs)
        dry_ok = sum(1 for i in incs if i.dry_run and i.dry_run.ran and i.dry_run.valid)
        dry_bad = sum(1 for i in incs if i.dry_run and i.dry_run.ran and not i.dry_run.valid)
        cache_reuse = sources.get("cache", 0)
        intercepted = sum(1 for i in incs for v in i.validator if not v.valid)
        llm_lat = [c.latency_ms for i in incs if i.llm for c in i.llm.calls
                   if c.error is None and not c.provider.startswith("heuristic")]
        by_kind: Dict[str, Dict[str, float]] = {}
        for kind in sorted({i.overlap_kind or "isolated" for i in incs}):
            group = [i for i in incs if (i.overlap_kind or "isolated") == kind]
            g_short = sum(i.outcome.shortest_cost for i in group)
            by_kind[kind] = {
                "incidents": len(group),
                "disruptions": sum(len(i.disruptions) for i in group),
                "success_rate": round(sum(1 for i in group if ok(i)) / len(group), 3),
                "first_attempt_success_rate": round(
                    sum(1 for i in group if i.validator and i.validator[0].valid) / len(group), 3),
                "fallback_rate": round(sum(1 for i in group if i.fallback_used) / len(group), 3),
                "mean_recovery_steps": mean([i.outcome.recovery_steps for i in group]),
                "cost_delta_pct": round(100.0 * sum(i.outcome.cost_delta for i in group) / g_short, 2) if g_short else 0.0,
            }
        lookups = int(extras.pop("cache_lookups", 0))
        hits = int(extras.pop("cache_hits", 0))

        return ReplayMetrics(
            repair_success_rate=round(success / n, 3) if n else 1.0,
            first_attempt_success_rate=round(first_ok / n, 3) if n else 1.0,
            recovered_rate=round(recovered / n, 3) if n else 1.0,
            mean_recovery_steps=mean([i.outcome.recovery_steps for i in incs]),
            cost_delta_pct=round(100.0 * cost / shortest, 2) if shortest else 0.0,
            sum_of_costs_delta=int(cost),
            makespan_delta=int(max([i.outcome.recovery_steps for i in incs], default=0)),
            llm_calls_avoided=cache_reuse,
            mean_llm_latency_ms=mean(llm_lat),
            cache_lookups=lookups,
            cache_hits=hits,
            cache_misses=int(extras.pop("cache_misses", 0)),
            cache_hit_rate_pct=round(100.0 * hits / lookups, 1) if lookups else 0.0,
            dry_run_successes=dry_ok,
            dry_run_failures=dry_bad,
            verified_cache_reuse_count=cache_reuse,
            verified_reuse_rate_pct=round(100.0 * cache_reuse / lookups, 1) if lookups else 0.0,
            mean_solver_latency_ms=mean(self.solver_latencies),
            mean_validator_latency_ms=mean(self.validator_latencies),
            mean_total_repair_latency_ms=mean(self.total_repair_latencies),
            safety_violations_intercepted=intercepted,
            executed_safety_violations=sources.get("unvalidated", 0),
            failed_repair_attempts=sum(1 for i in incs if i.validator and not i.validator[0].valid),
            fallback_invocations=sum(1 for i in incs if i.fallback_used),
            emergency_holds=sources.get("emergency_hold", 0),
            incidents_total=n,
            incidents_by_type=by_type,
            incidents_isolated=sum(1 for i in incs if not i.interacts_with),
            incidents_interacting=sum(1 for i in incs if i.interacts_with),
            disruptions_without_impact=self.disruptions_without_impact,
            decision_sources=sources,
            by_overlap_kind=by_kind,
            solver_choices=solvers,
            **extras,
        )
