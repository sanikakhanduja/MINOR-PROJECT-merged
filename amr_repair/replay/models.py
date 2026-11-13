"""Pydantic schemas and serialization models for the warehouse simulation replay."""

from typing import List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field


class ReplayMeta(BaseModel):
    seed: int
    mode: Literal["llm", "baseline_pibt", "baseline_lns", "baseline_pbs"]
    map_name: str
    steps: int
    run_id: Optional[str] = None
    num_robots: Optional[int] = None
    planner: Optional[str] = None          # background lifelong planner
    llm_provider: Optional[str] = None     # provider that answered ("heuristic" = no LLM)
    llm_model: Optional[str] = None
    cache_enabled: Optional[bool] = None
    validator_enabled: Optional[bool] = None


class MapData(BaseModel):
    width: int
    height: int
    obstacles: List[Tuple[int, int]]


RobotState = Literal["moving", "idle", "delayed", "failed", "replanning"]


class Frame(BaseModel):
    t: int
    pos: List[Tuple[int, int]]  # Indexed by robot id
    state: List[RobotState]     # Indexed by robot id
    goals: Optional[List[Tuple[int, int]]] = None  # Optional ground-truth goals


DisruptionType = Literal["blocked_cell", "robot_failure", "robot_delay"]


class Disruption(BaseModel):
    type: DisruptionType
    cells: List[Tuple[int, int]] = Field(default_factory=list)
    robot: Optional[int] = None
    k_steps: Optional[int] = None
    # Step at which the disruption stops being active (blocked cell cleared / delay over).
    # None for robot failures (permanent; follow frames[].state).
    active_until: Optional[int] = None


class DisruptionEvent(Disruption):
    """Every disruption that fired, including ones that affected no robot (and so
    produced no incident). Lets the viewer show the complete disruption picture."""
    t: int
    group: Optional[int] = None          # overlap group from the generator
    overlap_kind: Optional[str] = None   # generator label: isolated | same_aisle | same_robot | simultaneous | cascade
    incident_id: Optional[str] = None    # incident it was part of (None if it had no impact)
    impact: bool = False                 # True if it invalidated at least one robot's plan


CongestionLevel = Literal["low", "medium", "high"]


class Congestion(BaseModel):
    level: CongestionLevel
    region: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x0, y0), (x1, y1))
    density: Optional[float] = None


class CacheResult(BaseModel):
    hit: bool
    matched: Optional[str] = None
    similarity: Optional[float] = None


class DryRunResult(BaseModel):
    ran: bool
    valid: bool
    violation: Optional[str] = None


SolverType = Literal["pibt", "lns", "windowed_pbs"]


class LLMDirective(BaseModel):
    robots: List[int]
    solver: SolverType
    priority: List[int]
    horizon: int
    rationale: Optional[str] = None


class LLMCallLog(BaseModel):
    attempt: int                       # 1 = first choice, 2 = retry with violation
    provider: str                      # "groq" | "openai" | "ollama" | "custom" | "heuristic" ...
    model: Optional[str] = None
    latency_ms: float = 0.0
    tokens: int = 0
    api_requests: int = 0              # HTTP requests incl. Instructor schema re-asks
    error: Optional[str] = None
    directive: Optional[LLMDirective] = None


class LLMResult(BaseModel):
    # True only if a real LLM produced a directive for this incident
    called: bool
    latency_ms: Optional[int] = None   # summed over attempts
    tokens: Optional[int] = None       # summed over attempts
    directive: Optional[LLMDirective] = None  # first directive
    provider: Optional[str] = None
    calls: List[LLMCallLog] = Field(default_factory=list)


class ValidatorAttempt(BaseModel):
    attempt: int
    valid: bool
    violation: Optional[str] = None


class PathsData(BaseModel):
    invalidated: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)
    repaired: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)


class IncidentOutcome(BaseModel):
    # Steps until every repaired robot is making net progress again (reached its goal
    # or is closer to it than at detection). Censored at the end of the run.
    recovery_steps: int
    # Extra steps of the executed repair vs. each robot's single-agent shortest path
    # on the disrupted map (sum over robots, >= 0). Path-quality measure.
    cost_delta: int
    shortest_cost: int = 0
    recovered: bool = True


class Incident(BaseModel):
    id: str
    t_detected: int
    disruptions: List[Disruption]
    affected_robots: List[int]
    congestion: Congestion
    interacts_with: List[str] = Field(default_factory=list)
    cache: Optional[CacheResult] = None
    dry_run: Optional[DryRunResult] = None
    llm: Optional[LLMResult] = None
    validator: List[ValidatorAttempt] = Field(default_factory=list)
    fallback_used: bool = False
    # "cache" | "llm" | "llm_retry" | "heuristic" | "heuristic_retry" | "baseline" | "fallback"
    # | "emergency_hold" | "unvalidated"
    decision_source: Optional[str] = None
    executed_directive: Optional[LLMDirective] = None
    executed_valid: Optional[bool] = None
    overlap_kind: Optional[str] = None   # isolated | simultaneous | sequential | compound
    related_disruptions: List[Dict] = Field(default_factory=list)  # earlier active disruptions overlapped
    must_replan: List[int] = Field(default_factory=list)
    candidate_robots: List[int] = Field(default_factory=list)
    failed_robots: List[int] = Field(default_factory=list)
    executed_at: int
    paths: PathsData
    outcome: IncidentOutcome


class ReplayMetrics(BaseModel):
    # Core research performance rates
    repair_success_rate: float = 1.0
    first_attempt_success_rate: float = 1.0
    recovered_rate: float = 1.0
    mean_recovery_steps: float = 0.0
    cost_delta_pct: float = 0.0
    sum_of_costs_delta: int = 0
    makespan_delta: int = 0

    # Fleet-level (lifelong MAPF) metrics
    throughput: int = 0                    # goals completed during the run
    executed_vertex_collisions: int = 0    # audited from frames, must be 0
    executed_edge_swaps: int = 0           # audited from frames, must be 0
    guard_interventions: int = 0           # low-level safety stops (should be 0)
    background_replans: int = 0
    conflict_replans: int = 0

    # LLM Telemetry
    llm_provider: Optional[str] = None
    heuristic_decisions: int = 0
    llm_errors: int = 0
    llm_api_requests: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_calls_avoided: int = 0
    mean_llm_latency_ms: float = 0.0

    # Cache Telemetry
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate_pct: float = 0.0
    dry_run_successes: int = 0
    dry_run_failures: int = 0
    verified_cache_reuse_count: int = 0
    verified_reuse_rate_pct: float = 0.0
    cache_insertions: int = 0
    cache_evictions: int = 0

    # Latencies & Repair Timing
    mean_solver_latency_ms: float = 0.0
    mean_validator_latency_ms: float = 0.0
    mean_total_repair_latency_ms: float = 0.0

    # Safety & Validator Telemetry
    safety_violations_intercepted: int = 0
    executed_safety_violations: int = 0    # unvalidated invalid plans executed (no-validator ablation)
    failed_repair_attempts: int = 0
    fallback_invocations: int = 0
    emergency_holds: int = 0

    # Incident breakdown
    incidents_total: int = 0
    incidents_by_type: Dict[str, int] = Field(default_factory=dict)
    incidents_isolated: int = 0
    incidents_interacting: int = 0
    disruptions_without_impact: int = 0
    decision_sources: Dict[str, int] = Field(default_factory=dict)
    # success / recovery / cost per overlap kind (isolated vs simultaneous / sequential / compound)
    by_overlap_kind: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    solver_choices: Dict[str, int] = Field(default_factory=dict)


class Replay(BaseModel):
    meta: ReplayMeta
    map: MapData
    frames: List[Frame]
    incidents: List[Incident]
    metrics: ReplayMetrics
    disruptions: List[DisruptionEvent] = Field(default_factory=list)
