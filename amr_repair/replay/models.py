"""Pydantic schemas and serialization models for the warehouse simulation replay."""

from typing import Any, List, Dict, Optional, Literal, Tuple
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
    # Generator settings, so a run can be re-simulated exactly (older replays lack them)
    disruption_rate: Optional[float] = None
    overlap_mode: Optional[str] = None
    max_block_size: Optional[int] = None
    cache_threshold: Optional[float] = None
    counterfactual_window: Optional[int] = None
    # Seconds of simulated time per step used to charge LLM thinking time to the robots
    # (decision delay = floor(latency / step)). None = thinking time was not charged.
    decision_step_seconds: Optional[float] = None
    # Strategy-cache entries loaded before the run (warm start from training seeds); 0 = empty cache
    cache_warm_entries: Optional[int] = None
    # "record" | "replay" when LLM answers went through a response store (llm/response_store.py)
    llm_response_store: Optional[str] = None


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
    # A stored strategy matched above the threshold. It is only *used* if its dry run passes
    # (see DryRunResult); the run-level cache_hits metric counts those verified hits.
    hit: bool
    matched: Optional[str] = None
    similarity: Optional[float] = None
    # Probe for the threshold curve: the most similar stored strategy in the same category,
    # whatever its similarity, and whether it passed a dry run on this incident.
    # {"similarity", "entry", "solver", "horizon", "valid"}. None = probe off or no candidate.
    probe: Optional[Dict[str, Any]] = None


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
    prompt_tokens: int = 0             # input tokens (summed over re-asks)
    completion_tokens: int = 0         # output tokens (summed over re-asks)
    api_requests: int = 0              # HTTP requests incl. Instructor schema re-asks
    error: Optional[str] = None
    directive: Optional[LLMDirective] = None
    returned_model: Optional[str] = None  # model version reported by the API response
    replayed: bool = False                # answer read from the response store (no API call made)


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
    # Steps the replanned robots waited for the LLM's answer: floor(LLM latency / step length)
    decision_delay_steps: int = 0
    overlap_kind: Optional[str] = None   # isolated | simultaneous | sequential | compound
    related_disruptions: List[Dict] = Field(default_factory=list)  # earlier active disruptions overlapped
    must_replan: List[int] = Field(default_factory=list)
    candidate_robots: List[int] = Field(default_factory=list)
    failed_robots: List[int] = Field(default_factory=list)
    executed_at: int
    paths: PathsData
    outcome: IncidentOutcome
    # Counterfactual branches from the detection state (loop/counterfactual.py):
    # {"window", "arms": {pibt|lns|windowed_pbs|executed: {...}}, "oracle", "matches_oracle",
    #  "goal_regret", "reproduced"}. None when the run was made without counterfactual_window.
    counterfactual: Optional[Dict[str, Any]] = None


class ReplayMetrics(BaseModel):
    # Core research performance rates
    repair_success_rate: float = 1.0
    first_attempt_success_rate: float = 1.0
    recovered_rate: float = 1.0
    mean_recovery_steps: float = 0.0
    cost_delta_pct: float = 0.0
    sum_of_costs_delta: int = 0
    max_recovery_steps: int = 0            # longest recovery of any incident in the run

    # Fleet-level (lifelong MAPF) metrics
    throughput: int = 0                    # goals completed during the run (total)
    throughput_per_step: float = 0.0       # goals completed per step (the lifelong-MAPF headline metric)
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
    llm_replayed_calls: int = 0            # answers read from the response store instead of the API
    decision_delay_steps: int = 0          # sum over incidents of the steps robots waited for the LLM

    # Cache Telemetry
    cache_lookups: int = 0
    cache_matches: int = 0                 # a stored strategy matched above the threshold
    cache_hits: int = 0                    # matches whose dry run passed (the strategy was used)
    cache_misses: int = 0
    cache_hit_rate_pct: float = 0.0        # hits / lookups
    cache_precision: Optional[float] = None  # hits / matches; None when nothing matched
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
    # Same-incident comparison of the executed decisions vs. fixed solvers, random choice,
    # single best solver and virtual best solver (loop/counterfactual.summarize). None = not measured.
    counterfactual: Optional[Dict[str, Any]] = None


class Replay(BaseModel):
    meta: ReplayMeta
    map: MapData
    frames: List[Frame]
    incidents: List[Incident]
    metrics: ReplayMetrics
    disruptions: List[DisruptionEvent] = Field(default_factory=list)
