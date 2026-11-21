"""Incident representation and feature extraction for the strategy cache and LLM."""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

SizeBucket = Literal["small", "medium", "large"]
CongestionLevel = Literal["low", "medium", "high"]
DisruptionType = Literal["blocked_cell", "robot_failure", "robot_delay"]


class DisruptionData(BaseModel):
    type: DisruptionType
    cells: List[Tuple[int, int]] = Field(default_factory=list)
    robot: Optional[int] = None
    k_steps: Optional[int] = None
    duration: Optional[int] = None  # temporary blockage / delay length; None = permanent


class CongestionData(BaseModel):
    level: CongestionLevel
    region: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x0, y0), (x1, y1))
    density: float


OverlapKind = Literal["isolated", "simultaneous", "sequential", "compound"]


class RelatedDisruption(BaseModel):
    """An earlier, still-active disruption this incident overlaps with."""
    incident_id: str
    t: int
    type: DisruptionType
    cells: List[Tuple[int, int]] = Field(default_factory=list)
    robot: Optional[int] = None


class RobotContext(BaseModel):
    """Per-robot facts the strategist may use (no map is ever sent)."""
    id: int
    role: Literal["must_replan", "conflicting", "delayed", "interacting", "nearby"]
    dist_to_goal: int
    delay_remaining: int = 0


class IncidentRecord(BaseModel):
    id: str
    t_detected: int
    disruptions: List[DisruptionData]
    # Robots involved in the incident (plans invalidated, or the disrupted robot itself)
    affected_robots: List[int]
    size_bucket: SizeBucket
    congestion: CongestionData
    interacts_with: List[str] = Field(default_factory=list)
    # Robots whose plan crosses a blocked cell / failed robot: they MUST be replanned
    must_replan: List[int] = Field(default_factory=list)
    # Robots the strategist may choose to replan (affected, non-failed, plus nearby)
    candidate_robots: List[int] = Field(default_factory=list)
    failed_robots: List[int] = Field(default_factory=list)
    delayed_robots: Dict[int, int] = Field(default_factory=dict)
    robot_context: List[RobotContext] = Field(default_factory=list)
    # isolated | simultaneous (same step) | sequential (overlaps an earlier active
    # disruption) | compound (3+ disruptions, or simultaneous AND sequential)
    overlap_kind: OverlapKind = "isolated"
    related_disruptions: List[RelatedDisruption] = Field(default_factory=list)

    def plannable_affected(self) -> List[int]:
        failed = set(self.failed_robots)
        return [r for r in self.affected_robots if r not in failed]

    def disruption_types(self) -> List[str]:
        return sorted({d.type for d in self.disruptions})

    def features(self) -> np.ndarray:
        """Fixed-length 12-d vector (kept for backwards compatibility):
        one-hot types x3, size bucket x3, congestion level x3,
        affected count / 10, interaction flag, density."""
        type_set = {d.type for d in self.disruptions}
        type_vec = [float(t in type_set) for t in ("blocked_cell", "robot_failure", "robot_delay")]
        size_vec = [float(self.size_bucket == s) for s in ("small", "medium", "large")]
        cong_vec = [float(self.congestion.level == c) for c in ("low", "medium", "high")]
        vec = type_vec + size_vec + cong_vec + [
            float(len(self.affected_robots)) / 10.0,
            1.0 if self.interacts_with else 0.0,
            float(np.clip(self.congestion.density, 0.0, 1.0)),
        ]
        return np.array(vec, dtype=np.float32)

    def soft_features(self) -> np.ndarray:
        """Continuous descriptors used for similarity *after* the hard filter
        (disruption types and the 1 / 2-3 / 4+ robot bucket are already equal by
        then; congestion and interaction are compared here).
        Each dimension is scaled to roughly [0, 1] so an RBF kernel on this
        vector gives thresholds that actually discriminate."""
        n_aff = len(self.plannable_affected())
        n_must = len(self.must_replan)
        n_delay = len(self.delayed_robots)
        return np.array([
            min(n_aff, 10) / 5.0,
            min(n_must, 10) / 5.0,
            1.0 if self.interacts_with else 0.0,
            min(len(self.disruptions), 4) / 2.0,
            min(n_delay, 4) / 2.0,
            float(np.clip(self.congestion.density * 5.0, 0.0, 1.0)),
        ], dtype=np.float32)
