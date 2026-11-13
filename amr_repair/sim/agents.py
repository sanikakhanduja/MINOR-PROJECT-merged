"""Robot agent representation for lifelong MAPF simulation."""

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

RobotState = Literal["moving", "idle", "delayed", "failed", "replanning"]


@dataclass
class Robot:
    id: int
    pos: Tuple[int, int]
    goal: Tuple[int, int]
    # Committed plan: cells occupied at t+1, t+2, ... (see amr_repair.sim.planning)
    path: List[Tuple[int, int]] = field(default_factory=list)
    state: RobotState = "idle"
    delay_remaining: int = 0
    # Absolute time until which the robot follows a repair plan (for display/metrics)
    repair_until: int = -1
    goals_completed: int = 0
    # Absolute time until which the robot executes a repair plan that was NOT
    # validated (no-validator ablation). The world does not silently fix such
    # plans; only the low-level execution guard protects against collisions.
    unvalidated_until: int = -1
