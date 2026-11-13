"""Directive schema for LLM-guided multi-robot path repair."""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class Directive(BaseModel):
    """Path repair directive produced by the LLM or retrieved from strategy cache."""

    robots: List[int] = Field(
        ...,
        description="List of robot IDs to replan. Must be non-empty.",
    )
    solver: Literal["pibt", "lns", "windowed_pbs"] = Field(
        ...,
        description="The multi-agent solver to execute: 'pibt', 'lns', or 'windowed_pbs'.",
    )
    priority: List[int] = Field(
        ...,
        description="Planning priority order. Must be a valid permutation of robots.",
    )
    horizon: int = Field(
        ...,
        ge=10,
        le=40,
        description="Planning horizon window in discrete time steps (10 to 40).",
    )
    rationale: str = Field(
        default="",
        max_length=200,
        description="Brief justification under 200 characters explaining the choice.",
    )

    @field_validator("robots")
    @classmethod
    def validate_robots_non_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("robots list cannot be empty.")
        if len(v) != len(set(v)):
            raise ValueError("robots list contains duplicate robot IDs.")
        return v

    @model_validator(mode="after")
    def validate_priority_is_permutation(self) -> "Directive":
        if set(self.priority) != set(self.robots) or len(self.priority) != len(self.robots):
            raise ValueError(
                f"priority ({self.priority}) must be an exact permutation of robots ({self.robots})."
            )
        return self


def grounded_directive(candidates, must_replan):
    """Directive subclass bound to one incident.

    Instructor validates the LLM output against this model, so hallucinated
    robot IDs (or omitting a robot whose plan crosses the disruption) raise a
    ValidationError whose message is sent back to the LLM as a re-ask.
    """
    allowed = set(int(r) for r in candidates)
    required = set(int(r) for r in must_replan)

    class IncidentDirective(Directive):
        @model_validator(mode="after")
        def _grounded(self) -> "IncidentDirective":
            unknown = sorted(set(self.robots) - allowed)
            if unknown:
                raise ValueError(
                    f"robots {unknown} are not candidates for this incident; "
                    f"choose only from candidate_robots={sorted(allowed)}"
                )
            missing = sorted(required - set(self.robots))
            if missing:
                raise ValueError(
                    f"robots {missing} are in must_replan (their plans cross the disruption) "
                    f"and must be included in robots"
                )
            return self

    IncidentDirective.__name__ = "Directive"
    return IncidentDirective
