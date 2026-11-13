"""System prompt and prompt formatter for the LLM strategist.

The LLM never sees the map. It sees a compact summary of the IncidentRecord
(disruptions, congestion, interactions, per-robot context), strategies that
already failed for this category, and on a retry the exact validator
violation of its previous directive.
"""

import json
from typing import List, Optional

from amr_repair.sentinel.incident import IncidentRecord

SYSTEM_PROMPT = """You are the repair strategist for a fleet of warehouse robots (lifelong multi-agent path finding).
A disruption has invalidated some robots' collision-free plans. Choose HOW to repair them:
which robots to replan, which classical solver to run, the planning priority order and the planning horizon.
A deterministic validator checks the resulting plans against the whole fleet before anything executes.

Solvers:
- pibt: PIBT, one-step greedy with priority inheritance. Fastest (~1 ms). Myopic: can end robots in
  unsafe parking spots or fail in dense areas / narrow aisles. Good for small, low-congestion incidents.
- lns: Large Neighbourhood Search over prioritized planning (space-time A*). Best path quality,
  moderate time (<=200 ms). Good when several robots compete for the same aisle.
- windowed_pbs: Priority-Based Search over priority orderings. Most robust for interacting /
  multi-robot conflicts and bottlenecks, slowest (<=500 ms).

Rules:
1. robots: subset of candidate_robots; MUST include every robot in must_replan. Replanning extra
   nearby robots can free space but costs time.
2. priority: an exact permutation of robots; earlier = planned first (gets right of way).
   Delayed robots cannot move until their delay ends; robots far from their goal often benefit
   from lower priority.
3. horizon: integer 10..40 (planning window). Longer horizons help when the detour is long.
4. Avoid strategies listed under rejected_strategies for this kind of incident.
   Overlapping disruptions (overlap_kind simultaneous / sequential / compound) constrain the same
   area at once: robots still executing an earlier repair may need to be replanned jointly.
5. rationale: one short sentence (<200 characters).
Return only JSON matching the schema."""


def incident_summary(incident: IncidentRecord) -> dict:
    return {
        "incident_id": incident.id,
        "disruptions": [
            {k: v for k, v in d.model_dump().items() if v not in (None, [])} for d in incident.disruptions
        ],
        "size_bucket": incident.size_bucket,
        "congestion": {"level": incident.congestion.level, "density": round(incident.congestion.density, 3)},
        "interacting": bool(incident.interacts_with),
        "overlap_kind": incident.overlap_kind,
        "interacts_with": incident.interacts_with,
        "overlapping_earlier_disruptions": [
            {k: v for k, v in d.model_dump().items() if v not in (None, [])} for d in incident.related_disruptions
        ],
        "failed_robots": incident.failed_robots,
        "must_replan": incident.must_replan,
        "candidate_robots": incident.candidate_robots,
        "robots": [c.model_dump() for c in incident.robot_context],
    }


def format_user_prompt(
    incident: IncidentRecord,
    violation: Optional[str] = None,
    rejected: Optional[List[dict]] = None,
    previous_directive: Optional[dict] = None,
) -> str:
    payload = {"incident": incident_summary(incident)}
    if rejected:
        payload["rejected_strategies"] = rejected
    prompt = "Incident detected:\n" + json.dumps(payload, indent=1)
    if violation:
        prompt += (
            f"\n\nCRITICAL: Previous directive failed validation: {violation}."
            + (f"\nPrevious directive: {json.dumps(previous_directive)}" if previous_directive else "")
            + "\nReturn a corrected directive that resolves this violation "
              "(e.g. change the solver, the priority order, the horizon, or replan additional robots)."
        )
    else:
        prompt += "\n\nSelect the repair directive."
    return prompt
