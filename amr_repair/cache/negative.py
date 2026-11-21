"""Negative memory: strategies that failed validation for a category of incident.

A *strategy signature* is ``(solver, horizon)``: the part of a directive that
transfers between incidents (robot IDs never do). A signature is negative for
a hard-filter category after it fails validation there, and is cleared again
when the same signature later passes validation in that category. Cache lookups
skip negative signatures, and the LLM is shown them as "rejected before".
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from amr_repair.llm.schema import Directive
from amr_repair.sentinel.incident import IncidentRecord

Signature = Tuple[str, int]


def robot_bucket(n: int) -> str:
    """Incident size by the number of robots that must be replanned."""
    return "1" if n <= 1 else ("2-3" if n <= 3 else "4+")


def get_hard_filter_key(incident: IncidentRecord) -> str:
    """Category key: disruption types | robots to replan (1, 2-3, 4+).

    Deliberately broad, so strategies transfer between incidents. Congestion,
    interaction with earlier disruptions and the other continuous descriptors
    are compared by the similarity score instead (IncidentRecord.soft_features)."""
    types = ",".join(incident.disruption_types())
    return f"{types}|{robot_bucket(len(incident.plannable_affected()))}"


def signature(directive: Directive) -> Signature:
    return (directive.solver, int(directive.horizon))


class NegativeMemory:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        # key -> {"solver:horizon": {"solver", "horizon", "failures", "last_violation"}}
        self.entries: Dict[str, Dict[str, dict]] = {}
        if self.filepath and os.path.exists(self.filepath):
            self.load(self.filepath)

    @staticmethod
    def _sig_key(sig: Signature) -> str:
        return f"{sig[0]}:{sig[1]}"

    def add(self, incident: IncidentRecord, directive: Directive, violation: Optional[str] = None) -> None:
        key = get_hard_filter_key(incident)
        sig = signature(directive)
        bucket = self.entries.setdefault(key, {})
        e = bucket.setdefault(self._sig_key(sig), {"solver": sig[0], "horizon": sig[1], "failures": 0})
        e["failures"] += 1
        if violation:
            e["last_violation"] = violation[:160]
        self._persist()

    def clear(self, incident: IncidentRecord, directive: Directive) -> None:
        """The signature just passed validation in this category: forget it."""
        key = get_hard_filter_key(incident)
        if self.entries.get(key, {}).pop(self._sig_key(signature(directive)), None) is not None:
            self._persist()

    def contains(self, incident: IncidentRecord, directive: Directive) -> bool:
        key = get_hard_filter_key(incident)
        return self._sig_key(signature(directive)) in self.entries.get(key, {})

    def rejected_for(self, incident: IncidentRecord) -> List[dict]:
        return list(self.entries.get(get_hard_filter_key(incident), {}).values())

    def __len__(self) -> int:
        return sum(len(v) for v in self.entries.values())

    def _persist(self) -> None:
        if self.filepath:
            self.save(self.filepath)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ignore files written by the old format (lists of directive dicts)
        self.entries = {k: v for k, v in data.items() if isinstance(v, dict)}
