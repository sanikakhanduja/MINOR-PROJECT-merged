"""Verified strategy cache.

Lookup = hard filter (disruption types, size bucket, congestion level,
isolated/interacting) followed by an RBF similarity on the incident's
continuous descriptors (``IncidentRecord.soft_features``), above a tuned
threshold. Strategies whose signature is in negative memory are skipped.
Only directives that passed the validator are ever stored. A hit is adapted to
the current incident (solver + horizon reused; robots = the incident's
plannable affected robots, ordered nearest-to-goal first) and must still pass
a dry run before it is executed.
"""

import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from amr_repair.cache.negative import NegativeMemory, get_hard_filter_key, signature
from amr_repair.llm.schema import Directive
from amr_repair.sentinel.incident import IncidentRecord

RBF_GAMMA = 2.0  # sim = exp(-gamma * ||a - b||^2)


class CacheEntry:
    def __init__(self, entry_id: str, hard_filter_key: str, features: np.ndarray,
                 incident_summary: dict, directive: Directive, outcome: dict,
                 last_accessed: float = 0.0, access_count: int = 1,
                 successes: int = 1, failures: int = 0, source: str = "llm"):
        self.incident_id = entry_id  # globally unique: "<run_id>:<incident id>"
        self.hard_filter_key = hard_filter_key
        self.features = np.asarray(features, dtype=np.float32)
        self.incident_summary = incident_summary
        self.directive = directive
        self.outcome = outcome
        self.last_accessed = last_accessed or time.time()
        self.access_count = access_count
        self.successes = successes
        self.failures = failures
        self.source = source

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id, "hard_filter_key": self.hard_filter_key,
            "features": self.features.tolist(), "incident_summary": self.incident_summary,
            "directive": self.directive.model_dump(), "outcome": self.outcome,
            "last_accessed": self.last_accessed, "access_count": self.access_count,
            "successes": self.successes, "failures": self.failures, "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(d["incident_id"], d["hard_filter_key"], np.array(d["features"], dtype=np.float32),
                   d.get("incident_summary", {}), Directive.model_validate(d["directive"]),
                   d.get("outcome", {}), d.get("last_accessed", 0.0), d.get("access_count", 1),
                   d.get("successes", 1), d.get("failures", 0), d.get("source", "llm"))


class StrategyCache:
    def __init__(self, threshold: float = 0.85, max_capacity: int = 500,
                 negative_memory: Optional[NegativeMemory] = None,
                 filepath: Optional[str] = None, namespace: str = "run"):
        self.threshold = threshold
        self.max_capacity = max(1, max_capacity)
        self.negative_memory = negative_memory if negative_memory is not None else NegativeMemory()
        self.filepath = filepath
        self.namespace = namespace
        self.entries: List[CacheEntry] = []
        self.lookups = self.hits = self.misses = 0
        self.insertions = self.updates = self.evictions = self.verified_reuses = 0
        self.skipped_negative = 0
        if self.filepath and os.path.exists(self.filepath):
            self.load(self.filepath)

    def entry_id(self, incident: IncidentRecord) -> str:
        return f"{self.namespace}:{incident.id}"

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(rbf_kernel(a.reshape(1, -1), b.reshape(1, -1), gamma=RBF_GAMMA)[0, 0])

    def lookup(self, incident: IncidentRecord, world=None) -> Optional[Tuple[Directive, str, float]]:
        """Best matching verified strategy above threshold, adapted to this incident."""
        self.lookups += 1
        robots = incident.plannable_affected()
        if self.threshold >= 1.0 or not self.entries or not robots:
            self.misses += 1
            return None
        key = get_hard_filter_key(incident)
        candidates = [e for e in self.entries if e.hard_filter_key == key]
        if not candidates:
            self.misses += 1
            return None
        feat = incident.soft_features().reshape(1, -1)
        sims = rbf_kernel(feat, np.vstack([e.features for e in candidates]), gamma=RBF_GAMMA)[0]
        for idx in np.argsort(-sims, kind="stable"):
            sim = float(sims[idx])
            if sim < self.threshold:
                break
            entry = candidates[idx]
            if self.negative_memory.contains(incident, entry.directive):
                self.skipped_negative += 1
                continue
            if world is not None:
                by_id = {r.id: r for r in world.robots}
                priority = sorted(robots, key=lambda r: (world.map.distance(by_id[r].pos, by_id[r].goal), r))
            else:
                priority = list(robots)
            adapted = Directive(robots=sorted(robots), solver=entry.directive.solver, priority=priority,
                                horizon=entry.directive.horizon,
                                rationale=f"Cache hit from {entry.incident_id} (sim={sim:.2f})")
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.hits += 1
            return adapted, entry.incident_id, sim
        self.misses += 1
        return None

    def record_verified_reuse(self, entry_id: str) -> None:
        self.verified_reuses += 1
        for e in self.entries:
            if e.incident_id == entry_id:
                e.successes += 1
                e.last_accessed = time.time()
        self._persist()

    def record_failed_reuse(self, entry_id: str) -> None:
        """Dry run failed: count it, and drop entries that fail more than they succeed."""
        for e in list(self.entries):
            if e.incident_id == entry_id:
                e.failures += 1
                if e.failures > e.successes:
                    self.entries.remove(e)
                    self.evictions += 1
        self._persist()

    def store(self, incident: IncidentRecord, directive: Directive, outcome: Optional[dict] = None,
              source: str = "llm") -> None:
        """Store a directive that has *already passed validation*."""
        eid = self.entry_id(incident)
        for e in self.entries:
            if e.incident_id == eid:
                e.directive, e.outcome = directive, outcome or e.outcome
                e.last_accessed = time.time()
                self.updates += 1
                self._persist()
                return
        while len(self.entries) >= self.max_capacity:
            oldest = min(range(len(self.entries)), key=lambda i: self.entries[i].last_accessed)
            self.entries.pop(oldest)
            self.evictions += 1
        self.entries.append(CacheEntry(
            entry_id=eid, hard_filter_key=get_hard_filter_key(incident), features=incident.soft_features(),
            incident_summary={"affected_count": len(incident.affected_robots),
                              "congestion_level": incident.congestion.level,
                              "types": incident.disruption_types()},
            directive=directive, outcome=outcome or {}, source=source,
        ))
        self.insertions += 1
        self._persist()

    def _persist(self) -> None:
        if self.filepath:
            self.save(self.filepath)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = []
        for d in data:
            # Entries from the old format used a different feature space; skip them.
            if len(d.get("features", [])) == 6 and d.get("hard_filter_key", "").count("|") == 3:
                self.entries.append(CacheEntry.from_dict(d))
