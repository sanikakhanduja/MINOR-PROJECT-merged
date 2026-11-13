"""Post-hoc audit of executed frames (independent of the planner and validator)."""

from collections import Counter
from typing import List, Tuple

from amr_repair.replay.models import Frame


def audit_frames(frames: List[Frame]) -> Tuple[int, int, int]:
    """Return (vertex collisions, edge swaps, teleports) actually executed."""
    vertex = swaps = teleports = 0
    for f0, f1 in zip(frames, frames[1:]):
        counts = Counter(map(tuple, f1.pos))
        vertex += sum(n - 1 for n in counts.values() if n > 1)
        where = {tuple(p): i for i, p in enumerate(f0.pos)}
        for i, (a, b) in enumerate(zip(f0.pos, f1.pos)):
            a, b = tuple(a), tuple(b)
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                teleports += 1
            j = where.get(b)
            if j is not None and j > i and tuple(f1.pos[j]) == a and a != b:
                swaps += 1
    return vertex, swaps, teleports
