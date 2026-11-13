"""Grid and map representation for MovingAI benchmark maps."""

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

Cell = Tuple[int, int]
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GridMap:
    """4-connected grid. ``blocked[y, x]`` is True for obstacles.

    All dynamic changes (blocked aisles, failed robots) must go through
    :meth:`block` / :meth:`unblock` so that the NetworkX graph, the boolean
    array and the cached distance maps stay consistent.
    """

    def __init__(self, blocked: np.ndarray, width: int, height: int):
        self.blocked: np.ndarray = blocked
        self.width: int = width
        self.height: int = height
        self.graph: nx.Graph = self._build_graph()
        self.version: int = 0  # bumped on every block/unblock
        self._dist_cache: "OrderedDict[Tuple[Cell, int], Dict[Cell, int]]" = OrderedDict()
        self._dist_cache_size = 1024

    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        for y in range(self.height):
            for x in range(self.width):
                if not self.blocked[y, x]:
                    g.add_node((x, y))
                    for dx, dy in ((1, 0), (0, 1)):
                        nx_, ny_ = x + dx, y + dy
                        if nx_ < self.width and ny_ < self.height and not self.blocked[ny_, nx_]:
                            g.add_edge((x, y), (nx_, ny_))
        return g

    # ------------------------------------------------------------------ queries
    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Cell) -> bool:
        x, y = cell
        if 0 <= x < self.width and 0 <= y < self.height:
            return not bool(self.blocked[y, x])
        return False

    def neighbors(self, cell: Cell) -> List[Cell]:
        """Free 4-neighbours of ``cell`` (works even if ``cell`` itself is blocked)."""
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in _DIRS if self.is_free((x + dx, y + dy))]

    def get_free_cells(self) -> List[Cell]:
        free_indices = np.argwhere(~self.blocked)
        return [(int(x), int(y)) for y, x in free_indices]

    def get_obstacle_cells(self) -> List[Cell]:
        obs_indices = np.argwhere(self.blocked)
        return [(int(x), int(y)) for y, x in obs_indices]

    def distance_map(self, goal: Cell) -> Dict[Cell, int]:
        """Exact BFS distance from every reachable free cell to ``goal`` on the
        *current* map. Cached per (goal, map version)."""
        key = (goal, self.version)
        d = self._dist_cache.get(key)
        if d is not None:
            self._dist_cache.move_to_end(key)
            return d
        if self.graph.has_node(goal):
            d = nx.single_source_shortest_path_length(self.graph, goal)
        else:
            d = {}
        self._dist_cache[key] = d
        if len(self._dist_cache) > self._dist_cache_size:
            self._dist_cache.popitem(last=False)
        return d

    def distance(self, a: Cell, b: Cell) -> int:
        """True shortest-path distance; falls back to a large Manhattan-based
        value if ``a`` cannot reach ``b``."""
        d = self.distance_map(b).get(a)
        if d is None:
            return 10_000 + abs(a[0] - b[0]) + abs(a[1] - b[1])
        return d

    # ----------------------------------------------------------------- updates
    def block(self, cells: Iterable[Cell]) -> List[Cell]:
        changed = []
        for c in cells:
            if self.is_free(c):
                x, y = c
                self.blocked[y, x] = True
                if self.graph.has_node(c):
                    self.graph.remove_node(c)
                changed.append(c)
        if changed:
            self.version += 1
        return changed

    def unblock(self, cells: Iterable[Cell]) -> List[Cell]:
        changed = []
        for c in cells:
            if self.in_bounds(c) and not self.is_free(c):
                x, y = c
                self.blocked[y, x] = False
                self.graph.add_node(c)
                for n in self.neighbors(c):
                    self.graph.add_edge(c, n)
                changed.append(c)
        if changed:
            self.version += 1
        return changed

    @classmethod
    def from_movingai(cls, file_path: str) -> "GridMap":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f if line.strip()]
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
        grid_lines = lines[4:4 + height]
        blocked = np.zeros((height, width), dtype=bool)
        for y, row in enumerate(grid_lines):
            for x, char in enumerate(row):
                if char in ("@", "T", "O"):
                    blocked[y, x] = True
        return cls(blocked, width, height)


def load_map(file_path: str) -> GridMap:
    """Helper function to load a MovingAI .map file."""
    return GridMap.from_movingai(file_path)
