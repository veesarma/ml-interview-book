"""A small Hierarchical Navigable Small World (HNSW) graph from scratch.

Malkov & Yashunin (2016). Each element gets a random top level
    l = floor(-ln(U) * mL),   U ~ Uniform(0, 1),   mL = 1 / ln(M)
so layer sizes decay geometrically. Insertion: greedy descent from the entry
point through layers above l (ef = 1), then at each layer <= l a beam search
with ef_construction candidates, connect to the M closest, prune neighbours to
M_max. Query: greedy descent to layer 0, beam search with ef >= k, return top-k.
Distances are squared L2.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


class HNSW:
    """Minimal multi-layer HNSW index over vectors (d,)."""

    def __init__(self, M: int = 8, ef_construction: int = 64, seed: int = 0) -> None:
        self.M = M
        self.M_max0 = 2 * M  # layer-0 nodes may keep twice as many links
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.vectors: list[np.ndarray] = []  # each (d,)
        self.levels: list[int] = []  # top level of each node
        # graph[l][i] = set of neighbour ids of node i at layer l
        self.graph: list[dict[int, set[int]]] = []
        self.entry: int | None = None

    def _dist(self, a: np.ndarray, i: int) -> float:
        diff = a - self.vectors[i]  # (d,)
        return float(diff @ diff)

    def _random_level(self) -> int:
        u = float(self.rng.uniform(1e-12, 1.0))
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: np.ndarray, entry: int, ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search at one layer. Returns up to ef (dist, id) pairs, closest first."""
        d0 = self._dist(q, entry)
        visited = {entry}
        candidates = [(d0, entry)]  # min-heap on distance
        results = [(-d0, entry)]  # max-heap (negated) of the current best ef
        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst = -results[0][0]
            if d_c > worst:
                break  # nothing closer left to expand
            for nb in self.graph[layer].get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                d_nb = self._dist(q, nb)
                worst = -results[0][0]
                if d_nb < worst or len(results) < ef:
                    heapq.heappush(candidates, (d_nb, nb))
                    heapq.heappush(results, (-d_nb, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-d, i) for d, i in results)

    def _connect(self, new: int, neighbours: list[int], layer: int) -> None:
        """Add bidirectional links and prune any node exceeding its degree cap."""
        m_max = self.M_max0 if layer == 0 else self.M
        self.graph[layer].setdefault(new, set()).update(neighbours)
        for nb in neighbours:
            links = self.graph[layer].setdefault(nb, set())
            links.add(new)
            if len(links) > m_max:  # keep the m_max closest (simple heuristic)
                keep = sorted(links, key=lambda j: self._dist(self.vectors[nb], j))[:m_max]
                self.graph[layer][nb] = set(keep)

    def add(self, x: np.ndarray) -> int:
        """Insert one vector x (d,). Returns its id."""
        idx = len(self.vectors)
        self.vectors.append(np.asarray(x, dtype=float))
        level = self._random_level()
        self.levels.append(level)
        while len(self.graph) <= level:
            self.graph.append({})
        if self.entry is None:
            self.entry = idx
            for l in range(level + 1):
                self.graph[l][idx] = set()
            return idx
        ep = self.entry
        top = self.levels[self.entry]
        # 1) greedy descent through the layers above the new node's level
        for l in range(top, level, -1):
            ep = self._search_layer(x, ep, 1, l)[0][1]
        # 2) beam search + connect on every layer the node participates in
        for l in range(min(level, top), -1, -1):
            found = self._search_layer(x, ep, self.ef_construction, l)
            neighbours = [i for _, i in found[: self.M]]
            self._connect(idx, neighbours, l)
            ep = found[0][1]
        if level > top:
            self.entry = idx
        return idx

    def search(self, q: np.ndarray, k: int, ef: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN of q (d,). -> (ids (k,), squared distances (k,))."""
        assert self.entry is not None, "index is empty"
        ef = max(ef, k)
        ep = self.entry
        for l in range(self.levels[self.entry], 0, -1):
            ep = self._search_layer(q, ep, 1, l)[0][1]
        found = self._search_layer(q, ep, ef, 0)[:k]
        ids = np.array([i for _, i in found], dtype=int)  # (k,)
        d2 = np.array([d for d, _ in found])  # (k,)
        return ids, d2
