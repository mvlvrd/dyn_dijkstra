import heapq
from typing import Union

import numpy as np

from sssp import apsp
from SingleSinkDynGraph import FixSizeGraph, _new_2d_array


from internals import FTYPE, ITYPE

"""
We assume that the graphs have a fixed number of vertexes.
This allows to use an efficient 2d-array representation of the graph.
"""


class DynDijkstra(FixSizeGraph):
    def insert_edge(self, a, b, length: FTYPE):
        self.lengths[a, b] = length

    def delete_edge(self, a, b):
        self.lengths[a, b] = np.inf

    def get_distances(self):
        dists, _ = apsp(self.lengths)
        spg = _getSpgtoSink(self.lengths, dists)
        return dists, spg


class DynRR(FixSizeGraph):
    """
    This maintains a graph of fixed number of vertexes (N_vertex)
    and keeps an array with the single-sink-shortest-path distances and the predecessors.
    Uses　Ramalingam-Reps: https://www.sciencedirect.com/science/article/abs/pii/S0196677496900462

    TODO: Check if it is better to keep the shortest past spg instead of recalculating it each time.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "graph" in kwargs:
            self.distances, _ = apsp(self.lengths)
        else:
            self.distances = _new_2d_array(self.N_vertex)

    def delete_edge(self, v, w):
        self.lengths[v, w] = np.inf
        affectedSinks = _deleteUpdate(self.lengths.T, self.distances.T, w, v, v, onlyPhase1=True)
        affectedSources = _deleteUpdate(self.lengths, self.distances, v, w, w, onlyPhase1=True)
        for sink in affectedSinks.nonzero()[0]:
            _deleteUpdate(self.lengths, self.distances, v, w, sink)
        for source in affectedSources.nonzero()[0]:
            _deleteUpdate(self.lengths.T, self.distances.T, w, v, source)

    def insert_edge(self, v, w, c):
        old_edge = self.lengths[v, w]
        if c < old_edge:
            self._insert_edge(v, w, c)
        elif c > old_edge:
            self.delete_edge(v, w)
            self._insert_edge(v, w, c)

    def _insert_edge(self, v, w, c):
        self.lengths[v, w] = c
        affectedSinks = _insertUpdate(self.lengths.T, self.distances.T.copy(), w, v, v)
        affectedSources = _insertUpdate(self.lengths, self.distances, v, w, w)
        for sink in affectedSinks.nonzero()[0]:
            _insertUpdate(self.lengths, self.distances, v, w, sink)
        for source in affectedSources.nonzero()[0]:
            _insertUpdate(self.lengths.T, self.distances.T, w, v, source)

    def get_distances(self):
        spg = _getSpgtoSink(self.lengths, self.distances)
        return self.distances, spg


def _deleteUpdate(lengths, distances, v, w, sink: int, onlyPhase1=False) -> Union[np.ndarray, None]:
    N_vertex = lengths.shape[0]
    affectedVertices = np.zeros(N_vertex, dtype=np.bool_)
    sp_v = SP_all_heads(lengths, distances, v, sink)
    if not sp_v.any():
        workSet = {v}
        # Phase 1: Identify vertices in AFFECTED (the vertices whose shortest distance to sink has increased).
        while workSet:
            u = workSet.pop()
            affectedVertices[u] = True
            xs = SP_all_tails(lengths, distances, u, sink)
            tmp = []
            for ix, x in enumerate(xs):
                if x:
                    sp_y = SP_all_heads(lengths, distances, ix, sink)
                    if affectedVertices[sp_y].all():
                        tmp.append(ix)
            workSet = workSet.union(tmp)
        if onlyPhase1:
            return affectedVertices

        # Phase 2: Determine new distances to sink for all vertices in AffectedVertices.
        priorityQueue = []
        na = ~ affectedVertices
        tmp = lengths[affectedVertices][:, na] + distances[na, sink]
        dist_to_sink = np.min(tmp, axis=1, initial=np.inf)
        distances[affectedVertices, sink] = dist_to_sink
        for a in np.nonzero(affectedVertices)[0]:
            heapq.heappush(priorityQueue, (distances[a, sink], a))
        while priorityQueue:
            dist_a_sink, a = heapq.heappop(priorityQueue)
            c = (lengths[:, a] < np.inf) & (lengths[:, a] + distances[a, sink] < distances[:, sink])
            for ix, x in enumerate(c):
                if x:
                    distances[ix, sink] = lengths[ix, a] + dist_a_sink
                    heapq.heappush(priorityQueue, (distances[ix, sink], ix))

    if onlyPhase1:
        return affectedVertices


def _insertUpdate(lengths, distances, v, w, sink):
    N_vertex = lengths.shape[0]
    workSet = {(v, w)}
    visitedVertices = np.zeros(N_vertex, dtype=np.bool_)
    visitedVertices[v] = 1
    affectedVertices = np.zeros(N_vertex, dtype=np.bool_)
    while workSet:
        x, u = workSet.pop()
        temp_dist = lengths[x, u] + distances[u, sink]
        if temp_dist < distances[x, sink]:
            affectedVertices[x] = True
            distances[x, sink] = temp_dist
            sp = SP_all_tails(lengths, distances, x, v) & ~ visitedVertices
            for iy, y in enumerate(sp):
                if y:
                    workSet.add((iy, x))
                    visitedVertices[iy] = 1
    return affectedVertices


def SP_all_tails(lengths: np.ndarray, distances: np.ndarray, head: int, sink: int) -> np.ndarray:
    kk = (lengths[:, head] < np.inf) &\
         (distances[:, sink] == lengths[:, head] + distances[head, sink]) &\
         (distances[:, sink] < np.inf)
    kk[head] = False
    return kk


def SP_all_heads(lengths, distances, tail: int, sink: int) -> np.ndarray:
    kk = (lengths[tail, :] < np.inf) &\
          (distances[tail, sink] == lengths[tail, :] + distances[:, sink]) &\
          (distances[tail, sink] < np.inf)
    kk[tail] = False
    return kk


def _getSpgtoSink(lengths: np.ndarray, distances: np.ndarray, sink=None):
    """
    Reconstruct the shortest path matrix spg for a graph (lengths)
    given the shortest path distances
    :param lengths: np.ndarray(N_vertex, N_vertex)
    :param distances: np.ndarray(N_vertex, N_vertex) or np.ndarray(N_vertex)
    :param sink:
    If sink is an integer, distances[i] is the shortest path distances array from every vertex to sink
    if sink is not given, distances[i,j] is the shortest path distances array from i to j
    :return: spg: np.ndarray(N_vertex, N_vertex, N_sinks)
    """
    N_vertex = lengths.shape[0]
    if sink:
        N_sinks = 1
    else:
        N_sinks = N_vertex
        sink = slice(None)
        # Convert (N,) array in (1,N) array
        distances = np.expand_dims(distances, axis=1)

    spg = np.zeros((N_vertex, N_vertex, N_sinks), dtype=np.bool_)
    for a in np.arange(N_vertex, dtype=ITYPE):
        for b in np.arange(N_vertex, dtype=ITYPE):
            c = (distances[a, sink] == lengths[a, b] + distances[b, sink]) & (lengths[a, b] < np.inf)
            spg[a, b, c[0]] = True
    return spg
