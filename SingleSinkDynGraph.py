import heapq
import numpy as np

from sssp import sSinksp

from internals import FTYPE, ITYPE

"""
We assume that the graphs have a fixed cardinality.
"""


def _new_2d_array(n):
    x = np.empty((n, n), dtype=FTYPE)
    x.fill(np.inf)
    np.fill_diagonal(x, 0)
    return x


class FixSizeGraph:
    def __init__(self, N_vertex=None, graph: np.ndarray=None):
        if N_vertex is None:
            self.lengths = graph
            self.N_vertex = self.lengths.shape[0]
        else:
            self.N_vertex = N_vertex
            self.lengths = _new_2d_array(self.N_vertex)


class FixSizeGraphSingleSink(FixSizeGraph):
    def __init__(self, sink, **kwargs):
        self.sink = sink
        super().__init__(**kwargs)


class SingleSinkDynDijkstra(FixSizeGraphSingleSink):
    def insert_edge(self, a, b, length: FTYPE):
        self.lengths[a, b] = length

    def delete_edge(self, a, b):
        self.lengths[a, b] = np.inf

    def get_distances(self):
        return sSinksp(self.lengths, self.sink)


class SingleSinkDynRRL(FixSizeGraphSingleSink):
    """
    This maintains a graph of fixed number of vertexes (N_vertex)
    and keeps an array with the single-sink-shortest-path distances and the predecessors.
    Uses　Ramalingam-Reps: https://www.sciencedirect.com/science/article/abs/pii/S0196677496900462
    """
    def __init__(self, sink, **kwargs):
        super().__init__(sink, **kwargs)
        self.distances, self.spg = None, None

    def delete_edge(self, v, w):
        self.lengths[v, w] = np.inf
        if self.spg[v, w] == 1:
            self.spg[v, w] = 0
            if self._is_sp_outdegree_zero(v):
                # Identify affected vertices
                workSet = {v}
                affectedVertices = np.zeros(self.N_vertex, dtype=np.bool_)
                while workSet:
                    u = workSet.pop()
                    affectedVertices[u] = 1
                    xs = self.spg[:, u].copy()
                    self.spg[:, u] = 0
                    tmp = [i for i, _ in enumerate(xs) if _ and self._is_sp_outdegree_zero(i)]
                    workSet = workSet.union(tmp)

                # Recalculate shortest paths from affected vertices
                v_queue = []
                na = np.invert(affectedVertices)
                # This is ugly but it's a workaround to this bug https://github.com/numpy/numpy/issues/13255
                tmp = self.lengths[affectedVertices][:, na] + self.distances[na]
                dist_a = np.min(tmp, axis=1, initial=np.inf)
                self.distances[affectedVertices] = dist_a
                # TODO: This is a very ugly loop
                for (i, x) in enumerate(affectedVertices):
                    if x:
                        dist_i = self.distances[i]
                        if dist_i < np.inf:
                            heapq.heappush(v_queue, (dist_i, i))
                while v_queue:
                    dist_a, a = heapq.heappop(v_queue)
                    self.spg[a, self.lengths[a, :] + self.distances == dist_a] = 1
                    # Update pred(a)
                    in_v = self.lengths[:, a] + dist_a < self.distances
                    new_dist = self.lengths[in_v, a] + dist_a
                    self.distances[in_v] = new_dist
                    for d, i in zip(new_dist, in_v.nonzero()[0]):
                        heapq.heappush(v_queue, (d, i))

    def insert_edge(self, v, w, c):
        old_edge = self.lengths[v, w]
        if c < old_edge:
            self._insert_edge(v, w, c)
        elif c > old_edge:
            self.delete_edge(v, w)
            self._insert_edge(v, w, c)

    def _insert_edge(self, v, w, c):
        self.lengths[v, w] = c
        v_queue = []
        temp_dist = self.lengths[v, w] + self.distances[w]
        if temp_dist < self.distances[v]:
            self.distances[v] = temp_dist
            heapq.heappush(v_queue, (0., v))
        elif temp_dist == self.distances[v]:
            self.spg[v, w] = 1
        while v_queue:
            _, u = heapq.heappop(v_queue)
            dist_u = self.distances[u]
            self.spg[u, :] = np.where(self.lengths[u, :] + self.distances == dist_u, 1, 0)

            new_candidate_dists = self.lengths[:, u] + dist_u
            new_heap_pred = new_candidate_dists < self.distances
            self.distances[new_heap_pred] = new_candidate_dists[new_heap_pred]
            dist_v = self.distances[v]
            for ix in np.arange(0, self.N_vertex):
                if new_heap_pred[ix]:
                    heapq.heappush(v_queue, (self.distances[ix] - dist_v, ix))  # TODO: This should be adjust, not push.

            self.spg[(new_candidate_dists == self.distances), u] = 1

    def _is_sp_outdegree_zero(self, i):
        _x = np.ma.array(self.spg[i, :], mask=False)
        _x[i] = np.ma.masked
        return not _x.any()

    def get_distances(self):
        if self.distances is None:
            self.distances, succs = sSinksp(self.lengths, self.sink)
            self.spg = np.zeros((self.N_vertex, self.N_vertex), dtype=np.bool_)
            self.spg[np.arange(0, self.N_vertex), succs] = 1
        return self.distances, self.spg
