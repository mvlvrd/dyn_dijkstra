import heapq
import numpy as np

from internals import FTYPE, ITYPE


def apsp(edge_dists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # We calculate the All Pairs Shortest Paths solution for each vertex.
    # This is not the most efficient way. We should try to implement Johnson-Dijkstra algorithm
    N_vertex = edge_dists.shape[0]

    dists = np.empty((N_vertex, N_vertex), dtype=FTYPE)
    prev = np.empty((N_vertex, N_vertex), dtype=ITYPE)

    for i in range(N_vertex):
        d, p = sSourcesp(edge_dists, i)
        dists[i, :] = d
        prev[i, :] = p

    return dists, prev


def sSinksp(edge_dists: np.ndarray, sink) -> tuple[np.ndarray, np.ndarray]:
    """
    :param edge_dists: (np.ndarray[N_vertex, N_vertex])
    :param sink: int-like (index of the sink in edge_dists)
    :return: (dists np.ndarray[N_vertex], spg np.ndarray[N_vertex])
            The distances and the successors in the Shortest Paths Graph.
    """
    # We need to take transpose of the adjacency matrix
    # to convert the single source solution into single sink problem.
    return sSourcesp(edge_dists.T, sink)


def sSourcesp(edge_dists: np.ndarray, source) -> tuple[np.ndarray, np.ndarray]:
    """
    :param edge_dists: (np.ndarray[N_vertex, N_vertex])
    :param source: int-like (index of the source in edge_dists)
    :return: (dists np.ndarray[N_vertex], spg np.ndarray[N_vertex])
            The distances and the predecessors in the Shortest Paths Graph.
    """
    # TODO: Test that we are returning all possible Shortest Paths.
    N_vertex = edge_dists.shape[0]

    dists = np.empty(N_vertex, dtype=FTYPE)
    dists.fill(np.inf)
    dists[source] = 0

    prev = np.arange(0, N_vertex, 1)

    queue = []  # A list of 2-tuples in format (FTYPE, ITYPE)
    heapq.heappush(queue, (dists[source], source))

    while queue:
        dist, i = heapq.heappop(queue)
        for j in range(N_vertex):
            if j == i:
                continue
            new_cost = dist + edge_dists[i, j]
            if new_cost < dists[j]:
                dists[j] = new_cost
                prev[j] = i
                # This should be replaced by a decrease_key operation for better performance
                heapq.heappush(queue, (new_cost, j))

    return dists, prev
