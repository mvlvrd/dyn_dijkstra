import unittest
import time

import numpy as np

from sssp import sSourcesp
from SingleSinkDynGraph import SingleSinkDynRRL, SingleSinkDynDijkstra, _new_2d_array
from DynGraph import DynRR, DynDijkstra


class MyTestCase(unittest.TestCase):

    def _assert2graphs(self, graph1, graph2):
        d1, spg1 = graph1.get_distances()
        d2, spg2 = graph2.get_distances()
        _d1, _d2 = d1[d1 < np.inf], d2[d2 < np.inf]
        np.testing.assert_array_equal(_d1, _d2)
        np.testing.assert_equal(spg1, spg2)

    def setUp(self):
        self.edges = _new_2d_array(7)

        self.edges[0, 1] = 2.
        self.edges[0, 6] = 10.
        self.edges[1, 2] = 3.
        self.edges[2, 3] = 2.
        self.edges[2, 5] = 1.
        self.edges[4, 5] = 7.
        self.edges[5, 2] = 9.
        self.edges[5, 3] = 8.
        self.edges[5, 6] = 0.
        self.edges[6, 0] = 4.
        self.edges[6, 2] = 5.

    def test0(self):
        distances, prevs = sSourcesp(self.edges.copy(), 0)

        expected_distances = np.array([0., 2., 5., 7., np.inf, 6., 6.])
        expected_prevs = np.array([0, 0, 1, 2, 4, 2, 5])

        np.testing.assert_equal(distances, expected_distances)
        np.testing.assert_array_equal(prevs, expected_prevs)

    def test1(self):
        self._test_sSinks1(SingleSinkDynRRL)
        self._test_sSinks1(SingleSinkDynDijkstra)

    def test2(self):
        N_vertex = 20
        allPairs_d = DynDijkstra(N_vertex=N_vertex)
        allPairs_rr = DynRR(N_vertex=N_vertex)
        rng = np.random.default_rng(seed=2)
        self._testRandom(allPairs_d, allPairs_rr, rng=rng, N_actions=100)

    def test3(self):
        rng = np.random.default_rng(seed=3)
        random_lengths = np.round(rng.exponential(size=2*(16,)), decimals=1)*10
        allPairs_d = DynDijkstra(graph=random_lengths.copy())
        allPairs_rr = DynRR(graph=random_lengths.copy())
        self._testRandom(allPairs_d, allPairs_rr, rng=rng, N_actions=100)

    def _testRandom(self, allPairs_d, allPairs_rr, rng, N_actions):
        N_vertex = allPairs_d.N_vertex

        edge_lengths = np.round(rng.lognormal(size=N_actions), decimals=1)*10
        actions = rng.choice(["insert", "delete"], size=N_actions)
        edges = rng.integers(0, N_vertex, size=(N_actions, 2))

        dTimes, rrTimes = [], []
        for _ in range(N_actions):
            action = actions[_]
            a, b = edges[_]
            edge_length = edge_lengths[_]
            # print(action, a, b, edge_length)

            if a == b:
                continue
            if action == "insert":
                start = time.time()
                allPairs_d.insert_edge(a, b, edge_length)
                end = time.time()
                dTimes.append(end - start)
                start = time.time()
                allPairs_rr.insert_edge(a, b, edge_length)
                end = time.time()
                rrTimes.append(end-start)
            else:
                start = time.time()
                allPairs_d.delete_edge(a, b)
                end = time.time()
                dTimes.append(end-start)
                start = time.time()
                allPairs_rr.delete_edge(a, b)
                end = time.time()
                rrTimes.append(end-start)

            self._assert2graphs(allPairs_d, allPairs_rr)

        for a, b in zip(dTimes, rrTimes):
            print(a, b)
        return dTimes, rrTimes

    def _test_sSinks1(self, testClass):
        sSink = testClass(graph=self.edges.copy(), sink=0)
        allPairs_d = DynDijkstra(graph=self.edges.copy())
        allPairs_rr = DynRR(graph=self.edges.copy())

        dists1, spg_ = sSink.get_distances()
        expDists1 = np.array([0, 8, 5, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists1, expDists1)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.delete_edge(6, 2)
        dists_, spg_ = sSink.get_distances()
        expDists = np.array([0, 8, 5, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists_, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(6, 2, 5.)
        dists_, spg_ = sSink.get_distances()
        expDists = np.array([0, 8, 5, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists_, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(1, 0, 2.)
        dists2, spg_ = sSink.get_distances()
        expDists = np.array([0, 2, 5, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists2, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(2, 4, 5.)
        dists3, spg_ = sSink.get_distances()
        expDists = np.array([0, 2, 5, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists3, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.delete_edge(2, 5)
        dists4, spg_ = sSink.get_distances()
        expDists = np.array([0, 2, 16, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists4, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.delete_edge(1, 0)
        dists5, spg_ = sSink.get_distances()
        expDists = np.array([0, 19, 16, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists5, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(3, 4, 1.)
        dists6, spg_ = sSink.get_distances()
        expDists = np.array([0, 17, 14, 12, 11, 4, 4])
        np.testing.assert_equal(dists6, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.delete_edge(3, 4)
        dists7, spg_ = sSink.get_distances()
        expDists = np.array([0, 19, 16, np.inf, 11, 4, 4])
        np.testing.assert_equal(dists7, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(3, 4, 1.)
        dists8, spg_ = sSink.get_distances()
        expDists = np.array([0, 17, 14, 12, 11, 4, 4])
        np.testing.assert_equal(dists8, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(2, 4, 5.)
        dists9, spg_ = sSink.get_distances()
        np.testing.assert_equal(dists9, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)

        for x in sSink, allPairs_d, allPairs_rr:
            x.insert_edge(2, 4, 3.)
        dists10, spg_ = sSink.get_distances()
        np.testing.assert_equal(dists10, expDists)
        self._assert2graphs(allPairs_d, allPairs_rr)


if __name__ == '__main__':
    unittest.main()
