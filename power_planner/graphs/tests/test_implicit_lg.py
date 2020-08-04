import numpy as np
import unittest
from types import SimpleNamespace
from power_planner import graphs
from power_planner.utils.utils import bresenham_line


class TestImplicitLG(unittest.TestCase):

    expl_shape = (50, 50)

    # create configuration
    cfg = SimpleNamespace()
    cfg.pylon_dist_min = 3
    cfg.pylon_dist_max = 5
    start_inds = np.array([6, 6])
    dest_inds = np.array([41, 43])
    cfg.start_inds = start_inds
    cfg.dest_inds = dest_inds
    cfg.angle_weight = 0.25
    cfg.edge_weight = 0
    cfg.max_angle = np.pi / 2
    cfg.max_angle_lg = np.pi / 4
    cfg.layer_classes = ["dummy_class"]
    cfg.class_weights = [1]

    # construct simple line instance
    example_inst = np.ones(expl_shape)
    # construct corresponding corridor
    working_expl_corr = np.zeros(expl_shape)
    line = bresenham_line(
        start_inds[0], start_inds[1], dest_inds[0], dest_inds[1]
    )
    for (i, j) in line:
        working_expl_corr[i - 1:i + 1, j - 1:j + 1] = 1

    # construct instance that required 90 degree angle
    high_angle_corr = np.zeros(expl_shape)
    high_angle_corr[start_inds[0], start_inds[1]:dest_inds[1] - 3] = 1
    high_angle_corr[start_inds[0], dest_inds[1]] = 1
    high_angle_corr[start_inds[0] + 3:dest_inds[0] + 1, dest_inds[1]] = 1

    def test_correct_shortest_path(self) -> None:
        """ Test the implicit line graph construction """
        graph = graphs.ImplicitLG(
            np.array([self.example_inst]),
            self.working_expl_corr,
            n_iters=10,
            verbose=0
        )
        path, path_costs, cost_sum = graph.single_sp(**vars(self.cfg))
        self.assertListEqual(graph.cost_weights.tolist(), [0.2, 0.8])
        self.assertTupleEqual(graph.instance.shape, self.expl_shape)
        self.assertEqual(np.sum(graph.cost_weights), 1)
        self.assertNotEqual(len(graph.shifts), 0)
        # all initialized to infinity
        # self.assertTrue(not np.any([graph.dists[graph.dists < np.inf]]))
        # start point was set to normalized value
        start_ind = graph.pos2node[tuple(self.start_inds)]
        self.assertEqual(0.8, graph.dists[start_ind, 0])
        # all start dists have same value
        self.assertEqual(len(np.unique(graph.dists[start_ind])), 1)
        # not all values still inf
        self.assertLess(np.min(graph.dists[start_ind]), np.inf)
        # get actual best path
        # path, path_costs, cost_sum = graph.get_shortest_path(
        #     self.start_inds, self.dest_inds
        # )
        self.assertNotEqual(len(path), 0)
        self.assertEqual(len(path), len(path_costs))
        self.assertGreaterEqual(cost_sum, 5)
        weighted_costs = np.sum(path_costs, axis=0) * graph.cost_weights
        self.assertEqual(np.sum(weighted_costs), cost_sum)
        for (i, j) in path:
            self.assertEqual(self.example_inst[i, j], 1)

    def test_edge_costs(self) -> None:
        graph = graphs.ImplicitLG(
            np.array([self.example_inst]),
            self.working_expl_corr,
            n_iters=10,
            verbose=0
        )
        self.cfg.angle_weight = 0
        self.cfg.edge_weight = 0.5
        path, path_costs, cost_sum = graph.single_sp(**vars(self.cfg))
        self.cfg.angle_weight = 0.25
        self.cfg.edge_weight = 0
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        dest_costs = np.min(graph.dists[dest_ind])
        dest_costs_gt = len(path)  # everywhere 1
        a = []
        path = np.array(path)
        for p in range(len(path) - 1):
            line = bresenham_line(
                path[p, 0], path[p, 1], path[p + 1, 0], path[p + 1, 1]
            )[1:-1]
            line_costs = [self.example_inst[i, j] for (i, j) in line]
            line_costs = [l for l in line_costs if l < np.inf]
            a.append(np.mean(line_costs) * 0.5)
        dest_costs_gt += np.sum(a)
        self.assertEqual(dest_costs, dest_costs_gt)
        self.assertEqual(dest_costs, cost_sum)

    def test_angle_sp(self) -> None:
        graph = graphs.ImplicitLG(
            np.array([self.example_inst]),
            self.high_angle_corr,
            n_iters=10,
            verbose=0
        )
        path, path_costs, cost_sum = graph.single_sp(**vars(self.cfg))
        # assert that destination can NOT be reached
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        self.assertFalse(np.min(graph.dists[dest_ind]) < np.inf)

        # NEXT TRY: more angles allowed
        self.cfg.max_angle_lg = np.pi
        graph = graphs.ImplicitLG(
            np.array([self.example_inst]),
            self.high_angle_corr,
            n_iters=10,
            verbose=0
        )
        path, path_costs, cost_sum = graph.single_sp(**vars(self.cfg))
        # assert that dest CAN be reached
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        self.assertTrue(np.min(graph.dists[dest_ind]) < np.inf)

        # same with linegraph:
        lg_graph = graphs.LineGraph(
            np.array([self.example_inst]), self.high_angle_corr, verbose=0
        )
        path_lg, path_costs_lg, cost_sum_lg = lg_graph.single_sp(
            **vars(self.cfg)
        )
        self.cfg.max_angle_lg = np.pi / 4
        # assert that path is non-empty
        self.assertTrue(len(path_lg) > 0)
        self.assertEqual(len(path_lg), len(path_costs_lg))
        self.assertGreaterEqual(cost_sum_lg, 5)
        weighted_costs = np.sum(path_costs_lg, axis=0) * lg_graph.cost_weights
        self.assertEqual(np.sum(weighted_costs), cost_sum_lg)
        # assert that all points are equal to 1
        for (i, j) in path_lg:
            self.assertEqual(self.example_inst[i, j], 1)

        lg_graph = graphs.LineGraph(
            np.array([self.example_inst]), self.high_angle_corr, verbose=0
        )
        path_lg, path_costs_lg, cost_sum_lg = lg_graph.single_sp(
            **vars(self.cfg)
        )
        self.assertTrue(len(path_lg) == 0)


if __name__ == '__main__':
    unittest.main()
