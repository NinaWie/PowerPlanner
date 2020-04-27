import numpy as np
import unittest
from power_planner import graphs
from power_planner.utils.utils import bresenham_line


class TestImplicitLG(unittest.TestCase):

    # construct simple line instance
    expl_shape = (40, 40)
    working_expl = np.zeros(expl_shape)
    start_inds = np.array([1, 1])
    dest_inds = np.array([38, 36])
    working_expl += np.inf
    line = bresenham_line(
        start_inds[0], start_inds[1], dest_inds[0], dest_inds[1]
    )
    for (i, j) in line:
        working_expl[i, j] = 1

    # construct instance that required 90 degree angle
    example2 = np.zeros(expl_shape)
    example2 += np.inf
    example2[start_inds[0], start_inds[1]:dest_inds[1] - 3] = 1
    example2[start_inds[0], dest_inds[1]] = 1
    example2[start_inds[0] + 3:dest_inds[0] + 1, dest_inds[1]] = 1

    # construct random cost surface to assure that lg and impl output same
    example3 = np.random.rand(*expl_shape)

    def test_equal_to_lg(self) -> None:
        # compare whether angle costs are decreasing
        ang_costs_prev = np.inf
        all_angle_costs = []
        for ang_weight in [0.1, 0.3, 0.5, 0.7]:
            max_angle = np.pi / 2
            impl_lg = graphs.ImplicitLG(
                np.array([self.example3]),
                np.ones(self.expl_shape),
                n_iters=10,
                verbose=0
            )
            impl_lg = self.build_graph(
                impl_lg, max_angle_lg=max_angle, ang_weight=ang_weight
            )
            path, path_costs, cost_sum = impl_lg.get_shortest_path(
                self.start_inds, self.dest_inds
            )
            # get lg path
            lg_graph = graphs.LineGraph(
                np.array([self.example3]), np.ones(self.expl_shape), verbose=0
            )
            lg_graph = self.build_graph(
                lg_graph, max_angle_lg=max_angle, ang_weight=ang_weight
            )
            lg_graph.sum_costs()
            source_v, target_v = lg_graph.add_start_and_dest(
                self.start_inds, self.dest_inds
            )
            path_lg, path_costs_lg, cost_sum_lg = lg_graph.get_shortest_path(
                source_v, target_v
            )
            # assert that lg and other one are equal
            self.assertListEqual(list(path_lg), list(path))
            self.assertListEqual(list(path_costs_lg), list(path_costs))
            self.assertEqual(cost_sum, cost_sum_lg)

            angle_costs = np.sum(np.asarray(path_costs)[:, 0])
            all_angle_costs.append(angle_costs)
            self.assertLessEqual(angle_costs, ang_costs_prev)
            ang_costs_prev = angle_costs
        # check that diverse costs appear when varying the angle
        self.assertTrue(len(np.unique(all_angle_costs)) > 1)
        print("Done testing equal LG and Implicit")

    def build_graph(self, graph, max_angle_lg=np.pi / 4, ang_weight=0.25):
        graph.set_shift(
            3,
            5,
            self.dest_inds - self.start_inds,
            np.pi / 2,
            max_angle_lg=max_angle_lg
        )
        graph.set_edge_costs(["dummy_class"], [1], angle_weight=ang_weight)
        graph.add_nodes()
        corridor = np.ones(self.expl_shape) * 0.5
        graph.set_corridor(
            corridor, self.start_inds, self.dest_inds, factor_or_n_edges=1
        )
        graph.add_edges()
        return graph

    def test_correct_shortest_path(self) -> None:
        """ Test the implicit line graph construction """
        graph = graphs.ImplicitLG(
            np.array([self.working_expl]),
            np.ones(self.expl_shape),
            n_iters=10,
            verbose=0
        )
        graph = self.build_graph(graph)
        self.assertListEqual(graph.cost_weights.tolist(), [0.2, 0.8])
        self.assertTupleEqual(graph.instance.shape, self.expl_shape)
        self.assertEqual(np.sum(graph.cost_weights), 1)
        self.assertNotEqual(len(graph.shifts), 0)
        # all initialized to infinity
        # self.assertTrue(not np.any([graph.dists[graph.dists < np.inf]]))
        # start point was set to normalized value
        self.assertEqual(
            0.8, graph.dists[0, self.start_inds[0], self.start_inds[1]]
        )
        # all start dists have same value
        self.assertEqual(
            len(
                np.unique(
                    graph.dists[:, self.start_inds[0], self.start_inds[1]]
                )
            ), 1
        )
        # not all values still inf
        self.assertLess(
            np.min(graph.dists[:, self.dest_inds[0], self.dest_inds[1]]),
            np.inf
        )
        # get actual best path
        path, path_costs, cost_sum = graph.get_shortest_path(
            self.start_inds, self.dest_inds
        )
        self.assertNotEqual(len(path), 0)
        self.assertEqual(len(path), len(path_costs))
        self.assertGreaterEqual(cost_sum, 5)
        weighted_costs = np.sum(path_costs, axis=0) * graph.cost_weights
        self.assertEqual(np.sum(weighted_costs), cost_sum)
        for (i, j) in path:
            self.assertEqual(self.working_expl[i, j], 1)

    def test_angle_sp(self) -> None:
        graph = graphs.ImplicitLG(
            np.array([self.example2]),
            np.ones(self.expl_shape),
            n_iters=10,
            verbose=0
        )
        graph = self.build_graph(graph, max_angle_lg=np.pi / 4)
        # assert that destination can NOT be reached
        self.assertFalse(
            np.min(graph.dists[:, self.dest_inds[0], self.dest_inds[1]]) <
            np.inf
        )
        # NEXT TRY: more angles allowed
        graph = graphs.ImplicitLG(
            np.array([self.example2]),
            np.ones(self.expl_shape),
            n_iters=10,
            verbose=0
        )
        graph = self.build_graph(graph, max_angle_lg=np.pi)
        # assert that dest CAN be reached
        self.assertTrue(
            np.min(graph.dists[:, self.dest_inds[0], self.dest_inds[1]]) <
            np.inf
        )

        # same with linegraph:
        lg_graph = graphs.LineGraph(
            np.array([self.example2]), np.ones(self.expl_shape), verbose=0
        )
        lg_graph = self.build_graph(lg_graph, max_angle_lg=np.pi)
        lg_graph.sum_costs()
        source_v, target_v = lg_graph.add_start_and_dest(
            self.start_inds, self.dest_inds
        )
        path_lg, path_costs_lg, cost_sum_lg = lg_graph.get_shortest_path(
            source_v, target_v
        )
        # assert that path is non-empty
        self.assertTrue(len(path_lg) > 0)
        self.assertEqual(len(path_lg), len(path_costs_lg))
        self.assertGreaterEqual(cost_sum_lg, 5)
        weighted_costs = np.sum(path_costs_lg, axis=0) * lg_graph.cost_weights
        self.assertEqual(np.sum(weighted_costs), cost_sum_lg)
        # assert that all points are equal to 1
        for (i, j) in path_lg:
            self.assertEqual(self.example2[i, j], 1)

        lg_graph = graphs.LineGraph(
            np.array([self.example2]), np.ones(self.expl_shape), verbose=0
        )
        lg_graph = self.build_graph(lg_graph, max_angle_lg=np.pi / 4)
        lg_graph.sum_costs()
        source_v, target_v = lg_graph.add_start_and_dest(
            self.start_inds, self.dest_inds
        )
        path_lg, path_costs_lg, cost_sum_lg = lg_graph.get_shortest_path(
            source_v, target_v
        )
        self.assertTrue(len(path_lg) == 0)


if __name__ == '__main__':
    unittest.main()
