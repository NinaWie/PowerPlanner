import numpy as np
import unittest
from power_planner import graphs
from power_planner.utils.utils import bresenham_line


class TestCompGraphs(unittest.TestCase):
    # define params for test instance
    expl_shape = (100, 100)
    start_inds = np.array([6, 6])
    dest_inds = np.array([90, 85])

    # construct random cost surface to assure that lg and impl output same
    example3 = np.random.rand(*expl_shape)

    # constuct hard_cons with padding
    hard_cons = np.ones(expl_shape)
    hard_cons[:, :5] = 0
    hard_cons[:, -5:] = 0
    hard_cons[:5, :] = 0
    hard_cons[-5:, :] = 0

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

    def test_lg_equal_weighted(self) -> None:
        # compare whether angle costs are decreasing
        max_angle_lg = np.pi
        impl_lg = graphs.ImplicitLG(
            np.array([self.example3]), self.hard_cons, n_iters=15, verbose=0
        )
        impl_lg = self.build_graph(
            impl_lg, max_angle_lg=max_angle_lg, ang_weight=0
        )
        path, path_costs, cost_sum = impl_lg.get_shortest_path(
            self.start_inds, self.dest_inds
        )
        # get lg path
        lg_graph = graphs.LineGraph(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        lg_graph = self.build_graph(
            lg_graph, max_angle_lg=max_angle_lg, ang_weight=0
        )
        lg_graph.sum_costs()
        source_v, target_v = lg_graph.add_start_and_dest(
            self.start_inds, self.dest_inds
        )
        path_lg, path_costs_lg, cost_sum_lg = lg_graph.get_shortest_path(
            source_v, target_v
        )
        # get weighted path
        wg_graph = graphs.WeightedGraph(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        wg_graph = self.build_graph(wg_graph, max_angle_lg=max_angle_lg)
        wg_graph.sum_costs()
        source_v, target_v = wg_graph.add_start_and_dest(
            self.start_inds, self.dest_inds
        )
        path_wg, path_costs_wg, cost_sum_wg = wg_graph.get_shortest_path(
            source_v, target_v
        )
        # assert equal:
        self.assertListEqual(list(path_lg), list(path_wg))
        costs_lg_wo_angle = (np.array(path_costs_lg)[:, 1]).tolist()
        costs_wo_angle = (np.array(path_costs)[:, 1]).tolist()
        flat_costs_wg = list(np.array(path_costs_wg).flatten())
        self.assertListEqual(costs_lg_wo_angle, flat_costs_wg)
        self.assertListEqual(list(path), list(path_wg))
        self.assertListEqual(costs_wo_angle, flat_costs_wg)
        # COST SUMS ARE NOT EQUAL! - angle costs mit drin
        self.assertAlmostEqual(cost_sum_wg, cost_sum_lg)
        self.assertAlmostEqual(cost_sum_wg, cost_sum)

    def test_equal_to_lg(self) -> None:
        # compare whether angle costs are decreasing
        ang_costs_prev = np.inf
        all_angle_costs = []
        for ang_weight in [0.1, 0.3, 0.5, 0.7]:
            max_angle = np.pi / 2
            impl_lg = graphs.ImplicitLG(
                np.array([self.example3]),
                self.hard_cons,
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
                np.array([self.example3]), self.hard_cons, verbose=0
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


if __name__ == '__main__':
    unittest.main()
