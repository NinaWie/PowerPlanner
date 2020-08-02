import numpy as np
import unittest
from types import SimpleNamespace
from power_planner import graphs
from power_planner.utils.utils import get_distance_surface


class TestCompGraphs(unittest.TestCase):
    # define params for test instance
    expl_shape = (50, 50)
    start_inds = np.array([6, 6])
    dest_inds = np.array([44, 40])

    # create configuration
    cfg = SimpleNamespace()
    cfg.PYLON_DIST_MIN = 3
    cfg.PYLON_DIST_MAX = 5
    cfg.start_inds = start_inds
    cfg.dest_inds = dest_inds
    cfg.ANGLE_WEIGHT = 0.25
    cfg.EDGE_WEIGHT = 0
    cfg.MAX_ANGLE = np.pi / 2
    cfg.MAX_ANGLE_LG = np.pi / 4
    cfg.layer_classes = ["dummy_class"]
    cfg.class_weights = [1]

    # define corridor
    corridor = np.ones(expl_shape) * 0.5

    # construct random cost surface to assure that lg and impl output same
    example3 = np.random.rand(*expl_shape)

    # constuct hard_cons with padding
    hard_cons = np.ones(expl_shape)
    hard_cons[:, :5] = 0
    hard_cons[:, -5:] = 0
    hard_cons[:5, :] = 0
    hard_cons[-5:, :] = 0

    def test_lg_equal_weighted(self) -> None:
        # compare whether angle costs are decreasing
        self.cfg.MAX_ANGLE_LG = np.pi
        impl_lg = graphs.ImplicitLG(
            np.array([self.example3]), self.hard_cons, n_iters=15, verbose=0
        )
        impl_lg.set_corridor(self.corridor, self.start_inds, self.dest_inds)
        self.cfg.ANGLE_WEIGHT = 0
        path, path_costs, cost_sum = impl_lg.single_sp(**vars(self.cfg))
        # get lg path
        lg_graph = graphs.LineGraph(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        lg_graph.set_corridor(self.corridor, self.start_inds, self.dest_inds)
        path_lg, path_costs_lg, cost_sum_lg = lg_graph.single_sp(
            **vars(self.cfg)
        )
        # get weighted path
        wg_graph = graphs.WeightedGraph(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        wg_graph.set_corridor(self.corridor, self.start_inds, self.dest_inds)
        path_wg, path_costs_wg, cost_sum_wg = wg_graph.single_sp(
            **vars(self.cfg)
        )
        # assert equal:
        self.assertListEqual(list(path_lg), list(path_wg))
        costs_lg_wo_angle = (np.array(path_costs_lg)[:, 1]).tolist()
        costs_wo_angle = (np.array(path_costs)[:, 1]).tolist()
        flat_costs_wg = list(np.array(path_costs_wg).flatten())
        self.assertListEqual(costs_lg_wo_angle, flat_costs_wg)
        self.assertListEqual(list(path), list(path_wg))
        self.assertListEqual(costs_wo_angle, flat_costs_wg)
        # COST SUMS ARE EQUAL! - angle costs multiplied by zero
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
            self.cfg.ANGLE_WEIGHT = ang_weight
            self.cfg.MAX_ANGLE_LG = max_angle
            path, path_costs, cost_sum = impl_lg.single_sp(**vars(self.cfg))
            # get lg path
            lg_graph = graphs.LineGraph(
                np.array([self.example3]), self.hard_cons, verbose=0
            )
            path_lg, path_costs_lg, cost_sum_lg = lg_graph.single_sp(
                **vars(self.cfg)
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

    def test_corridor(self) -> None:
        # compare whether angle costs are decreasing
        path_artificial = [[self.start_inds.tolist(), self.dest_inds.tolist()]]
        self.corridor = get_distance_surface(
            self.expl_shape, path_artificial, mode="dilation", n_dilate=20
        )
        self.test_equal_to_lg()
        print("done in method")
        self.corridor = np.ones(self.expl_shape) * 0.5


if __name__ == '__main__':
    unittest.main()
