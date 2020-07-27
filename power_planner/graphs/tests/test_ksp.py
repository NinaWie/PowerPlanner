import unittest
import numpy as np
from types import SimpleNamespace
from power_planner.graphs.weighted_ksp import WeightedKSP
from power_planner.graphs.implicit_lg import ImplicitLG
from power_planner.ksp import KSP


class TestKsp(unittest.TestCase):

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

    # construct random cost surface to assure that lg and impl output same
    example3 = np.random.rand(*expl_shape)

    # constuct hard_cons with padding
    hard_cons = np.ones(expl_shape)
    hard_cons[:, :5] = 0
    hard_cons[:, -5:] = 0
    hard_cons[:5, :] = 0
    hard_cons[-5:, :] = 0

    def test_ksp(self) -> None:
        wg = WeightedKSP(np.array([self.example3]), self.hard_cons, verbose=0)
        bestpath, _, best_cost_sum = wg.single_sp(**vars(self.cfg))
        wg.get_shortest_path_tree()
        best2, _, best_cost_sum2 = wg.transform_path(wg.best_path)
        # assert that SP tree path is optimal one
        for b in range(len(best2)):
            self.assertListEqual(list(best2[b]), list(bestpath[b]))
        self.assertEqual(best_cost_sum, best_cost_sum2)
        # TEST DIVERSE
        ksp = wg.dispersion_ksp(
            9, cost_thresh=1.05, dist_mode="eucl_mean", count_thresh=3
        )
        for k in ksp:
            path = k[0]
            self.assertListEqual(list(self.start_inds), list(path[0]))
            self.assertListEqual(list(self.dest_inds), list(path[-1]))
            cost = k[2]
            # print(k[1])
            self.assertLessEqual(cost, best_cost_sum * 1.05)
        # TEST LC KSP
        # ksp = graph.k_shortest_paths(source_v, target_v, cfg.KSP)
        # TODO

    def compare_ksp(self) -> None:
        max_angle_lg = np.pi
        # get impl lg ksp
        impl_lg = ImplicitLG(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        _ = impl_lg.sp_trees(**vars(self.cfg))
        ksp = KSP(impl_lg)
        ksp_lg = ksp.find_ksp(10)
        # get weighted ksp
        wg_graph = WeightedKSP(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        bestpath, _, best_cost_sum = wg.single_sp(**vars(self.cfg))
        wg_graph.get_shortest_path_tree()
        ksp_wg = wg_graph.find_ksp(10)
        for k in range(10):
            path1 = ksp_wg[k][0]
            path2 = ksp_lg[k][0]
            for p in range(len(path1)):
                self.assertListEqual(list(path1[p]), list(path2[p]))

    # # test that all fulfill hard constraints
    # paths = [k[0] for k in ksp]
    # for p in paths:
    #     p = np.array(p)
    #     plt.scatter(p[:,1], p[:,0])
    #     for (i,j) in p:
    #         self.assertEqual(instance_corr[i,j],0)


if __name__ == "__main__":
    unittest.main()
