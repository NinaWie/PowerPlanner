import unittest
import numpy as np

from power_planner.graphs.weighted_ksp import WeightedKSP
from power_planner.graphs.impl_ksp import ImplicitKSP


class TestKsp(unittest.TestCase):

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
        graph.sum_costs()
        source_v, target_v = graph.add_start_and_dest(
            self.start_inds, self.dest_inds
        )
        return graph, source_v, target_v

    def test_ksp(self) -> None:
        wg = WeightedKSP(np.array([self.example3]), self.hard_cons, verbose=0)
        wg, source_v, target_v = self.build_graph(wg, ang_weight=0)
        bestpath, _, best_cost_sum = wg.get_shortest_path(source_v, target_v)
        # self.assertListEqual(bestpath.)
        wg.get_shortest_path_tree(source_v, target_v)
        best2, _, best_cost_sum2 = wg.transform_path(wg.best_path)
        # assert that SP tree path is optimal one
        for b in range(len(best2)):
            self.assertListEqual(list(best2[b]), list(bestpath[b]))
        self.assertEqual(best_cost_sum, best_cost_sum2)
        # TEST DIVERSE
        ksp = wg.dispersion_ksp(
            source_v,
            target_v,
            9,
            cost_thresh=1.05,
            dist_mode="eucl_mean",
            count_thresh=3
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
        impl_lg = ImplicitKSP(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        impl_lg, _, _ = self.build_graph(
            impl_lg, max_angle_lg=max_angle_lg, ang_weight=0
        )
        impl_lg.get_shortest_path_tree(self.start_inds, self.dest_inds)
        ksp_lg = impl_lg.find_ksp(self.start_inds, self.dest_inds, 10)
        # get weighted ksp
        wg_graph = WeightedKSP(
            np.array([self.example3]), self.hard_cons, verbose=0
        )
        wg_graph, source_v, target_v = self.build_graph(
            wg_graph, max_angle_lg=max_angle_lg
        )
        wg_graph.get_shortest_path_tree(source_v, target_v)
        ksp_wg = wg_graph.k_shortest_paths(source_v, target_v, 10)
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
