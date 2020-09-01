from power_planner.utils.utils import normalize
#  graph imports
from .weighted_ksp import WeightedKSP
from .line_graph import LineGraph

import numpy as np
import time
# import matplotlib.pyplot as plt


class RandomGraph():

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        self.hard_constraints = hard_constraints
        self.cost_instance = cost_instance
        self.time_logs = {}

    def set_corridor(
        self,
        corridor,
        start_inds,
        dest_inds,
        factor_or_n_edges=0,
        mode="gauss"
    ):
        """
        factor: in this case ratio of edges to remove
        """
        tic = time.time()
        # set corridor
        corridor = normalize(
            corridor * (self.hard_constraints > 0).astype(int)
        )

        # AUTO MODE: compute factor from desired number of edges
        if factor_or_n_edges >= 1:
            assert factor_or_n_edges > 10, "cannot reduce edge num by so much"
            # compute factor automatically
            n_nodes_hard = len(corridor[corridor > 0])
            # edges_approx is the number of edges we would get if we take all
            n_edges_approx = len(self.shift_tuples) * n_nodes_hard
            if self.verbose:
                print(
                    "Desired edges", factor_or_n_edges, "n nodes in corridor",
                    n_nodes_hard, "approximate_edges", n_edges_approx
                )
            # ratio of edges to delete x
            ratio_keep = (factor_or_n_edges / n_edges_approx)
            # min because might happen that corridor has less edges anyways
            factor = max([1 - ratio_keep, 0])
            if self.verbose:
                print("Ratio of edges to remove :", round(factor, 2))
            self.factor = factor
        # NOT AUTO
        else:
            factor = factor_or_n_edges
            self.factor = factor_or_n_edges

        # first case: keep all edges
        if self.factor == 0:
            self.corridor = (corridor > 0).astype(int) * 1.1
            self.time_logs["downsample"] = round(time.time() - tic, 3)
            return 0

        # Different strategies to construct corridor prob distribution
        if self.verbose:
            print("MODE", mode)
        if mode == "gauss":
            gauss = lambda x: np.exp(-(x - 1)**2 / (0.5))
            corridor = normalize(gauss(corridor))
        elif mode == "same":
            corridor = corridor
        elif mode == "squared":
            corridor = corridor**2
        elif mode == "squareroot":
            corridor = np.sqrt(corridor)
        else:
            raise NotImplementedError("mode must be gauss, squared...")
        # compute current mean of nonzero values
        mean_val_now = np.mean(corridor[corridor > 0])
        # current number entries
        # n_entris = len(corridor[corridor > 0])
        if mean_val_now > 1 - factor:
            # make mean smaller
            scale_factor = (1 - factor) / mean_val_now
            corridor = corridor * scale_factor
        else:
            # make mean larger
            scale_factor = factor / (1 - mean_val_now)
            larger_zero = (corridor > 0).astype(int)
            corridor = 1 - scale_factor + corridor * scale_factor
            # have to reset to 0
            corridor = corridor * larger_zero
        # # TEST OUTPUTS:
        # print("test corridor outputs: mean")
        # print(np.mean(corridor[corridor > 0]), "should be", (1 - factor))
        # arr = np.random.rand(*corridor.shape)
        # corr_thresh = (corridor > arr).astype(int)
        # leftover = len(corr_thresh[corr_thresh > 0])
        # print(
        #     "RATIO DEL", (n_entris - leftover) / n_entris, "shouldbe",factor
        # )
        self.corridor = corridor * 1.1

        self.time_logs["downsample"] = round(time.time() - tic, 3)

    def set_cost_rest(self):
        prob_arr = np.random.rand(*self.corridor.shape)
        prob_arr = (self.corridor > prob_arr).astype(int)
        self.cost_rest = self.cost_instance * prob_arr


class RandomWeightedGraph(WeightedKSP, RandomGraph):

    def __init__(
        self, cost_instance, hard_constraints, directed=True, verbose=1
    ):
        self.corridor = hard_constraints * 1.1
        super(RandomWeightedGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            verbose=verbose
        )

    def set_corridor(
        self,
        corridor,
        start_inds,
        dest_inds,
        factor_or_n_edges=0,
        mode="gauss",
        **kwargs
    ):
        """
        factor: in this case ratio of edges to remove
        """
        RandomGraph.set_corridor(
            self,
            corridor,
            start_inds,
            dest_inds,
            factor_or_n_edges=factor_or_n_edges,
            mode=mode
        )

    def set_cost_rest(self):
        RandomGraph.set_cost_rest(self)


class RandomLineGraph(LineGraph, RandomGraph):

    def __init__(
        self, cost_instance, hard_constraints, directed=True, verbose=1
    ):
        # initialize corridor to all
        self.corridor = hard_constraints * 1.1
        super(RandomLineGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            verbose=verbose
        )

    def set_corridor(
        self,
        corridor,
        start_inds,
        dest_inds,
        factor_or_n_edges=0,
        mode="gauss"
    ):
        """
        factor: in this case ratio of edges to remove
        """
        RandomGraph.set_corridor(
            self,
            corridor,
            start_inds,
            dest_inds,
            factor_or_n_edges=factor_or_n_edges,
            mode=mode
        )

    def set_cost_rest(self):
        RandomGraph.set_cost_rest(self)
