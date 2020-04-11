from power_planner.constraints import ConstraintUtils
from power_planner.utils import get_donut_vals, normalize
from power_planner.utils_instance import CostUtils
#  graph imports
from .weighted_ksp import WeightedKSP
from .line_graph import LineGraph

import numpy as np
from graph_tool.all import Graph, shortest_path, remove_labeled_edges
import time
import networkx as nx
import matplotlib.pyplot as plt


class RandomGraph():

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        print("init of RandomGraph")

    def set_corridor(self, factor, corridor, start_inds, dest_inds):
        """
        factor: in this case ratio of edges to remove
        """
        tic = time.time()
        assert factor < 1, "for RandomGraph factor must be smaller 1"
        self.factor = factor

        # set corridor: exp of distances
        corridor = corridor * (self.hard_constraints > 0).astype(int)
        if self.factor == 0:
            # make sure all points in corridor are taken
            self.corridor = (corridor > 0).astype(int) * 1.1
        else:
            corridor = normalize(corridor)**2
            corr_greater_zero = corridor[corridor > 0]
            cutoff = np.quantile(corr_greater_zero, factor)
            if cutoff == 1:
                self.corridor = corridor - factor
            else:
                self.corridor = corridor + 0.5 - max([cutoff, 0.5])
                # set cutoff # normalize(np.exp(
                # self.cutoff = max([0.5, cutoff])  # must be at least 0.5!
                # - 0.5 + self.cutoff
            print(
                "max min corridor", np.max(self.corridor),
                np.min(self.corridor)
            )
            print("cutoff corridor vals", cutoff)
        self.time_logs["downsample"] = round(time.time() - tic, 3)

    def set_cost_rest(self):
        prob_arr = np.random.rand(*self.corridor.shape)
        prob_arr = (self.corridor > prob_arr).astype(int)
        self.cost_rest = self.cost_instance * prob_arr


class RandomWeightedGraph(WeightedKSP, RandomGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(RandomWeightedGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )

    def set_corridor(self, factor, corridor, start_inds, dest_inds):
        """
        factor: in this case ratio of edges to remove
        """
        RandomGraph.set_corridor(self, factor, corridor, start_inds, dest_inds)

    def set_cost_rest(self):
        RandomGraph.set_cost_rest(self)


class RandomLineGraph(LineGraph, RandomGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(RandomLineGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )

    def set_corridor(self, factor, corridor, start_inds, dest_inds):
        """
        factor: in this case ratio of edges to remove
        """
        RandomGraph.set_corridor(self, factor, corridor, start_inds, dest_inds)

    def set_cost_rest(self):
        RandomGraph.set_cost_rest(self)
