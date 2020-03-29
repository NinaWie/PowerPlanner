from power_planner.constraints import ConstraintUtils
from power_planner.utils import get_donut_vals, normalize
from weighted_graph import WeightedGraph
from power_planner.utils_instance import CostUtils

import numpy as np
from graph_tool.all import Graph, shortest_path, remove_labeled_edges
import time
import networkx as nx
import matplotlib.pyplot as plt


class RandomGraph(WeightedGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(RandomGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )

    def set_cost_rest(self, factor, corridor, start_inds, dest_inds):
        """
        factor: in this case ratio of edges to remove
        """
        assert factor < 1, "for RandomGraph factor must be smaller 1"
        self.factor = factor
        # set pos2node
        self.pos2node = self.pos2node_orig
        # set corridor: exp of distances
        corridor = corridor * (self.hard_constraints > 0).astype(int)
        if self.factor == 0:
            self.corridor = (corridor > 0).astype(int)  # binarize
            self.cutoff = 0.5
        else:
            self.corridor = normalize(np.exp(normalize(corridor)))
            # set cutoff
            cutoff = np.quantile(self.corridor, factor)
            self.cutoff = max([0.5, cutoff])  # must be at least 0.5!
        print("max min corridor", np.max(corridor), np.min(corridor))
        print("cutoff corridor vals", self.cutoff)

    def add_edges(self):
        """
        overwrite method to randomly use edges
        """

        tic_function = time.time()

        n_edges = 0
        # kernels, posneg = ConstraintUtils.get_kernel(self.shifts,
        # self.shift_vals)
        # edge_array = []

        times_edge_list = []
        times_add_edges = []

        if self.verbose:
            print("n_neighbors:", len(self.shifts))

        for i in range(len(self.shifts)):

            prob_arr = np.random.rand(*self.corridor.shape) - 0.5 + self.cutoff
            prob_arr = (self.corridor > prob_arr).astype(int)

            self.cost_rest = self.cost_instance * prob_arr

            inds_orig = self.pos2node[prob_arr > 0]  # changed!

            tic_edges = time.time()

            # compute shift and weights
            inds_shifted, weights_arr = self._compute_edge_costs(i)
            assert len(inds_shifted) == len(
                inds_orig
            ), "orig:{},shifted:{}".format(len(inds_orig), len(inds_shifted))

            # concatenete indices and weights, select feasible ones
            inds_arr = np.asarray([inds_orig, inds_shifted])
            inds_weights = np.concatenate((inds_arr, weights_arr), axis=0)
            pos_inds = inds_shifted >= 0
            out = np.swapaxes(inds_weights, 1, 0)[pos_inds]

            # remove edges with high costs:
            # first two columns of out are indices
            # weights_arr = np.mean(out[:, 2:], axis=1)
            # weights_mean = np.quantile(weights_arr, 0.9)
            # inds_higher = np.where(weights_arr < weights_mean)
            # out = out[inds_higher[0]]

            # Error if -1 entries because graph-tool crashes with -1 nodes
            if np.any(out[:2].flatten() == -1):
                print(np.where(out[:2] == -1))
                raise RuntimeError

            n_edges += len(out)
            times_edge_list.append(round(time.time() - tic_edges, 3))

            # add edges to graph
            tic_graph = time.time()
            if self.graphtool:
                self.graph.add_edge_list(out, eprops=self.cost_props)
            else:
                nx_edge_list = [(e[0], e[1], {"weight": e[2]}) for e in out]
                self.graph.add_edges_from(nx_edge_list)
            times_add_edges.append(round(time.time() - tic_graph, 3))

            # alternative: collect edges here and add alltogether
            # edge_array.append(out)

        self._update_time_logs(times_add_edges, times_edge_list, tic_function)
        if self.verbose:
            print("DONE adding", n_edges, "edges:", time.time() - tic_function)