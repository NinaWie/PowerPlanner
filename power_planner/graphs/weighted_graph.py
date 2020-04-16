from power_planner.utils.utils_constraints import ConstraintUtils
from power_planner.utils.utils import get_donut_vals
from .general_graph import GeneralGraph

import numpy as np
import time


class WeightedGraph(GeneralGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        # assert cost_instance.shape == hard_constraints.shape
        # , "Cost size must be equal to corridor definition size!"

        # time logs
        self.time_logs = {}
        tic = time.time()

        # indicator whether to use networkx or graph tool
        self.graphtool = graphtool

        # cost surface
        self.cost_instance = cost_instance
        self.hard_constraints = hard_constraints
        self.x_len, self.y_len = hard_constraints.shape

        # initialize graph:
        GeneralGraph.__init__(
            self, directed=directed, graphtool=graphtool, verbose=verbose
        )

        # print statements
        self.verbose = verbose

        self.time_logs["init_graph"] = round(time.time() - tic, 3)

        # original pos2node: all filled except for hard constraints
        self.pos2node = np.arange(1, self.x_len * self.y_len + 1).reshape(
            (self.x_len, self.y_len)
        )
        self.pos2node *= (self.hard_constraints > 0).astype(int)
        self.pos2node -= 1
        if self.verbose:
            print("initialized weighted graph pos2node")

    def set_corridor(
        self,
        dist_surface,
        start_inds,
        dest_inds,
        sample_func="mean",
        sample_method="simple",
        factor_or_n_edges=1
    ):
        tic = time.time()
        # set cost rest according to corridor
        GeneralGraph.set_corridor(
            self,
            dist_surface,
            start_inds,
            dest_inds,
            sample_func=sample_func,
            sample_method=sample_method,
            factor_or_n_edges=factor_or_n_edges
        )

        # define pos2node accordingly:
        # corridor = (dist_surface == 0).astype(int)
        # inverted_corridor = np.absolute(1 - corridor).astype(bool)
        inverted_corridor = (dist_surface == 0).astype(bool)
        # set all which are not in the corridor to -1
        self.pos2node[inverted_corridor] = -1
        self.time_logs["set_cost_rest"] = round(time.time() - tic, 3)

    def set_shift(self, lower, upper, vec, max_angle, max_angle_lg=0):
        GeneralGraph.set_shift(self, lower, upper, vec, max_angle)
        self.shift_vals = get_donut_vals(self.shifts, vec)

    def add_nodes(self):
        tic = time.time()
        # add nodes to graph
        n_nodes = self.x_len * self.y_len
        # len(np.unique(self.pos2node))
        GeneralGraph.add_nodes(self, n_nodes)
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)

    def set_cost_rest(self):
        # in case of the non-random graph, we don't have to set cost_rest
        pass

    def _compute_edges(self, shift):
        # inds orig:
        inds_orig = self.pos2node[np.mean(self.cost_rest, axis=0) > 0]
        # switch axes for shift
        cost_rest_switched = np.moveaxis(self.cost_rest, 0, -1)
        # shift by shift
        costs_shifted = ConstraintUtils.shift_surface(
            cost_rest_switched, shift
        )
        # switch axes back
        costs_shifted = np.moveaxis(costs_shifted, -1, 0)

        # must be cost instance because otherwise plus zero values!
        weights = (costs_shifted + self.cost_instance) / 2
        # # WITH EDGE WEIGHTS
        # weights = ConstraintUtils.convolve_faster
        # (self.cost_rest, kernels[i], posneg[i])
        # weights = weights1 + 2 * weights2
        # print(
        #     "max node weights", np.max(weights1), "max edge weights:",
        #     np.max(weights2), "min node weights", np.min(weights1),
        #     "min edge weights:", np.min(weights2)
        # )

        mean_costs_shifted = np.mean(costs_shifted, axis=0) > 0

        inds_shifted = self.pos2node[mean_costs_shifted]

        assert len(inds_shifted) == len(
            inds_orig
        ), "orig:{},shifted:{}".format(len(inds_orig), len(inds_shifted))

        # take weights of the shifted ones
        weights_arr = np.array(
            [w[mean_costs_shifted] for i, w in enumerate(weights)]
        )

        # concatenete indices and weights, select feasible ones
        inds_arr = np.asarray([inds_orig, inds_shifted])
        inds_weights = np.concatenate((inds_arr, weights_arr), axis=0)
        pos_inds = inds_shifted >= 0
        edge_arr_final = np.swapaxes(inds_weights, 1, 0)[pos_inds]

        # remove edges with high costs:
        # first two columns of out are indices
        # weights_arr = np.mean(out[:, 2:], axis=1)
        # weights_mean = np.quantile(weights_arr, 0.9)
        # inds_higher = np.where(weights_arr < weights_mean)
        # out = out[inds_higher[0]]

        return edge_arr_final

    def remove_vertices(self, dist_surface, delete_padding=0):
        """
        Remove edges in a certain corridor (or all) to replace them by
        a refined surface

        @param dist_surface: a surface where each pixel value corresponds to
        the distance of the pixel to the shortest path
        @param delete_padding: define padding in which part of the corridor to
        delete vertices (cannot delete all because then graph unconnected)
        """
        tic = time.time()
        # #Possibility 1: remove all edges of vertices in (smaller) corridor
        # corridor = (dist_surface>delete_padding).astype(int)
        # corr_vertices = self.pos2node * corridor
        # new_vertices = corr_vertices[corr_vertices>0]
        # for v in new_vertices:
        #     self.graph.clear_vertex(self.graph.vertex(v))
        # #Possibility 2: remove all edges --> only considering corridor then
        self.graph.clear_edges()
        self.graph.shrink_to_fit()
        # #Possibility 3: remove all out_edges of corridor vertices
        # corridor = (dist_surface>0).astype(int)
        # corr_vertices = self.pos2node * corridor
        # new_vertices = corr_vertices[corr_vertices>0]
        # remove_property = self.graph.new_edge_property("float")
        # remove_property.a = np.zeros(self.weight.get_array().shape)
        # for v in new_vertices:
        #     for e in self.graph.vertex(v).out_edges():
        #         remove_property[e] = 1
        # remove_labeled_edges(self.graph, remove_property)
        self.time_logs["remove_edges"] = round(time.time() - tic, 3)

    def add_start_and_dest(self, start_inds, dest_inds):
        """
        In this case only get the corresponding vertex (no need to add)
        """
        start_node_ind = self.pos2node[start_inds[0], start_inds[1]]
        dest_node_ind = self.pos2node[dest_inds[0], dest_inds[1]]
        if self.graphtool:
            return self.graph.vertex(start_node_ind
                                     ), self.graph.vertex(dest_node_ind)
        else:
            return start_node_ind, dest_node_ind

    def get_shortest_path(self, source, target):
        """
        Compute shortest path from source vertex to target vertex
        """
        tic = (time.time())
        # #if source and target are given as indices:
        vertices_path = GeneralGraph.get_shortest_path(self, source, target)

        path = []
        for v in vertices_path:
            if self.graphtool:
                ind = self.graph.vertex_index[v]
            else:
                ind = int(v)
            path.append((ind // self.y_len, ind % self.y_len))

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)

        out_costs = [self.cost_instance[:, i, j].tolist() for (i, j) in path]
        cost_sum = np.dot(
            self.cost_weights, np.sum(np.array(out_costs), axis=0)
        )
        return path, out_costs, cost_sum
