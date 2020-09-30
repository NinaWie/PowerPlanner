from power_planner.utils.utils import (
    get_half_donut, angle, discrete_angle_costs, bresenham_line, angle_360
)
from power_planner.utils.utils_costs import CostUtils
from power_planner.utils.utils_ksp import KspUtils
from power_planner.plotting import plot_pareto_scatter_3d
from power_planner.graphs.fast_shortest_path import (
    sp_dag, sp_dag_reversed, topological_sort_jit, del_after_dest, edge_costs,
    average_lcp, sp_bf, efficient_update_sp
)
import warnings
import numpy as np
import time
import pickle
from numba.typed import List
import matplotlib.pyplot as plt


class ImplicitLG():

    def __init__(
        self,
        instance,
        instance_corr,
        edge_instance=None,
        directed=True,
        verbose=1,
        n_iters=50
    ):
        self.cost_instance = instance
        self.hard_constraints = instance_corr
        if edge_instance is None:
            self.edge_cost_instance = instance.copy()
        else:
            self.edge_cost_instance = edge_instance
        self.x_len, self.y_len = instance_corr.shape
        self.n_iters = n_iters
        self.time_logs = {}
        self.verbose = verbose
        self.directed = directed

        # construct cost rest
        inf_corr = np.absolute(1 - self.hard_constraints).astype(float)
        inf_corr[inf_corr > 0] = np.inf
        self.cost_rest = self.cost_instance + inf_corr

    def _precompute_angles(self):
        tic = time.time()
        angles_all = np.zeros((len(self.shifts), len(self.shifts)))
        angles_all += np.inf
        for i in range(len(self.shifts)):
            for j, s in enumerate(self.shifts):
                ang = angle(s, self.shifts[i])
                if ang <= self.angle_norm_factor:
                    angles_all[i, j] = discrete_angle_costs(
                        ang, self.angle_norm_factor
                    )
        self.time_logs["compute_angles"] = round(time.time() - tic, 3)
        # multiply with angle weights, need to prevent that not inf * 0
        angles_all[angles_all < np.inf
                   ] = angles_all[angles_all < np.inf] * self.angle_weight
        return angles_all

    def set_shift(
        self,
        start,
        dest,
        pylon_dist_min=3,
        pylon_dist_max=5,
        max_angle=np.pi / 2,
        max_angle_lg=np.pi / 2,
        **kwargs
    ):
        """
        Initialize shift variable by getting the donut values
        :param pylon_dist_min, pylon_dist_max: min and max distance of pylons
        :param vec: vector of diretion of edges
        :param max_angle: Maximum angle of edges to vec
        """
        self.start_inds = np.asarray(start)
        self.dest_inds = np.asarray(dest)
        self.angle_norm_factor = max_angle_lg
        vec = self.dest_inds - self.start_inds
        shifts = get_half_donut(
            pylon_dist_min, pylon_dist_max, vec, angle_max=max_angle
        )
        shift_angles = [angle_360(s, vec) for s in shifts]
        # sort the shifts
        self.shifts = np.asarray(shifts)[np.argsort(shift_angles)]
        self.shift_tuples = self.shifts

        # construct bresenham lines
        shift_lines = List()
        for shift in self.shifts:
            line = bresenham_line(0, 0, shift[0], shift[1])
            shift_lines.append(np.array(line[1:-1]))
        self.shift_lines = shift_lines

    def add_nodes(self):
        tic = time.time()
        # SORT --> Make stack
        tmp_list = self._helper_list()
        visit_points = (self.instance < np.inf).astype(int)
        stack = topological_sort_jit(
            self.dest_inds[0], self.dest_inds[1],
            np.asarray(self.shifts) * (-1), visit_points, tmp_list
        )
        stack = del_after_dest(stack, self.start_inds[0], self.start_inds[1])
        if self.verbose:
            print("time topo sort:", round(time.time() - tic, 3))
            print("stack length", len(stack))
        tic = time.time()

        self.stack_array = np.array(stack)
        self.dists = np.zeros(
            (len(self.stack_array), len(self.shifts))
        ) + np.inf
        # self.dists = np.concatenate((self.stack_array, amend), axis=1)
        self.dists[0, :] = self.instance[tuple(self.start_inds)]
        self.pos2node = (np.zeros(self.instance.shape) -
                         1).astype(int)  # -1 for the unfilled ones
        # make mapping to position
        for i in range(len(self.stack_array)):
            (x, y) = tuple(self.stack_array[i])
            self.pos2node[x, y] = i

        # self.dists = np.zeros((len(self.shifts), self.x_len, self.y_len))
        # self.dists += np.inf
        # i, j = self.start_inds
        # self.dists[:, i, j] = self.instance[i, j]
        self.preds = np.zeros(self.dists.shape) - 1
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)
        self.n_pixels = self.x_len * self.y_len
        self.n_nodes = len(self.stack_array)
        self.n_edges = len(self.shifts) * len(self.dists)
        if self.verbose:
            print("memory taken (dists shape):", self.n_edges)

    def set_corridor(
        self, corridor, start_inds, dest_inds, sample_func="mean",
        sample_method="simple", factor_or_n_edges=1
    ):  # yapf: disable
        # assert factor_or_n_edges == 1, "pipeline not implemented yet"
        corridor = (corridor > 0).astype(int) * (self.hard_constraints >
                                                 0).astype(int)
        inf_corr = np.absolute(1 - corridor).astype(float)
        inf_corr[inf_corr > 0] = np.inf

        self.factor = factor_or_n_edges
        self.cost_rest = self.cost_instance + inf_corr
        # downsample
        tic = time.time()
        if self.factor > 1:
            self.cost_rest = CostUtils.inf_downsample(
                self.cost_rest, self.factor
            )

        self.time_logs["downsample"] = round(time.time() - tic, 3)

        # repeat because edge artifacts
        self.cost_rest = self.cost_rest + inf_corr

        # add start and end TODO ugly
        self.cost_rest[:, dest_inds[0], dest_inds[1]
                       ] = self.cost_instance[:, dest_inds[0], dest_inds[1]]
        self.cost_rest[:, start_inds[0], start_inds[1]
                       ] = self.cost_instance[:, start_inds[0], start_inds[1]]

        self.start_inds = start_inds
        self.dest_inds = dest_inds

    def set_edge_costs(
        self,
        layer_classes=["resistance"],
        class_weights=[1],
        angle_weight=0.1,
        **kwargs
    ):
        """
        angle_weight: how to consider angles in contrast to all other costs!
        """
        tic = time.time()
        assert len(layer_classes) == len(
            class_weights
        ), f"classes ({len(layer_classes)}) and\
            weights({len(class_weights)}) must be of same length!"

        assert len(layer_classes) == len(
            self.cost_rest
        ), f"classes ({len(layer_classes)}) and\
            instance layers ({len(self.cost_rest)}) must be of same length!"

        # set weights and add angle weight
        self.cost_classes = ["angle"] + list(layer_classes)
        ang_weight_norm = angle_weight * np.sum(class_weights)
        self.cost_weights = np.array([ang_weight_norm] + list(class_weights))
        # print("class weights", class_weights)
        self.cost_weights = self.cost_weights / np.sum(self.cost_weights)
        if self.verbose:
            print("cost weights", self.cost_weights)

        # set angle weight and already multiply with angles
        self.angle_weight = self.cost_weights[0]
        # in precomute angles, it is multiplied with angle weights
        self.angle_cost_array = self._precompute_angles()

        # define instance by weighted sum
        self.instance = np.sum(
            np.moveaxis(self.cost_rest, 0, -1) * self.cost_weights[1:], axis=2
        )
        # if one weight is zero, have to correct 0*inf errors
        if np.any(np.isnan(self.instance)):
            self.instance[np.isnan(self.instance)] = np.inf

        self.edge_inst = np.sum(
            np.moveaxis(self.edge_cost_instance, 0, -1) *
            self.cost_weights[1:],
            axis=2
        )
        # dirty_extend = self.edge_inst.copy()
        # x_len, y_len = self.edge_inst.shape
        # for i in range(1, x_len - 1):
        #     for j in range(1, y_len - 1):
        #         if np.any(self.edge_inst[i - 1:i + 2, j - 1:j + 2] == np.inf):
        #             dirty_extend[i, j] = np.inf
        # self.edge_inst = dirty_extend
        self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
        if self.verbose:
            print("instance shape", self.instance.shape)

    # --------------------------------------------------------------------
    # SHORTEST PATH COMPUTATION

    def add_edges(self, mode="DAG", iters=100, edge_weight=0, **kwargs):
        self.edge_weight = edge_weight
        shift_norms = np.array([np.linalg.norm(s) for s in self.shifts])
        if np.any(shift_norms == 1):
            warnings.warn("Raster approach, EDGE WEIGHT IS SET TO ZERO")
            self.edge_weight = 0

        shift_norms = [np.linalg.norm(s) for s in self.shifts]
        tic = time.time()
        # precompute edge costs
        if self.edge_weight > 0:
            self.edge_cost = np.zeros(self.preds.shape) + np.inf
            self.edge_cost = edge_costs(
                self.stack_array,
                self.pos2node, np.array(self.shifts), self.edge_cost,
                self.edge_inst.copy(), self.shift_lines, self.edge_weight
            )
            if self.verbose:
                print("Computed edge instance")
        else:
            self.edge_cost = np.zeros(self.preds.shape)
        # RUN - either directed acyclic or BF algorithm
        if mode == "BF":
            # TODO: nr iterations argument
            self.dists, self.preds = sp_bf(
                iters, self.stack_array, np.array(self.shifts),
                self.angle_cost_array, self.dists, self.preds, self.instance,
                self.edge_cost
            )
        elif mode == "DAG":
            self.dists, self.preds = sp_dag(
                self.stack_array, self.pos2node, np.array(self.shifts),
                self.angle_cost_array, self.dists, self.preds, self.instance,
                self.edge_cost
            )
        else:
            raise ValueError("wrong mode input: " + mode)
        # print(np.min(self.dists[:, self.dest_inds[0], self.dest_inds[1]]))
        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time edges:", round(time.time() - tic, 3))

    # ----------------------------------------------------------------------
    # SHORTEST PATH TREE

    def get_shortest_path_tree(self, source, target):
        """
        Compute costs from dest to all edges
        """
        tic = time.time()

        # initialize dists array
        self.dists_ba = np.zeros(self.dists.shape) + np.inf
        # this time need to set all incoming edges of dest to zero
        # d0, d1 = self.dest_inds
        # for s, (i, j) in enumerate(self.shifts):
        #     pos_index = self.pos2node[d0 + i, d1 + j]
        #     self.dists_ba[pos_index, s] = 0

        # compute distances: new method because out edges instead of in
        self.dists_ba, self.preds_ba = sp_dag_reversed(
            self.stack_array, self.pos2node,
            np.array(self.shifts) * (-1), self.angle_cost_array, self.dists_ba,
            self.instance, self.edge_cost, self.shift_lines, self.edge_weight
        )
        self.time_logs["shortest_path_tree"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time shortest_path_tree:", round(time.time() - tic, 3))
        # self._display_dists(self.dists_ba)
        # distance in ba: take IN edges to source, by computing in neighbors
        # take their first dim value (out edge to source) + source val
        (s0, s1) = self.start_inds
        start_dests = []
        for s, (i, j) in enumerate(self.shifts):
            ind = self.pos2node[s0 + i, s1 + j]
            if ind >= 0:
                start_dests.append(self.dists_ba[ind, s])
            else:
                start_dests.append(np.inf)
        d_ba_arg = np.argmin(start_dests)
        (i, j) = self.shifts[d_ba_arg]
        d_ba = self.dists_ba[self.
                             pos2node[s0 + i, s1 +
                                      j], d_ba_arg] + self.instance[s0, s1]

        d_ab = np.min(self.dists[self.pos2node[tuple(self.dest_inds)], :])
        assert np.isclose(
            d_ba, d_ab
        ), "start to dest != dest to start " + str(d_ab) + " " + str(d_ba)
        # compute best path
        self.best_path = np.array(
            KspUtils.get_sp_dest_shift(
                self.dists_ba,
                self.preds_ba,
                self.pos2node,
                self.dest_inds,
                self.start_inds,
                np.array(self.shifts) * (-1),
                d_ba_arg,
                dest_edge=True
            )
        )
        # assert np.all(self.best_path == np.array(self.sp)), "paths differ"

    def _combined_paths(self, start, dest, best_shift, best_edge):
        """
        Compute path through one specific edge (with bi-directed predecessors)

        Arguments:
            start: overall source vertex
            dest: overall target vertex
            best_shift: the neighbor index of the edge
            best_edge: the vertex of the edge
        """
        # compute path from start to middle point - incoming edge
        best_edge = np.array(best_edge)
        path_ac = KspUtils.get_sp_start_shift(
            self.dists, self.preds, self.pos2node, start, best_edge,
            np.array(self.shifts), best_shift
        )
        # compute path from middle point to dest - outgoing edge
        path_cb = KspUtils.get_sp_dest_shift(
            self.dists_ba, self.preds_ba, self.pos2node, dest, best_edge,
            np.array(self.shifts) * (-1), best_shift
        )
        # concatenate
        together = np.concatenate(
            (np.flip(np.array(path_ac), axis=0), np.array(path_cb)[1:]),
            axis=0
        )
        return together

    # ---------------------------------------------------------------------
    # Functions to output path (backtrack) and corresponding costs

    def transform_path(self, path):
        path_costs = np.array(
            [self.cost_instance[:, p[0], p[1]] for p in path]
        )
        # include angle costs
        ang_costs = CostUtils.compute_angle_costs(path, self.angle_norm_factor)
        # prevent that inf * 0 if zero edge weight
        edge_costs = 0
        if self.edge_weight != 0:
            edge_costs = CostUtils.compute_edge_costs(path, self.edge_inst)
        # print("unweighted edge costs", np.sum(edge_costs))
        path_costs = np.concatenate(
            (np.swapaxes(np.array([ang_costs]), 1, 0), path_costs), axis=1
        )
        cost_sum = np.dot(
            self.cost_weights, np.sum(np.array(path_costs), axis=0)
        ) + np.sum(edge_costs) * self.edge_weight
        # cost_sum = np.dot(
        #     self.class_weights, np.sum(np.array(path_costs), axis=0)
        # )  # scalar: weighted sum of the summed class costs
        return np.asarray(path
                          ).tolist(), path_costs.tolist(), cost_sum.tolist()

    def _display_dists(self, edge_array, func=np.min):
        arr = np.zeros(self.pos2node.shape)
        for i in range(len(self.pos2node)):
            for j in range(len(self.pos2node[0])):
                ind = self.pos2node[i, j]
                if ind >= 0:
                    arr[i, j] = func(edge_array[ind, :])
        plt.imshow(arr)
        plt.savefig("dists_ba_view.png")

    def get_shortest_path(self, start_inds, dest_inds, ret_only_path=False):
        dest_ind_stack = self.pos2node[tuple(dest_inds)]
        if not np.any(self.dists[dest_ind_stack, :] < np.inf):
            warnings.warn("empty path")
            return [], [], 0
        tic = time.time()
        curr_point = dest_inds
        path = [dest_inds]
        # first minimum: angles don't matter, just min of in-edges
        min_shift = np.argmin(self.dists[dest_ind_stack, :])
        # track back until start inds
        while np.any(curr_point - start_inds):
            new_point = curr_point - self.shifts[int(min_shift)]
            # get new shift from argmins
            curr_ind_stack = self.pos2node[tuple(curr_point)]
            min_shift = self.preds[curr_ind_stack, int(min_shift)]
            if min_shift == -1:
                print(path)
                raise RuntimeError("Problem! predecessor -1!")
            path.append(new_point)
            curr_point = new_point

        path = np.flip(np.asarray(path), axis=0)
        if ret_only_path:
            return path
        self.sp = path
        self.time_logs["path"] = round(time.time() - tic, 3)
        return self.transform_path(path)

    # ----------------------------------------------------------------------
    # Other auxiliary functions
    def save_graph(self, out_path):
        with open(out_path + ".dat", "wb") as outfile:
            pickle.dump((self.dists, self.preds), outfile)

    def _helper_list(self):
        tmp_list = List()
        tmp_list_inner = List()
        tmp_list_inner.append(0)
        tmp_list_inner.append(0)
        tmp_list.append(tmp_list_inner)
        return tmp_list

    def add_start_and_dest(self, source, dest):
        # here simply return the indices for start and destination
        return source, dest

    def sum_costs(self):
        pass

    def remove_vertices(self, dist_surface, delete_padding=0):
        pass

    # -----------------------------------------------------------------------
    # INTERFACE

    def single_sp(self, power=1, **kwargs):
        """
        Function for full processing until shortest path
        """
        self.start_inds = kwargs["start_inds"]
        self.dest_inds = kwargs["dest_inds"]
        self.set_shift(self.start_inds, self.dest_inds, **kwargs)
        if self.verbose:
            print("1) Initialize shifts and instance (corridor)")
        self.set_edge_costs(**kwargs)
        self.instance = self.instance**power
        # add vertices
        self.add_nodes()
        if self.verbose:
            print("2) Initialize distances to inf and predecessors")
        self.add_edges(**kwargs)
        if self.verbose:
            print("3) Compute source shortest path tree")
            print("number of vertices and edges:", self.n_nodes, self.n_edges)

        # get actual best path
        path, path_costs, cost_sum = self.get_shortest_path(
            self.start_inds, self.dest_inds
        )
        if self.verbose:
            print("4) shortest path", cost_sum)
        return path, path_costs, cost_sum

    def sp_trees(self, **kwargs):
        start_inds = kwargs["start_inds"]
        dest_inds = kwargs["dest_inds"]
        path, path_costs, cost_sum = self.single_sp(**kwargs)
        self.get_shortest_path_tree(start_inds, dest_inds)
        return path, path_costs, cost_sum

    def pareto(self, save_img_path=None, **kwargs):
        """
        vary: dictionary of the form
            "var_name":[1,2,3]
            where var_name refers to the variable to change and the list
            specificies the possible values
        cost_names: The names of the classes to be compared
        """
        pareto, weight_list, cost_sum = [], [], []
        compare_names = ["angle", "edge_costs", "resistance"]
        angle_weights = [0.1, 0.3]  # , 0.6, 0.9]
        edge_weights = [0.2, 0.5]  # , 0.8, 1.5, 2.0]
        minimal_weights = np.array(
            [np.min(angle_weights),
             np.min(edge_weights)]
        )
        maximal_weights = np.array(
            [np.max(angle_weights),
             np.max(edge_weights)]
        )
        # iterate over combinations
        for a_w in angle_weights:
            for e_w in edge_weights:
                kwargs["ANGLE_WEIGHT"] = a_w
                kwargs["EDGE_WEIGHT"] = e_w
                path, _, _ = self.single_sp(**kwargs)

                # get path costs
                path_cost_raw, column_names = self.raw_path_costs(path)
                path_cost_table = np.array(
                    [
                        path_cost_raw[:, column_names.index(comp_name)]
                        for comp_name in compare_names
                    ]
                )
                pareto.append(np.sum(path_cost_table, axis=1))
                cost_sum.append(np.sum(path_cost_table))

                # cumbersome transformation of weights
                normed_weights = (np.array([a_w, e_w]) - minimal_weights
                                  ) / (maximal_weights - minimal_weights)
                weights = list(normed_weights)
                weights.append(np.max([1.5 - np.sum(weights), 0]))
                weights = weights / np.sum(weights)
                weight_list.append(np.array(weights))
        pareto = np.asarray(pareto)
        weight_list = np.asarray(weight_list)

        # save as pickle
        with open(save_img_path + "_pareto_data.dat", "wb") as outfile:
            pickle.dump(
                (pareto, weight_list, compare_names, cost_sum), outfile
            )

        plot_pareto_scatter_3d(
            pareto,
            weight_list,
            compare_names,
            cost_sum=cost_sum,
            out_path=save_img_path
        )
