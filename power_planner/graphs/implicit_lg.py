from power_planner.utils.utils import (
    get_half_donut, angle, discrete_angle_costs, bresenham_line
)
from power_planner.utils.utils_costs import CostUtils
from power_planner.utils.utils_ksp import KspUtils
from power_planner.graphs.fast_shortest_path import (
    sp_dag, sp_dag_reversed, topological_sort_jit, del_after_dest, edge_costs,
    average_lcp, sp_bf
)
import numpy as np
import warnings
import pandas as pd
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

    def set_shift(self, lower, upper, vec, max_angle, max_angle_lg=np.pi / 4):
        """
        Initialize shift variable by getting the donut values
        :param lower, upper: min and max distance of pylons
        :param vec: vector of diretion of edges
        :param max_angle: Maximum angle of edges to vec
        """
        self.angle_norm_factor = max_angle_lg
        self.shifts = get_half_donut(lower, upper, vec, angle_max=max_angle)
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
        self.n_nodes = self.x_len * self.y_len
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

    def set_edge_costs(self, layer_classes, layer_weights, angle_weight=0.5):
        """
        angle_weight: how to consider angles in contrast to all other costs!
        """
        # set weights and add angle weight
        self.cost_classes = ["angle"] + list(layer_classes)
        ang_weight_norm = angle_weight * np.sum(layer_weights)
        self.cost_weights = np.array([ang_weight_norm] + list(layer_weights))
        # print("class weights", layer_weights)
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
        dirty_extend = self.edge_inst.copy()
        x_len, y_len = self.edge_inst.shape
        for i in range(1, x_len - 1):
            for j in range(1, y_len - 1):
                if np.any(self.edge_inst[i - 1:i + 2, j - 1:j + 2] == np.inf):
                    dirty_extend[i, j] = np.inf
        self.edge_inst = dirty_extend
        if self.verbose:
            print("instance shape", self.instance.shape)

    # --------------------------------------------------------------------
    # SHORTEST PATH COMPUTATION

    def add_edges(self, mode="DAG", iters=100, edge_weight=0, height_weight=0):
        self.edge_weight = edge_weight
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
                iters, stack, np.array(self.shifts), self.angle_cost_array,
                self.dists, self.preds, self.instance, self.edge_cost
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
        self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
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
        self._display_dists()
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
        #     self.layer_weights, np.sum(np.array(path_costs), axis=0)
        # )  # scalar: weighted sum of the summed class costs
        return np.asarray(path
                          ).tolist(), path_costs.tolist(), cost_sum.tolist()

    def _display_dists(self):
        arr = np.zeros(self.pos2node.shape)
        for i in range(len(self.pos2node)):
            for j in range(len(self.pos2node[0])):
                ind = self.pos2node[i, j]
                if ind >= 0:
                    arr[i, j] = np.min(self.dists_ba[ind, :])
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
        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        return self.transform_path(path)

    # ---------------------------------------------------------------------
    # Compute raw (unnormalized) costs and output csv

    def raw_path_costs(self, path):
        """
        Compute raw angles, edge costs, pylon heights and normal costs
        (without weighting)
        Arguments:
            List or array of path coordinates
        """
        path_costs = np.array(
            [self.cost_instance[:, p[0], p[1]] for p in path]
        )
        # raw angle costs
        ang_costs = CostUtils.compute_raw_angles(path)
        # raw edge costs
        edge_costs = CostUtils.compute_edge_costs(path, self.edge_inst)
        # pylon heights
        try:
            heights = np.expand_dims(self.heights, 1)
        except AttributeError:
            heights = np.zeros((len(edge_costs), 1))
        # concatenate
        print(
            np.expand_dims(ang_costs, 1).shape,
            np.expand_dims(edge_costs, 1).shape, path_costs.shape,
            heights.shape
        )
        all_costs = np.concatenate(
            (
                np.expand_dims(ang_costs, 1), path_costs,
                np.expand_dims(edge_costs, 1), heights
            ), 1
        )
        names = self.cost_classes + ["edge_costs", "heigths"]
        assert all_costs.shape[1] == len(names)
        return all_costs, names

    def save_path_cost_csv(self, save_path, paths, **kwargs):
        """
        save coordinates in original instance (tifs) without padding etc
        """
        # out_path_list = []
        for i, path in enumerate(paths):
            # compute raw costs and column names
            raw_cost, names = self.raw_path_costs(path)
            # round raw costs of this particular path
            raw_costs = np.around(raw_cost, 2)
            # scale and shift
            scaled_path = np.asarray(path) * kwargs["scale"]
            shift_to_orig = kwargs["orig_start"] - scaled_path[0]
            power_path = scaled_path + shift_to_orig
            # out_path_list.append(shifted_path.tolist())

            coordinates = [kwargs["transform_matrix"] * p for p in power_path]

            all_coords = np.concatenate(
                (coordinates, power_path, raw_costs), axis=1
            )
            df = pd.DataFrame(
                all_coords, columns=["X", "Y", "X_raw", "Y_raw"] + names
            )
            df.to_csv(save_path + "_" + str(i) + ".csv", index=False)

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
        self.set_shift(
            kwargs["PYLON_DIST_MIN"],
            kwargs["PYLON_DIST_MAX"],
            self.dest_inds - self.start_inds,
            kwargs["MAX_ANGLE"],
            max_angle_lg=kwargs["MAX_ANGLE_LG"]
        )
        print("1) Initialize shifts and instance (corridor)")
        self.set_edge_costs(
            kwargs["layer_classes"],
            kwargs["class_weights"],
            angle_weight=kwargs["ANGLE_WEIGHT"]
        )
        self.instance = self.instance**power
        # add vertices
        self.add_nodes()
        print("2) Initialize distances to inf and predecessors")
        self.add_edges(edge_weight=kwargs["EDGE_WEIGHT"])
        print("3) Compute source shortest path tree")
        print("number of vertices and edges:", self.n_nodes, self.n_edges)

        # get actual best path
        path, path_costs, cost_sum = self.get_shortest_path(
            self.start_inds, self.dest_inds
        )
        print("4) shortest path", cost_sum)
        return path, path_costs, cost_sum

    def sp_trees(self, **kwargs):
        start_inds = kwargs["start_inds"]
        dest_inds = kwargs["dest_inds"]
        path, path_costs, cost_sum = self.single_sp(**kwargs)
        self.get_shortest_path_tree(start_inds, dest_inds)
        return path, path_costs, cost_sum
