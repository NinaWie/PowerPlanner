from power_planner.utils.utils import (
    get_half_donut, angle, discrete_angle_costs, bresenham_line
)
from power_planner.utils.utils_costs import CostUtils
import numpy as np
import time
import pickle
from numba import jit
from numba.typed import List
# import matplotlib.pyplot as plt


@jit(nopython=True)
def topological_sort_jit(v_x, v_y, shifts, to_visit, stack):
    """
    Fast C++ (numba) recursive method for topological sorting
    Arguments:
        v_x, v_y: current vertex
        shifts: array of length n_neighborsx2 to iterate over neighbors
        to_visit: 2D array of size of instance to remember visited nodes
        stack: list of topologically sorted vertices
    Returns:
        stack
    """
    # Mark the current node as visited.
    to_visit[v_x, v_y] = 0
    # Recur for all the vertices adjacent to this vertex
    for s in range(len(shifts)):
        neigh_x = v_x + shifts[s, 0]
        neigh_y = v_y + shifts[s, 1]
        if to_visit[neigh_x, neigh_y] == 1:
            topological_sort_jit(neigh_x, neigh_y, shifts, to_visit, stack)
    # Push current vertex to stack which stores result
    l_tmp = List()
    l_tmp.append(v_x)
    l_tmp.append(v_y)
    stack.append(l_tmp)
    return stack


@jit(nopython=True)
def del_after_dest(stack, d_x, d_y):
    for i in range(len(stack)):
        if stack[i][0] == d_x and stack[i][1] == d_y:
            return stack[i:]


@jit(nopython=True)
def edge_costs(stack, shifts, edge_cost, edge_inst, shift_lines, edge_weight):
    """
    Pre-compute all edge costs
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[-i - 1][0]
        v_y = stack[-i - 1][1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if (
                0 <= neigh_x < edge_cost.shape[1]
                and 0 <= neigh_y < edge_cost.shape[2]
                and edge_inst[neigh_x, neigh_y] < np.inf
            ):
                bres_line = shift_lines[s] + np.array([v_x, v_y])
                edge_cost_list = np.zeros(len(bres_line))
                for k in range(len(bres_line)):
                    edge_cost_list[k] = edge_inst[bres_line[k, 0],
                                                  bres_line[k, 1]]
                edge_cost[s, neigh_x, neigh_y
                          ] = edge_weight * np.mean(edge_cost_list)
    return edge_cost


@jit(nopython=True)
def add_in_edges(stack, shifts, angles_all, dists, preds, instance, edge_cost):
    """
    Fast C++ (numba) method to compute the cumulative distances from start
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[-i - 1][0]
        v_y = stack[-i - 1][1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if (
                0 <= neigh_x < dists.shape[1] and 0 <= neigh_y < dists.shape[2]
                and instance[neigh_x, neigh_y] < np.inf
            ):
                # add up pylon cost + angle cost + edge cost
                cost_per_angle = dists[:, v_x, v_y] + angles_all[s] + instance[
                    neigh_x, neigh_y] + edge_cost[s, neigh_x, neigh_y]
                # update distances and predecessors
                dists[s, neigh_x, neigh_y] = np.min(cost_per_angle)
                preds[s, neigh_x, neigh_y] = np.argmin(cost_per_angle)
    return dists, preds


class ImplicitLG():

    def __init__(
        self,
        instance,
        instance_corr,
        graphtool=1,
        directed=True,
        verbose=1,
        n_iters=50,
        fill_val=np.inf
    ):
        self.cost_instance = instance
        self.hard_constraints = instance_corr
        self.x_len, self.y_len = instance_corr.shape
        self.fill_val = fill_val
        self.n_iters = n_iters
        self.time_logs = {}
        self.verbose = verbose
        self.directed = directed

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
        shift_lines = []
        for shift in self.shifts:
            line = bresenham_line(0, 0, shift[0], shift[1])
            shift_lines.append(np.array(line[1:-1]))
        shift_lines = List(shift_lines)
        self.shift_lines = shift_lines

    def add_nodes(self):
        tic = time.time()

        self.dists = np.zeros((len(self.shifts), self.x_len, self.y_len))
        self.dists += self.fill_val
        i, j = self.start_inds
        self.dists[:, i, j] = self.instance[i, j]
        self.preds = np.zeros(self.dists.shape)
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)
        self.n_nodes = self.x_len * self.y_len
        self.n_edges = len(self.shifts) * self.x_len * self.y_len
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
        inf_corr[inf_corr > 0] = self.fill_val

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
        self, layer_classes, layer_weights, angle_weight=0.5, cab_forb=False
    ):
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
        if cab_forb:
            # ueberspannen is forbidden as well
            self.edge_inst = self.instance.copy()
        else:
            # edge inst: without inf values
            self.edge_inst = np.sum(
                np.moveaxis(self.cost_instance, 0, -1) * self.cost_weights[1:],
                axis=2
            )
        # # other method: fill with max - problematic for evaluation
        # edge_inst = self.instance.copy()
        # edge_inst[edge_inst == np.inf] = np.max(edge_inst[edge_inst < np.inf])
        # self.edge_inst = edge_inst

        if self.verbose:
            print("instance shape", self.instance.shape)

    def add_edges(self, mode="DAG", edge_weight=0):
        self.edge_weight = edge_weight
        if mode == "BF":
            self.add_edges_BF()
        elif mode == "DAG":
            tic = time.time()
            # SORT
            tmp_list = self._helper_list()
            visit_points = (self.instance < np.inf).astype(int)
            stack = topological_sort_jit(
                self.start_inds[0], self.start_inds[1],
                np.asarray(self.shifts), visit_points, tmp_list
            )
            # stack = del_after_dest(stack, self.dest_inds[0], self.dest_inds[1])
            if self.verbose:
                print("time topo sort:", round(time.time() - tic, 3))
                print("stack length", len(stack))
            tic = time.time()
            # precompute edge costs
            self.edge_cost = np.zeros(self.preds.shape)
            if self.edge_weight > 0:
                self.edge_cost = edge_costs(
                    stack, np.array(self.shifts), self.edge_cost,
                    self.edge_inst, self.shift_lines, self.edge_weight
                )
            # RUN - add edges
            self.dists, self.preds = add_in_edges(
                stack, np.array(self.shifts), self.angle_cost_array,
                self.dists, self.preds, self.instance, self.edge_cost
            )
            self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
            if self.verbose:
                print("time edges:", round(time.time() - tic, 3))

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

    def transform_path(self, path):
        path_costs = np.array(
            [self.cost_instance[:, p[0], p[1]] for p in path]
        )
        # include angle costs
        ang_costs = ConstraintUtils.compute_angle_costs(
            path, self.angle_norm_factor
        )
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
        # concatenate
        all_costs = np.concatenate(
            (
                path_costs, np.expand_dims(ang_costs,
                                           1), np.expand_dims(edge_costs, 1)
            ), 1
        )
        return all_costs

    def get_shortest_path(self, start_inds, dest_inds, ret_only_path=False):
        if not np.any(self.dists[:, dest_inds[0], dest_inds[1]] < np.inf):
            raise RuntimeWarning("empty path")
        tic = time.time()
        curr_point = dest_inds
        path = [dest_inds]
        # first minimum: angles don't matter, just min of in-edges
        min_shift = np.argmin(self.dists[:, dest_inds[0], dest_inds[1]])
        # track back until start inds
        while np.any(curr_point - start_inds):
            new_point = curr_point - self.shifts[int(min_shift)]
            # get new shift from argmins
            min_shift = self.preds[int(min_shift), curr_point[0], curr_point[1]
                                   ]
            path.append(new_point)
            curr_point = new_point

        path = np.flip(np.asarray(path), axis=0)
        if ret_only_path:
            return path
        self.sp = path
        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        return self.transform_path(path)

    def save_graph(self, out_path):
        with open(out_path + ".dat", "wb") as outfile:
            pickle.dump((self.dists, self.preds), outfile)

    def add_edges_BF(self):
        tic = time.time()

        for _ in range(self.n_iters):
            # iterate over edges
            for i in range(len(self.shifts)):
                # shift dists by this shift
                # todo: avoid swaping dimenions each time
                cost_switched = np.moveaxis(self.dists, 0, -1)
                # shift by shift
                costs_shifted = CostUtils.shift_surface(
                    cost_switched, self.shifts[i], fill_val=self.fill_val
                )

                # add new costs for current edge
                angle_cost = self.angle_cost_array[i]
                together = np.moveaxis(
                    costs_shifted + angle_cost, -1, 0
                ) + self.instance
                # 28 x 10 x 10 + 28 angles + 10 x 10

                # get argmin for each edge
                # --> remember where the value on this edge came from
                argmin_together = np.argmin(together, axis=0)
                # get minimum path cost for each edge
                # weighted_costs_shifted = np.min(together, axis=0)
                weighted_costs_shifted = np.take_along_axis(
                    together, argmin_together[None, :, :], axis=0
                )[0, :, :]

                concat = np.array([self.dists[i], weighted_costs_shifted])
                # get spots that are actually updated
                changed_ones = np.argmin(concat, axis=0)
                # update predecessors
                self.preds[i, changed_ones > 0] = argmin_together[
                    changed_ones > 0]

                # update accumulated path costs
                self.dists[i] = np.min(concat, axis=0)

        self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
        time_per_iter = (time.time() - tic) / self.n_iters
        time_per_shift = (time.time() -
                          tic) / (self.n_iters * len(self.shifts))
        self.time_logs["add_edge"] = round(time_per_iter, 3)
        self.time_logs["edge_list"] = round(time_per_shift, 3)


# def add_edges_DAG(self, stack):
#     # TODO: build stack and graph at the same time?
#     shifts = np.asarray(self.shifts)
#     for i in range(len(stack)):
#         v_x, v_y = tuple(stack[-i - 1])
#         update_neighbors(
#             v_x, v_y, shifts, self.angle_cost_array, self.dists,
#             self.preds, self.instance
#         )

# @jit(nopython=True)
# def update_neighbors(
#     v_x, v_y, shifts, angles_all, dists, preds, instance
# ):
#     i = 0
#     for s in shifts:
#         neigh_x = v_x + s[0]
#         neigh_y = v_y + s[1]
#         if 0 <= neigh_x < dists.shape[1] and 0 <= neigh_y < dists.shape[2]:
#             cost_per_angle = dists[:, v_x, v_y] + angles_all[i] + instance[
#                 neigh_x, neigh_y]
#             dists[i, neigh_x, neigh_y] = np.min(cost_per_angle)
#             preds[i, neigh_x, neigh_y] = np.argmin(cost_per_angle)
#         i += 1
