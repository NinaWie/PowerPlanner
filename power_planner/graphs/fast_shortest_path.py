from numba import jit
import time
import numpy as np
from numba.typed import List
from power_planner.utils.utils_constraints import ConstraintUtils


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
        v_x = stack[i][0]
        v_y = stack[i][1]
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


@jit(nopython=True)
def average_lcp(stack, shifts, angles_all, dists, preds, instance, edge_cost):
    """
    Fast C++ (numba) method to compute the cumulative distances from start
    """
    counter = np.ones(dists.shape)
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i][0]
        v_y = stack[i][1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if (
                0 <= neigh_x < dists.shape[1] and 0 <= neigh_y < dists.shape[2]
                and instance[neigh_x, neigh_y] < np.inf
            ):
                # add up pylon cost + angle cost + edge cost
                cost_per_angle = (
                    (dists[:, v_x, v_y] * counter[:, v_x, v_y]) +
                    angles_all[s] + instance[neigh_x, neigh_y] +
                    edge_cost[s, neigh_x, neigh_y]
                ) / (counter[:, v_x, v_y] + 1)  # div by counter + 1
                # update distances and predecessors
                dists[s, neigh_x, neigh_y] = np.min(cost_per_angle)
                preds[s, neigh_x, neigh_y] = np.argmin(cost_per_angle)
                # update counter
                counter[s, neigh_x, neigh_y
                        ] = counter[np.argmin(cost_per_angle), v_x, v_y] + 1
    return dists, preds


@jit(nopython=True)
def add_out_edges(
    stack, shifts, angles_all, dists, instance, edge_inst, shift_lines,
    edge_weight
):
    """
    Compute cumulative distances with each point of dists containing OUT edges
    """
    preds = np.zeros(dists.shape) - 1
    # preds = preds - 1
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i][0]
        v_y = stack[i][1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]

            if (0 <= neigh_x < dists.shape[1]) and (
                0 <= neigh_y < dists.shape[2]
            ) and (instance[neigh_x, neigh_y] < np.inf):
                # compute edge costs
                if edge_weight > 0:
                    bres_line = shift_lines[s] + np.array([neigh_x, neigh_y])
                    edge_cost_list = np.zeros(len(bres_line))
                    for k in range(len(bres_line)):
                        edge_cost_list[k] = edge_inst[bres_line[k, 0],
                                                      bres_line[k, 1]]
                    edge_cost = edge_weight * np.mean(edge_cost_list)
                else:
                    edge_cost = 0
                # iterate over incoming edges for angle
                cost_per_angle = np.zeros(len(shifts))
                for s2 in range(len(shifts)):
                    in_neigh_x = v_x - shifts[s2][0]
                    in_neigh_y = v_y - shifts[s2][1]
                    if (
                        0 <= in_neigh_x < dists.shape[1]
                        and 0 <= in_neigh_y < dists.shape[2]
                    ):
                        cost_per_angle[s2] = dists[
                            s2, in_neigh_x, in_neigh_y
                        ] + angles_all[s, s2] + instance[v_x, v_y] + edge_cost
                    else:
                        cost_per_angle[s2] = np.inf
                dists[s, v_x, v_y] = np.min(cost_per_angle)
                preds[s, v_x, v_y] = np.argmin(cost_per_angle)
    return dists, preds


def add_edges_BF(n_iters, shifts, instance, angle_cost_array, dists, preds):
    """
    TODO: implement numba version of bellman ford algorithm
    """
    tic = time.time()

    for _ in range(n_iters):
        # iterate over edges
        for i in range(len(shifts)):
            # shift dists by this shift
            # todo: avoid swaping dimenions each time
            cost_switched = np.moveaxis(dists, 0, -1)
            # shift by shift
            costs_shifted = ConstraintUtils.shift_surface(
                cost_switched, shifts[i], fill_val=np.inf
            )

            # add new costs for current edge
            angle_cost = angle_cost_array[i]
            together = np.moveaxis(
                costs_shifted + angle_cost, -1, 0
            ) + instance
            # 28 x 10 x 10 + 28 angles + 10 x 10

            # get argmin for each edge
            # --> remember where the value on this edge came from
            argmin_together = np.argmin(together, axis=0)
            # get minimum path cost for each edge
            # weighted_costs_shifted = np.min(together, axis=0)
            weighted_costs_shifted = np.take_along_axis(
                together, argmin_together[None, :, :], axis=0
            )[0, :, :]

            concat = np.array([dists[i], weighted_costs_shifted])
            # get spots that are actually updated
            changed_ones = np.argmin(concat, axis=0)
            # update predecessors
            preds[i, changed_ones > 0] = argmin_together[changed_ones > 0]

            # update accumulated path costs
            dists[i] = np.min(concat, axis=0)

    # self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
    # time_per_iter = (time.time() - tic) / self.n_iters
    # time_per_shift = (time.time() -
    #                   tic) / (self.n_iters * len(self.shifts))
    # self.time_logs["add_edge"] = round(time_per_iter, 3)
    # self.time_logs["edge_list"] = round(time_per_shift, 3)
    return dists, preds
