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
def edge_costs(
    stack, pos2node, shifts, edge_cost, edge_inst, shift_lines, edge_weight
):
    """
    Pre-compute all edge costs
    """
    edge_inst_len_x, edge_inst_len_y = edge_inst.shape
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if (
                0 <= neigh_x < edge_inst_len_x
                and 0 <= neigh_y < edge_inst_len_y
                and edge_inst[neigh_x, neigh_y] < np.inf
            ):
                bres_line = shift_lines[s] + np.array([v_x, v_y])
                edge_cost_list = np.zeros(len(bres_line))
                for k in range(len(bres_line)):
                    edge_cost_list[k] = edge_inst[bres_line[k, 0],
                                                  bres_line[k, 1]]
                neigh_ind = pos2node[neigh_x, neigh_y]
                edge_cost[neigh_ind, s] = edge_weight * np.mean(edge_cost_list)
    return edge_cost


@jit(nopython=True)
def sp_dag(
    stack, pos2node, shifts, angles_all, dists, preds, instance, edge_cost
):
    """
    Angle-weighted dynamic program for Directed Acyclic Graphs (O(n))
    Implemented with numba for performance

    Arguments:
        n_iters: Int - At most the number of vertices in the graph, if known
            then the maximum length of the shortest path
        stack: List of tuples - order in which to consider the vertices
            MUST BE TOPOLOGICALLY SORTED for this algorithm to work
        shifts: np array of size (x,2) --> indicating the neighborhood for each
            vertex
        angles_all: np array, angle cost for each shift (precomputed)
        dists: np array of size m --> indicates distance of each edge from the
            source vertex
        preds: np array of size m --> indicates predecessor for each edge
        instance: 2D array, for each vertex the cost
        edge_cost: np array of size m --> edge cost for each edge
    """
    inst_x_len, inst_y_len = instance.shape
    # print(len(stack))
    for i in range(len(dists)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]
        for s in range(len(shifts)):
            neigh_x = int(v_x + shifts[s][0])
            neigh_y = int(v_y + shifts[s][1])
            if (
                0 <= neigh_x < inst_x_len and 0 <= neigh_y < inst_y_len
                and pos2node[neigh_x, neigh_y] >= 0
            ):
                neigh_stack_ind = pos2node[neigh_x, neigh_y]
                # add up pylon cost + angle cost + edge cost
                cost_per_angle = dists[i] + angles_all[s] + instance[
                    neigh_x, neigh_y] + edge_cost[neigh_stack_ind, s]
                # update distances and predecessors
                dists[neigh_stack_ind, s] = np.min(cost_per_angle)
                preds[neigh_stack_ind, s] = np.argmin(cost_per_angle)
    return dists, preds


@jit(nopython=True)
def average_lcp(stack, shifts, angles_all, dists, preds, instance, edge_cost):
    """
    Compute the least cost AVERAGE path (with running average)
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
def sp_dag_reversed(
    stack, pos2node, shifts, angles_all, dists, instance, edge_cost,
    shift_lines, edge_weight
):
    """
    Compute cumulative distances from TARGET vertex
    """
    inst_len_x, inst_len_y = instance.shape
    preds = np.zeros(dists.shape) - 1
    # initialize: outgoing edges of dest ind
    dest_x = stack[-1, 0]
    dest_y = stack[-1, 1]
    for s in range(len(shifts)):
        dists[-1, s] = instance[dest_x, dest_y] + edge_cost[-1, s]
    # update OUTGOING edge distances
    for i in range(1, len(stack)):
        v_x = stack[-i - 1, 0]
        v_y = stack[-i - 1, 1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]

            if (0 <= neigh_x <
                inst_len_x) and (0 <= neigh_y < inst_len_y
                                 ) and (pos2node[neigh_x, neigh_y] >= 0):
                edge_cost_this = edge_cost[-i - 1, s]
                # iterate over incoming edges for angle
                cost_per_angle = np.zeros(len(shifts))
                for s2 in range(len(shifts)):
                    in_neigh_x = v_x - shifts[s2][0]
                    in_neigh_y = v_y - shifts[s2][1]
                    if (
                        0 <= in_neigh_x < inst_len_x
                        and 0 <= in_neigh_y < inst_len_y
                        and pos2node[in_neigh_x, in_neigh_y] >= 0
                    ):
                        in_neigh_ind = pos2node[in_neigh_x, in_neigh_y]
                        cost_per_angle[
                            s2] = dists[in_neigh_ind, s2] + angles_all[
                                s, s2] + instance[v_x, v_y] + edge_cost_this
                    else:
                        cost_per_angle[s2] = np.inf
                dists[-i - 1, s] = np.min(cost_per_angle)
                preds[-i - 1, s] = np.argmin(cost_per_angle)
    return dists, preds


@jit(nopython=True)
def sp_bf(
    n_iters, stack, shifts, angles_all, dists, preds, instance, edge_cost
):
    """
    Angle-weighted Bellman Ford algorithm (General graph)
    Implemented with numba for performance - O(lm) where l is the
    maximum length of the shortest path
    
    Arguments:
        n_iters: Int - At most the number of vertices in the graph, if known
            then the maximum length of the shortest path
        stack: List of tuples - order in which to consider the vertices
            Note: For this algorithm it does not matter, because done for
            a sufficient number of iterations
        shifts: np array of size (x,2) --> indicating the neighborhood for each
            vertex
        angles_all: np array, angle cost for each shift (precomputed)
        dists: np array of size m --> indicates distance of each edge from the
            source vertex
        preds: np array of size m --> indicates predecessor for each edge
        instance: 2D array, for each vertex the cost
        edge_cost: np array of size m --> edge cost for each edge
    """
    for _ in range(n_iters):
        for i in range(len(stack)):
            v_x = stack[i][0]
            v_y = stack[i][1]
            for s in range(len(shifts)):
                neigh_x = v_x + shifts[s][0]
                neigh_y = v_y + shifts[s][1]
                if (
                    0 <= neigh_x < dists.shape[1]
                    and 0 <= neigh_y < dists.shape[2]
                    and instance[neigh_x, neigh_y] < np.inf
                ):
                    # add up pylon cost + angle cost + edge cost
                    cost_per_angle = dists[:, v_x, v_y] + angles_all[
                        s] + instance[neigh_x, neigh_y] + edge_cost[s, neigh_x,
                                                                    neigh_y]
                    # update distances and predecessors if better
                    if np.min(cost_per_angle) < dists[s, neigh_x, neigh_y]:
                        dists[s, neigh_x, neigh_y] = np.min(cost_per_angle)
                        preds[s, neigh_x, neigh_y] = np.argmin(cost_per_angle)
    return dists, preds


def old_BF_slow(n_iters, shifts, instance, angle_cost_array, dists, preds):
    """
    Implementation of angle BF without numba, uses numpy.roll to update
    all vertices at the same time
    """
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
