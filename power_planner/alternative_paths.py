import numpy as np
from numba import jit
from power_planner.graphs.implicit_lg import (
    topological_sort_jit, del_after_dest
)


@jit(nopython=True)
def stack_after_window(stack, w_xmin, w_xmax, w_ymin, w_ymax):
    for i in range(len(stack)):
        if stack[-i - 1][0] >= w_xmin and (stack[-i - 1][0] <= w_xmax) and (
            stack[-i - 1][1] >= w_ymin
        ) and (stack[-i - 1][1] <= w_ymax):
            return stack[-i - 1:]


@jit(nopython=True)
def cut_edges(stack, shifts, dists, preds, marked):
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
            best_shift = preds[s, neigh_x, neigh_y]
            if best_shift >= 0 and marked[int(best_shift), v_x, v_y]:
                marked[s, neigh_x, neigh_y] = 1
            elif best_shift < 0:
                marked[s, neigh_x, neigh_y] = 2
            # if s==len(shifts)-1:
            # depends on topological sort whether that holds
    return marked


@jit(nopython=True)
def cut_edges_dest(stack, shifts, dists, preds, marked):
    """
    Fast C++ (numba) method to compute the cumulative distances from start
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i][0]
        v_y = stack[i][1]
        for s in range(len(shifts)):
            best_shift = int(preds[s, v_x, v_y])
            if best_shift >= 0:
                in_neigh_x = v_x - shifts[best_shift][0]
                in_neigh_y = v_y - shifts[best_shift][1]
                if marked[best_shift, in_neigh_x, in_neigh_y]:
                    marked[s, v_x, v_y] = 1
            else:
                marked[s, v_x, v_y] = 2
    return marked


class AlternativePaths:

    def __init__(self, graph):
        """
        Taking a finished graph with shortest path trees from source and dest
        Arguments
            graph: instance of ImplicitKSP
        """
        self.graph = graph

    def replacement_path(self, marked_out):
        # get vertices that have edges with the predecessor
        # but also other edges = cut edges
        crit_points_x, crit_points_y = np.where(
            np.all(
                np.asarray(
                    [
                        np.any(marked_out, axis=0),
                        (1 - np.absolute(np.all(marked_out, axis=0)))
                    ]
                ),
                axis=0
            )
        )
        # np.absolute(np.mean(marked_out, axis=0)-0.5)!=0.5)
        crit_points = []
        crit_dists = []
        for (x, y) in zip(crit_points_x, crit_points_y):
            assert not np.all(marked_out[:, x, y] == 1
                              ) and not np.all(marked_out[:, x, y] == 0)
            # compute distance for each edge
            for s in range(len(marked_out)):
                # cut edge: incoming edge that could replace the tree connectio
                if marked_out[s, x, y] == 0:
                    edge_dist = (
                        self.graph.dists[s, x, y] +
                        self.graph.dists_ba[s, x, y] -
                        self.graph.instance[x, y] -
                        self.graph.edge_cost[s, x, y]
                    )
                    if not np.isnan(edge_dist):
                        crit_points.append((x, y, s))
                        crit_dists.append(edge_dist)
        best_ind = np.argmin(crit_dists)
        b_x, b_y, b_s = crit_points[best_ind]
        # print(b_x, b_y, self.graph.shifts[b_s], b_s, np.min(crit_dists))
        vertices_path = self.graph._combined_paths(
            self.graph.start_inds, self.graph.dest_inds, b_s, [b_x, b_y]
        )
        return self.graph.transform_path(vertices_path)

    def replace_single_edge(self, u_x, u_y, v_x, v_y):
        """
        Compute the best replacement path for the edge (u,v)
        Arguments:
            u_x, u_y, v_x, v_y: Integers defining the coordinates of u and v
        Note: If e=(u,v) is not part of the LCP, then simply
        the LCP will be returned!
        """
        # get index of edge we want to change
        shift = np.asarray([v_x - u_x, v_y - u_y])
        shift_ind = np.argmin(
            [np.linalg.norm(s - shift) for s in self.graph.shifts]
        )
        marked = np.zeros(self.graph.dists.shape)
        # set this single edge as marked (predecessor to be replaced)
        marked[shift_ind, v_x, v_y] = 1

        # construct stack
        tmp_list = self.graph._helper_list()
        visit_points = (self.graph.instance < np.inf).astype(int)
        stack_source = topological_sort_jit(
            self.graph.dest_inds[0], self.graph.dest_inds[1],
            np.asarray(self.graph.shifts) * (-1), visit_points, tmp_list
        )
        stack_source = del_after_dest(
            stack_source, self.graph.start_inds[0], self.graph.start_inds[1]
        )

        # mark all edges in subtree of this edge
        marked_out = cut_edges(
            stack_source, self.graph.shifts, self.graph.dists,
            self.graph.preds, marked
        )

        # compute replacement path
        return self.replacement_path(marked_out)

    def replace_window(self, w_xmin, w_xmax, w_ymin, w_ymax):
        # get index of edge we want to change
        marked = np.zeros(self.graph.dists.shape)
        # set this single edge as marked (predecessor to be replaced)
        marked[:, w_xmin:w_xmax + 1, w_ymin:w_ymax + 1] = 1

        # get stack --> in both directions
        tmp_list = self.graph._helper_list()
        visit_points = (self.graph.instance < np.inf).astype(int)
        stack_source = topological_sort_jit(
            self.graph.dest_inds[0], self.graph.dest_inds[1],
            np.asarray(self.graph.shifts) * (-1), visit_points, tmp_list
        )
        stack_source = del_after_dest(
            stack_source, self.graph.start_inds[0], self.graph.start_inds[1]
        )
        marked_source = cut_edges(
            stack_source, np.asarray(self.graph.shifts), self.graph.dists,
            self.graph.preds, marked.copy()
        )
        visit_points = (self.graph.instance < np.inf).astype(int)
        # same for T_t
        stack_dest = topological_sort_jit(
            self.graph.start_inds[0], self.graph.start_inds[1],
            np.asarray(self.graph.shifts), visit_points, tmp_list
        )
        stack_dest = del_after_dest(
            stack_dest, self.graph.dest_inds[0], self.graph.dest_inds[1]
        )
        marked_dest = cut_edges_dest(
            stack_dest,
            np.asarray(self.graph.shifts) * (-1), self.graph.dists_ba,
            self.graph.preds_ba, marked.copy()
        )
        comb_marked = np.any(np.asarray([marked_source, marked_dest]), axis=0)
        # compute replacement path
        return self.replacement_path(comb_marked)

    def path_through_window(self, w_xmin, w_xmax, w_ymin, w_ymax):
        # distance sums
        summed_dists = (
            self.graph.dists + self.graph.dists_ba - self.graph.instance -
            self.graph.edge_cost
        )
        # mins along outgoing edges
        min_node_dists = np.min(summed_dists, axis=0)
        min_shift_dists = np.argmin(summed_dists, axis=0)
        # select window
        window = min_node_dists[w_xmin:w_xmax + 1, w_ymin:w_ymax + 1]
        _, arr_len = window.shape
        # get min vertex in this window
        current_best = np.nanargmin(window.flatten())
        # get coordinates
        (x2_wd, x3_wd) = current_best // arr_len, current_best % arr_len
        (x2, x3) = (x2_wd + w_xmin, x3_wd + w_ymin)
        # get shift for this optimum (which edge is used)
        x1 = min_shift_dists[x2, x3]
        # compute corresponding path
        vertices_path = self.graph._combined_paths(
            self.graph.start_inds, self.graph.dest_inds, x1, [x2, x3]
        )
        return self.graph.transform_path(vertices_path)
