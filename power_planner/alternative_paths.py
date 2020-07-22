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
def cut_edges(stack, shifts, preds, pos2node, marked):
    """
    Fast C++ (numba) method to compute the cumulative distances from start
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]
        v_ind = pos2node[v_x, v_y]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            neigh_ind = int(pos2node[neigh_x, neigh_y])
            best_shift = preds[neigh_ind, s]
            if best_shift >= 0 and marked[v_ind, int(best_shift)] == 1:
                marked[neigh_ind, s] = 1
            elif best_shift < 0:
                marked[neigh_ind, s] = 2
    return marked


@jit(nopython=True)
def cut_edges_dest(stack, shifts, preds, pos2node, marked):
    """
    Fast C++ (numba) method to compute the cumulative distances from start
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]
        for s in range(len(shifts)):
            v_ind = pos2node[v_x, v_y]
            if v_ind >= 0:
                best_shift = int(preds[v_ind, s])
                if best_shift >= 0:
                    in_neigh_x = v_x - shifts[best_shift][0]
                    in_neigh_y = v_y - shifts[best_shift][1]
                    in_neigh_ind = pos2node[in_neigh_x, in_neigh_y]
                    if in_neigh_ind >= 0 and marked[in_neigh_ind, best_shift
                                                    ] == 1:
                        marked[v_ind, s] = 1
                elif best_shift < 0:
                    marked[v_ind, s] = 2
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
        crit_points_list = np.where(
            np.all(
                np.asarray(
                    [
                        np.any(marked_out, axis=1),
                        (1 - np.absolute(np.all(marked_out, axis=1)))
                    ]
                ),
                axis=0
            )
        )
        print(crit_points_list)
        # np.absolute(np.mean(marked_out, axis=0)-0.5)!=0.5)
        crit_points = []
        crit_dists = []
        for stack_ind in crit_points_list[0]:
            # stack_ind = self.graph.pos2node[x, y]
            assert not np.all(marked_out[stack_ind] == 1
                              ) and not np.all(marked_out[stack_ind] == 0)
            # compute distance for each edge
            for s in range(len(self.graph.shifts)):
                (x, y) = tuple(self.graph.stack_array[stack_ind])
                # cut edge: incoming edge that could replace the tree connectio
                if marked_out[stack_ind, s] == 0:
                    edge_dist = (
                        self.graph.dists[stack_ind, s] +
                        self.graph.dists_ba[stack_ind, s] -
                        self.graph.instance[x, y] -
                        self.graph.edge_cost[stack_ind, s]
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

    def display_dists(self, dist, func=np.min):
        arr = np.zeros(self.graph.pos2node.shape)
        for i in range(len(self.graph.pos2node)):
            for j in range(len(self.graph.pos2node[0])):
                ind = self.graph.pos2node[i, j]
                if ind >= 0:
                    arr[i, j] = np.mean(dist[ind, :])
        plt.figure(figsize=(20, 10))
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

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
        v_ind = self.graph.pos2node[v_x, v_y]
        marked[v_ind, shift_ind] = 1

        # mark all edges in subtree of this edge
        marked_out = cut_edges(
            self.graph.stack_array, self.graph.shifts, self.graph.preds,
            self.graph.pos2node, marked
        )

        # compute replacement path
        return self.replacement_path(marked_out)

    def replace_window(self, w_xmin, w_xmax, w_ymin, w_ymax):
        # get index of edge we want to change
        marked = np.zeros(self.graph.dists.shape)
        # set this single edge as marked (predecessor to be replaced)
        print(self.graph.pos2node[w_xmin:w_xmax + 1, w_ymin:w_ymax + 1])
        for x in range(w_xmin, w_xmax + 1):
            for y in range(w_ymin, w_ymax + 1):
                ind_marked = self.graph.pos2node[x, y]
                if ind_marked >= 0:
                    marked[ind_marked, :] = 1

        marked_source = cut_edges(
            self.graph.stack_array, np.asarray(self.graph.shifts),
            self.graph.preds, self.graph.pos2node, marked.copy()
        )
        # self.display_dists(marked)
        stack_dest = np.flip(self.graph.stack_array, axis=0)
        marked_dest = cut_edges_dest(
            stack_dest,
            np.asarray(self.graph.shifts) * (-1), self.graph.preds_ba,
            self.graph.pos2node, marked.copy()
        )
        # self.display_dists(marked_dest)
        comb_marked = np.any(np.asarray([marked_source, marked_dest]), axis=0)
        # compute replacement path
        return self.replacement_path(comb_marked)

    def compute_min_node_dists(self):
        """
        Eppstein's algorithm: Sum up the two SP treest and iterate
        """
        # sum both dists_ab and dists_ba, inst and edges are counted twice!
        aux_inst = np.zeros(self.graph.dists.shape)
        for i in range(len(self.graph.dists)):
            (x, y) = tuple(self.graph.stack_array[i])
            aux_inst[i, :] = self.graph.instance[x, y]
        summed_dists = (
            self.graph.dists + self.graph.dists_ba - aux_inst -
            self.graph.edge_cost
        )
        # mins along outgoing edges
        min_node_dists = np.min(summed_dists, axis=1)
        min_shift_dists = np.argmin(summed_dists, axis=1)
        # project back to 2D:
        min_dists_2d = np.zeros(self.graph.instance.shape) + np.inf
        min_shifts_2d = np.zeros(self.graph.instance.shape)
        for (x, y) in self.graph.stack_array:
            pos_ind = self.graph.pos2node[x, y]
            min_dists_2d[x, y] = min_node_dists[pos_ind]
            min_shifts_2d[x, y] = min_shift_dists[pos_ind]
        return min_dists_2d, min_shifts_2d

    def path_through_window(self, w_xmin, w_xmax, w_ymin, w_ymax):
        # distances and shifts
        min_node_dists, min_shift_dists = self.compute_min_node_dists()
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
