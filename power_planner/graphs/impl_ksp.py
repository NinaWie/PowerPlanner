import numpy as np
import time
from numba import jit
from power_planner.graphs.implicit_lg import topological_sort_jit, ImplicitLG
from power_planner.utils.utils_ksp import KspUtils
from power_planner.utils.utils import get_distance_surface


@jit(nopython=True)
def add_out_edges(
    stack, shifts, angles_all, dists, instance, edge_inst, shift_lines,
    edge_weight
):
    """
    Compute cumulative distances with each point of dists containing OUT edges
    """
    preds = np.zeros(dists.shape)
    # preds = preds - 1
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[-i - 1][0]
        v_y = stack[-i - 1][1]
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


class ImplicitKSP(ImplicitLG):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(ImplicitKSP, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )

    def get_shortest_path_tree(self, source, target):
        """
        Compute costs from dest to all edges
        """
        tic = time.time()

        # initialize dists array
        self.dists_ba = np.zeros((len(self.shifts), self.x_len, self.y_len))
        self.dists_ba += self.fill_val
        i, j = self.dest_inds
        # this time need to set all incoming edges of dest to zero
        d0, d1 = self.dest_inds
        for s, (i, j) in enumerate(self.shifts):
            self.dists_ba[s, d0 + i, d1 + j] = 0

        # get stack
        tmp_list = self._helper_list()
        visit_points = (self.instance < np.inf).astype(int)
        stack = topological_sort_jit(
            self.dest_inds[0], self.dest_inds[1],
            np.asarray(self.shifts) * (-1), visit_points, tmp_list
        )
        # compute distances: new method because out edges instead of in
        self.dists_ba, self.preds_ba = add_out_edges(
            stack,
            np.array(self.shifts) * (-1), self.angle_cost_array, self.dists_ba,
            self.instance, self.edge_inst, self.shift_lines, self.edge_weight
        )
        self.time_logs["shortest_path_tree"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time shortest_path_tree:", round(time.time() - tic, 3))

        # distance in ba: take IN edges to source, by computing in neighbors
        # take their first dim value (out edge to source) + source val
        (s0, s1) = self.start_inds
        d_ba_arg = np.argmin(
            [
                self.dists_ba[s, s0 + i, s1 + j]
                for s, (i, j) in enumerate(self.shifts)
            ]
        )
        (i, j) = self.shifts[d_ba_arg]
        d_ba = self.dists_ba[d_ba_arg, s0 + i, s1 + j] + self.instance[s0, s1]

        d_ab = np.min(self.dists[:, self.dest_inds[0], self.dest_inds[1]])
        assert np.isclose(
            d_ba, d_ab
        ), "start to dest != dest to start " + str(d_ab) + " " + str(d_ba)
        # compute best path
        self.best_path = np.array(
            KspUtils.get_sp_dest_shift(
                self.dists_ba,
                self.preds_ba,
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
            self.dists, self.preds, start, best_edge, np.array(self.shifts),
            best_shift
        )
        # compute path from middle point to dest - outgoing edge
        path_cb = KspUtils.get_sp_dest_shift(
            self.dists_ba, self.preds_ba, dest, best_edge,
            np.array(self.shifts) * (-1), best_shift
        )
        # concatenate
        together = np.concatenate(
            (np.flip(np.array(path_ac), axis=0), np.array(path_cb)[1:]),
            axis=0
        )
        return together

    def compute_min_node_dists(self):
        """
        Eppstein's algorithm: Sum up the two SP treest and iterate
        """
        # sum both dists_ab and dists_ba, subtract inst because counted twice
        summed_dists = (self.dists + self.dists_ba - self.instance)
        # mins along outgoing edges
        min_node_dists = np.min(summed_dists, axis=0)
        min_shift_dists = np.argmin(summed_dists, axis=0)
        # argsort
        v_shortest = np.argsort(min_node_dists.flatten())
        return min_node_dists, v_shortest, min_shift_dists

    def laplace(self, source, dest, k, radius=20, cost_add=0.01):
        """
        Fast KSP method as tradeoff between diversity and cost
        (add additional cost to the paths found so far)
        Arguments:
            source, dest, k: see other ksp methods
            radius: distance from the previous paths in which penalty is added
            cost_add: cost_add of 0.05 means that 5% of the best path costs is
                the maximum costs that are added
        Returns:
            List of ksp with costs
        """
        tic = time.time()
        best_paths = [self.best_path]
        (min_node_dists, _, min_shift_dists) = self.compute_min_node_dists()
        # print(min_node_dists)
        _, _, best_cost = self.transform_path(self.best_path)
        factor = best_cost * cost_add
        # print(factor)
        _, arr_len = min_node_dists.shape
        for _ in range(k - 1):
            # set the already used vertices to inf
            for b in best_paths[-1]:
                min_node_dists[tuple(b)] = np.inf
            # add in corridor / distribution
            corridor = get_distance_surface(
                min_node_dists.shape,
                best_paths,
                mode="dilation",
                n_dilate=radius
            )
            corridor = corridor / np.max(corridor)
            min_node_dists = min_node_dists + corridor * factor
            # get min vertex
            current_best = np.nanargmin(min_node_dists.flatten())
            (x2, x3) = current_best // arr_len, current_best % arr_len
            # print(x2, x3, arr_len, current_best)
            x1 = min_shift_dists[x2, x3]
            if self.dists_ba[x1, x2, x3] == 0:
                continue
            # compute and add
            vertices_path = self._combined_paths(source, dest, x1, [x2, x3])
            best_paths.append(vertices_path)

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.verbose:
            print("Laplace KSP time:", self.time_logs["ksp"])
        return [self.transform_path(p) for p in best_paths]

    def set_max_cost(self, cost_thresh):
        """
        UNUSED Helper method to bound the maximum cost dependend on the best
        path costs
        Arguments:
            cost_thresh: threshold as ratio of the best path costs, e.g.
                1.01 means that at most 1% more costs than the best path
        """
        best_path_cells, _, best_cost = self.transform_path(self.best_path)
        correction = 0.5 * (
            self.instance[tuple(best_path_cells[0])] +
            self.instance[tuple(best_path_cells[-1])]
        )
        # print(best_cost, max_cost * best_c)
        assert np.isclose(best_cost, cost_thresh + correction)
        max_costs = best_cost * cost_thresh - correction
        return max_costs

    def dispersion_ksp(
        self, source, dest, k, cost_thresh, dist_mode="jaccard"
    ):
        """
        P-dispersion based algorithm to compute most diverse paths
        Arguments:
            cost_thresh: threshold as ratio of the best path costs, e.g.
                1.01 means that at most 1% more costs than the best path
        """
        tic = time.time()
        (min_node_dists, v_shortest,
         min_shift_dists) = self.compute_min_node_dists()
        sorted_dists = min_node_dists.flatten()[v_shortest]
        _, _, best_cost = self.transform_path(self.best_path)
        max_cost = best_cost * cost_thresh
        _, arr_len = min_node_dists.shape
        collected_path = []
        for j in range(len(v_shortest)):
            if sorted_dists[j] == sorted_dists[j - 1]:
                continue

            if sorted_dists[j] > max_cost:
                break

            # counter large enough --> expand
            (x2, x3) = v_shortest[j] // arr_len, v_shortest[j] % arr_len
            x1 = min_shift_dists[x2, x3]
            if self.dists_ba[x1, x2, x3] == 0:
                # print("inc edge to dest")
                # = 0 for inc edges of dest_inds (init of dists_ba)
                continue
            vertices_path = self._combined_paths(source, dest, x1, [x2, x3])
            collected_path.append(vertices_path)

        dists = KspUtils.pairwise_dists(collected_path, mode=dist_mode)

        # find the two which are most diverse (following 2-approx)
        max_dist_pair = np.argmax(dists)
        div_ksp = [max_dist_pair // len(dists), max_dist_pair % len(dists)]
        # greedily add the others
        for _ in range(k - 2):
            min_dists = []
            for i in range(len(dists)):
                min_dists.append(np.min([dists[i, div_ksp]]))
            div_ksp.append(np.argmax(min_dists))

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.verbose:
            print("Dispersion KSP time:", self.time_logs["ksp"])
        # transform path for output
        return [self.transform_path(collected_path[p]) for p in div_ksp]

    def most_diverse_jaccard(self, source, dest, k, cost_thresh):
        """
        See dispersion_ksp --> based on Jaccard metric
        """
        return self.dispersion_ksp(
            source, dest, k, cost_thresh, dist_mode="jaccard"
        )

    def most_diverse_eucl_max(self, source, dest, k, cost_thresh):
        """
        See dispersion_ksp --> based on Yen-Hausdorff distance
        """
        return self.dispersion_ksp(
            source, dest, k, cost_thresh, dist_mode="eucl_max"
        )

    def max_vertex_ksp(self, source, dest, k, min_dist=8):
        """
        K shortest path with greedily adding the next shortest vertex
        with sufficient eucledian distance from the previous paths

        Arguments:
            source, dest: vertices --> list with two entries
            k: int: number of paths to output
            min_dist: eucledian distance in pixels which is the minimum max
                dist of the next path to add
        """
        tic = time.time()
        (min_node_dists, v_shortest,
         min_shift_dists) = self.compute_min_node_dists()
        best_paths = [self.best_path]
        tup_path = [np.array(p) for p in self.best_path]
        # sp_set = set(tuple_path)
        sorted_dists = min_node_dists.flatten()[v_shortest]
        _, arr_len = min_node_dists.shape

        expanded = 0
        for j in range(len(v_shortest)):
            if sorted_dists[j] == sorted_dists[j - 1]:
                # we always check a path only if it is the x-th appearance
                # print(counter)
                continue

            # counter large enough --> expand
            (x2, x3) = v_shortest[j] // arr_len, v_shortest[j] % arr_len

            # compute eucledian distances
            eucl_dist = [
                np.linalg.norm(np.array([x2, x3]) - tup) for tup in tup_path
            ]
            if np.min(eucl_dist) > min_dist:
                expanded += 1
                x1 = min_shift_dists[x2, x3]
                if self.dists_ba[x1, x2, x3] == 0:
                    # print("inc edge to dest")
                    # = 0 for inc edges of dest_inds (init of dists_ba)
                    continue
                vertices_path = self._combined_paths(
                    source, dest, x1, [x2, x3]
                )
                # assert np.any([np.array([x2,x3])==v for v in vertices_path])
                best_paths.append(vertices_path)
                for v in vertices_path:
                    v_in = [np.all(v == elem) for elem in tup_path]
                    if not np.any(v_in):
                        tup_path.append(v)

                if len(best_paths) >= k:
                    print(j, "expanded", expanded)
                    break
        self.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.verbose:
            print("max vertex time:", self.time_logs["ksp"])
        return [self.transform_path(path) for path in best_paths]

    def find_ksp(self, source, dest, k, overlap=0.5):
        """
        Greedy Find KSP algorithm

        Arguments:
            source, dest: vertices --> list with two entries
            k: int: number of paths to output
            overlap: ratio of vertices that are allowed to be contained in the
                previously computed SPs
        """
        tic = time.time()

        best_paths = [self.best_path]
        tuple_path = [tuple(p) for p in self.best_path]
        sp_set = set(tuple_path)
        # sum both dists_ab and dists_ba, subtract inst because counted twice
        (min_node_dists, v_shortest,
         min_shift_dists) = self.compute_min_node_dists()
        # argsort
        _, arr_len = min_node_dists.shape
        # sorted dists:
        sorted_dists = min_node_dists.flatten()[v_shortest]
        # iterate over edges from least to most costly
        for j in range(len(v_shortest)):
            if sorted_dists[j] == sorted_dists[j - 1]:
                # on the same path as the vertex before
                continue
            (x2, x3) = v_shortest[j] // arr_len, v_shortest[j] % arr_len
            x1 = min_shift_dists[x2, x3]

            # get shortest path through this node
            if self.dists_ba[x1, x2, x3] == 0:
                # = 0 for inc edges of dest_inds (init of dists_ba)
                continue
            vertices_path = self._combined_paths(source, dest, x1, [x2, x3])
            # compute similarity with previous paths
            # TODO: similarities
            already = np.array([tuple(u) in sp_set for u in vertices_path])
            # if similarity < threshold, add
            if np.sum(already) < len(already) * overlap:
                best_paths.append(vertices_path)
                tup_path = [tuple(p) for p in vertices_path]
                sp_set.update(tup_path)
                # _, _, cost = self.transform_path(vertices_path)
                # print("found new path with cost", cost)
                # print("sorted dist:", sorted_dists[j])
            if len(best_paths) >= k:
                print(j)
                break
        self.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.verbose:
            print("FIND KSP time:", self.time_logs["ksp"])
        return [self.transform_path(path) for path in best_paths]


# Iterate over edge costs!
# FOR EDGES instead of vertices (replace from summed_dists onwards)
# summed_dists = (self.dists + self.dists_ba - self.instance)
# # argsort
# e_shortest = np.argsort(summed_dists.flatten())
# # sorted dists:
# sorted_dists = summed_dists.flatten()[e_shortest]
# # iterate over edges from least to most costly
# for j in range(len(e_shortest)):
#     if sorted_dists[j] == sorted_dists[j - 1] or np.isnan(
#         sorted_dists[j]
#     ):
#         # already checked
#         continue
#     e = e_shortest[j]
#     # compute start and dest v
#     x1, x2, x3 = KspUtils._flat_ind_to_inds(e, summed_dists.shape)
# if self.dists_ba[x1, x2, x3] != 0: ... insert the rest
