import numpy as np
import time
from power_planner.utils.utils_ksp import KspUtils
from power_planner.utils.utils import get_distance_surface


class KSP:

    def __init__(self, graph):
        self.graph = graph
        try:
            print(self.graph.dists_ba.shape)
            print(self.graph.preds_ba.shape)
        except AttributeError:
            raise RuntimeError(
                "Cannot initialize KSP object with a graph without"\
                "shortest path trees in both directions!"
            )

    def compute_min_node_dists_bigmem(self):
        """
        Eppstein's algorithm: Sum up the two SP treest and iterate
        """
        # sum both dists_ab and dists_ba, inst and edges are counted twice!
        summed_dists = (
            self.graph.dists + self.graph.dists_ba - self.graph.instance -
            self.graph.edge_cost
        )
        # mins along outgoing edges
        min_node_dists = np.min(summed_dists, axis=0)
        min_shift_dists = np.argmin(summed_dists, axis=0)
        # argsort
        v_shortest = np.argsort(min_node_dists.flatten())
        return min_node_dists, v_shortest, min_shift_dists

    def compute_min_node_dists(self):
        """
        Eppstein's algorithm: Sum up the two SP treest and iterate
        """
        # sum both dists_ab and dists_ba, inst and edges are counted twice!
        aux_inst = np.zeros(self.graph.dists.shape)
        for i in range(len(self.graph.dists)):
            (x, y) = tuple(self.graph.stack_array[i])
            aux_inst[i, :] = self.graph.instance[x, y]

        double_counted = aux_inst + self.graph.edge_cost
        double_counted[double_counted == np.inf] = 0
        summed_dists = (
            self.graph.dists + self.graph.dists_ba - double_counted
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
        # argsort
        v_shortest = np.argsort(min_dists_2d.flatten())
        return min_dists_2d, v_shortest, min_shifts_2d

    def laplace(self, k, thresh=20, cost_add=0.01, metric="eucl_max"):
        """
        Fast KSP method as tradeoff between diversity and cost
        (add additional cost to the paths found so far)
        Arguments:
            self.graph.start_inds, self.graph.dest_inds, k: see other methods
            radius: distance from the previous paths in which penalty is added
            cost_add: cost_add of 0.05 means that 5% of the best path costs is
                the maximum costs that are added
        Returns:
            List of ksp with costs
        """
        if metric != "eucl_max":
            raise NotImplementedError(
                "penalizing only possible with Yau-Hausdorff distance so far"
            )
        tic = time.time()
        best_paths = [self.graph.best_path]
        (min_node_dists, _, min_shift_dists) = self.compute_min_node_dists()
        # print(min_node_dists)
        _, _, best_cost = self.graph.transform_path(self.graph.best_path)
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
                n_dilate=thresh
            )
            corridor = corridor / np.max(corridor)
            min_node_dists = min_node_dists + corridor * factor
            # get min vertex
            current_best = np.nanargmin(min_node_dists.flatten())
            (x2, x3) = current_best // arr_len, current_best % arr_len
            # print(x2, x3, arr_len, current_best)
            x1 = min_shift_dists[x2, x3]
            # compute and add
            vertices_path = self.graph._combined_paths(
                self.graph.start_inds, self.graph.dest_inds, x1, [x2, x3]
            )
            best_paths.append(vertices_path)

        self.graph.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.graph.verbose:
            print("Laplace KSP time:", self.graph.time_logs["ksp"])
        return [self.graph.transform_path(p) for p in best_paths]

    def set_max_cost(self, cost_thresh):
        """
        UNUSED Helper method to bound the maximum cost dependend on the best
        path costs
        Arguments:
            cost_thresh: threshold as ratio of the best path costs, e.g.
                1.01 means that at most 1% more costs than the best path
        """
        best_path_cells, _, best_cost = self.graph.transform_path(
            self.graph.best_path
        )
        correction = 0.5 * (
            self.graph.instance[tuple(best_path_cells[0])] +
            self.graph.instance[tuple(best_path_cells[-1])]
        )
        # print(best_cost, max_cost * best_c)
        assert np.isclose(best_cost, cost_thresh + correction)
        max_costs = best_cost * cost_thresh - correction
        return max_costs

    def dispersion_ksp(self, k, thresh, metric="jaccard"):
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
        _, _, best_cost = self.graph.transform_path(self.graph.best_path)
        max_cost = best_cost * thresh
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
            # if self.graph.dists_ba[x1, x2, x3] == 0:
            # print("inc edge to self.graph.dest_inds")
            # = 0 for inc edges of self.graph.dest_inds_inds (init of dists_ba)
            # continue
            vertices_path = self.graph._combined_paths(
                self.graph.start_inds, self.graph.dest_inds, x1, [x2, x3]
            )
            collected_path.append(vertices_path)

        dists = KspUtils.pairwise_dists(collected_path, mode=metric)

        # find the two which are most diverse (following 2-approx)
        max_dist_pair = np.argmax(dists)
        div_ksp = [max_dist_pair // len(dists), max_dist_pair % len(dists)]
        # greedily add the others
        for _ in range(k - 2):
            min_dists = []
            for i in range(len(dists)):
                min_dists.append(np.min([dists[i, div_ksp]]))
            div_ksp.append(np.argmax(min_dists))

        self.graph.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.graph.verbose:
            print("Dispersion KSP time:", self.graph.time_logs["ksp"])
        # transform path for output
        return [self.graph.transform_path(collected_path[p]) for p in div_ksp]

    def most_diverse_jaccard(self, k, thresh):
        """
        See dispersion_ksp --> based on Jaccard metric
        """
        return self.graph.dispersion_ksp(k, thresh, dist_mode="jaccard")

    def most_diverse_eucl_max(self, k, thresh):
        """
        See dispersion_ksp --> based on Yen-Hausdorff distance
        """
        return self.graph.dispersion_ksp(k, thresh, dist_mode="eucl_max")

    def find_ksp(self, k, thresh, metric="eucl_max"):
        """
        K shortest path with greedily adding the next shortest vertex
        with sufficient eucledian distance from the previous paths

        Arguments:
            self.graph.start_inds, self.graph.dest_inds: vertices
            --> list with two entries
            k: int: number of paths to output
            min_dist: eucledian distance in pixels which is the minimum max
                dist of the next path to add
        """
        tic = time.time()
        (min_node_dists, v_shortest,
         min_shift_dists) = self.compute_min_node_dists()
        best_paths = [self.graph.best_path]
        tup_path = [np.array(p) for p in self.graph.best_path]
        # sp_set = set(tuple_path)
        sorted_dists = min_node_dists.flatten()[v_shortest]
        _, arr_len = min_node_dists.shape

        expanded = 0
        for j in range(len(v_shortest)):
            if sorted_dists[j] == sorted_dists[j - 1]:
                # we always check a path only if it is the x-th appearance
                # print(counter)
                continue
            if sorted_dists[j] == np.inf:
                break
            # counter large enough --> expand
            (x2, x3) = v_shortest[j] // arr_len, v_shortest[j] % arr_len

            if "max" in metric:
                # compute eucledian distances
                eucl_dist = [
                    np.linalg.norm(np.array([x2, x3]) - tup)
                    for tup in tup_path
                ]
                # don't need to expand the path directly, only if selected
                if np.min(eucl_dist) > thresh:
                    x1 = min_shift_dists[x2, x3]
                    vertices_path = self.graph._combined_paths(
                        self.graph.start_inds, self.graph.dest_inds, x1,
                        [x2, x3]
                    )
                    for v in vertices_path:
                        v_in = [np.all(v == elem) for elem in tup_path]
                        if not np.any(v_in):
                            tup_path.append(v)
                    expanded += 1
            else:
                x1 = min_shift_dists[x2, x3]
                vertices_path = self.graph._combined_paths(
                    self.graph.start_inds, self.graph.dest_inds, x1, [x2, x3]
                )
                eucl_dist = [
                    KspUtils.path_distance(
                        prev_path, vertices_path, mode=metric
                    ) for prev_path in best_paths
                ]
                expanded += 1
            if np.min(eucl_dist) > thresh:
                # assert np.any([np.array([x2,x3])==v for v in vertices_path])
                best_paths.append(vertices_path)

                if len(best_paths) >= k:
                    print(j, "expanded", expanded)
                    break
        self.graph.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.graph.verbose:
            print("find ksp time:", self.graph.time_logs["ksp"])
        return [self.graph.transform_path(path) for path in best_paths]

    def min_set_intersection(self, k, thresh=0.5, metric=None):
        """
        Greedy Find KSP algorithm

        Arguments:
            self.graph.start_inds, self.graph.dest_inds: vertices -->
            list with two entries
            k: int: number of paths to output
            overlap: ratio of vertices that are allowed to be contained in the
                previously computed SPs
        """
        if metric is not None:
            raise NotImplementedError("no metrics implemented for max set alg")
        tic = time.time()

        best_paths = [self.graph.best_path]
        tuple_path = [tuple(p) for p in self.graph.best_path]
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
            if sorted_dists[j] == np.inf:
                break
            (x2, x3) = v_shortest[j] // arr_len, v_shortest[j] % arr_len
            x1 = min_shift_dists[x2, x3]

            # get shortest path through this node
            # if self.graph.dists_ba[x1, x2, x3] == 0:
            # = 0 for inc edges of self.graph.dest_inds_inds (init of dists_ba)
            # continue
            vertices_path = self.graph._combined_paths(
                self.graph.start_inds, self.graph.dest_inds, x1, [x2, x3]
            )
            # compute similarity with previous paths
            # TODO: similarities
            already = np.array([tuple(u) in sp_set for u in vertices_path])
            # if similarity < threshold, add
            if np.sum(already) < len(already) * thresh:
                best_paths.append(vertices_path)
                tup_path = [tuple(p) for p in vertices_path]
                sp_set.update(tup_path)
                # _, _, cost = self.graph.transform_path(vertices_path)
                # print("found new path with cost", cost)
                # print("sorted dist:", sorted_dists[j])
            if len(best_paths) >= k:
                print(j)
                break
        self.graph.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.graph.verbose:
            print("FIND KSP time:", self.graph.time_logs["ksp"])
        return [self.graph.transform_path(path) for path in best_paths]


# Iterate over edge costs!
# FOR EDGES instead of vertices (replace from summed_dists onwards)
# summed_dists = (self.graph.dists + self.graph.dists_ba - self.graph.instance)
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
#     # compute start and self.graph.dest_inds v
#     x1, x2, x3 = KspUtils._flat_ind_to_inds(e, summed_dists.shape)
# if self.graph.dists_ba[x1, x2, x3] != 0: ... insert the rest
