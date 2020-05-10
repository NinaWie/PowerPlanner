import numpy as np
import time
from power_planner.graphs.implicit_lg import topological_sort_jit, ImplicitLG
from power_planner.utils.utils_ksp import (
    KspUtils, add_out_edges, get_sp_start_shift, get_sp_dest_shift
)


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
        self.dists_ba[:, i, j] = self.instance[i, j]

        # get stack
        tmp_list = self._helper_list()
        visit_points = (self.instance > 0).astype(int)
        stack = topological_sort_jit(
            self.dest_inds[0], self.dest_inds[1],
            np.asarray(self.shifts) * (-1), visit_points, tmp_list
        )
        # compute distances: new method because out edges instead of in
        self.dists_ba, self.preds_ba = add_out_edges(
            stack[:-1],
            np.array(self.shifts) * (-1), self.angle_cost_array, self.dists_ba,
            self.instance
        )
        self.time_logs["shortest_path_tree"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time shortest_path_tree:", round(time.time() - tic, 3))
        d_ba = np.min(self.dists_ba[:, self.start_inds[0], self.start_inds[1]])
        d_ab = np.min(self.dists[:, self.dest_inds[0], self.dest_inds[1]])
        assert np.isclose(
            d_ba, d_ab
        ), "start to dest != dest to start " + str(d_ab) + " " + str(d_ab)
        # compute best path
        d_ba_arg = np.argmin(
            self.dists_ba[:, self.start_inds[0], self.start_inds[1]]
        )
        self.best_path = np.array(
            get_sp_dest_shift(
                self.dists_ba, self.preds_ba, self.dest_inds, self.start_inds,
                np.array(self.shifts) * (-1), d_ba_arg
            )
        )

    def _combined_paths(self, start, dest, best_shift, best_edge):
        # compute path from start to middle point - incoming edge
        best_edge = np.array(best_edge)
        path_ac = get_sp_start_shift(
            self.dists, self.preds, start, best_edge, np.array(self.shifts),
            best_shift
        )
        # compute path from middle point to dest - outgoing edge
        path_cb = get_sp_dest_shift(
            self.dists_ba, self.preds_ba, dest, best_edge,
            np.array(self.shifts) * (-1), best_shift
        )
        # concatenate
        together = np.concatenate(
            (np.flip(np.array(path_ac), axis=0), np.array(path_cb)[1:]),
            axis=0
        )
        return together

    def k_shortest_paths(self, source, dest, k, overlap=0.5):
        tic = time.time()

        best_paths = [self.best_path]
        tuple_path = [tuple(p) for p in self.best_path]
        sp_set = set(tuple_path)
        # sum both dists_ab and dists_ba, subtract inst because counted twice
        summed_dists = (self.dists + self.dists_ba - self.instance)
        # argsort
        e_shortest = np.argsort(summed_dists.flatten())
        # sorted dists:
        sorted_dists = summed_dists.flatten()[e_shortest]
        # iterate over edges from least to most costly
        for j in range(len(e_shortest)):
            if sorted_dists[j] == sorted_dists[j - 1]:
                # already checked
                continue
            e = e_shortest[j]
            # compute start and dest v
            x1, x2, x3 = KspUtils._flat_ind_to_inds(e, summed_dists.shape)
            # get shortest path through this node
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
                break

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        return [self.transform_path(path) for path in best_paths]
