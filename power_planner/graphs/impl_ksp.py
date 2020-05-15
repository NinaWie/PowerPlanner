import numpy as np
import time
from numba import jit
from power_planner.graphs.implicit_lg import topological_sort_jit, ImplicitLG
from power_planner.utils.utils_ksp import (
    KspUtils, get_sp_start_shift, get_sp_dest_shift
)


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
            get_sp_dest_shift(
                self.dists_ba,
                self.preds_ba,
                self.dest_inds,
                self.start_inds,
                np.array(self.shifts) * (-1),
                d_ba_arg,
                dest_edge=True
            )
        )
        assert np.all(self.best_path == np.array(self.sp)), "paths differ"

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
            if sorted_dists[j] == sorted_dists[j - 1] or np.isnan(
                sorted_dists[j]
            ):
                # already checked
                continue
            e = e_shortest[j]
            # compute start and dest v
            x1, x2, x3 = KspUtils._flat_ind_to_inds(e, summed_dists.shape)
            # get shortest path through this node
            if self.dists_ba[x1, x2, x3] != 0:
                # = 0 for inc edges of dest_inds (init of dists_ba)
                vertices_path = self._combined_paths(
                    source, dest, x1, [x2, x3]
                )
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
