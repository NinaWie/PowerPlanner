from power_planner.utils import angle
from power_planner import graphs
import numpy as np
import time
import matplotlib.pyplot as plt


class TwoPowerBF():

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
        self.cost_rest = instance * instance_corr
        self.graph_ab = graphs.PowerBF(
            instance, instance_corr, graphtool=1, verbose=1
        )
        self.graph_ba = graphs.PowerBF(
            instance, instance_corr, graphtool=1, verbose=1
        )
        self.time_logs = self.graph_ab.time_logs

    def set_edge_costs(self, layer_classes, class_weights, angle_weight=0.5):
        self.graph_ab.set_edge_costs(
            layer_classes, class_weights, angle_weight=angle_weight
        )
        self.graph_ba.set_edge_costs(
            layer_classes, class_weights, angle_weight=angle_weight
        )
        self.instance = self.graph_ab.instance

    def set_shift(self, min_dist, max_dist, vec, max_angle, max_angle_lg):
        self.graph_ab.set_shift(
            min_dist, max_dist, vec, max_angle, max_angle_lg
        )
        self.graph_ba.angle_norm_factor = max_angle_lg
        self.graph_ba.shifts = np.asarray(self.graph_ab.shifts) * (-1)
        # self.graph_ba.set_shift(
        #     min_dist, max_dist,
        #     np.asarray(vec) * (-1), max_angle, max_angle_lg
        # )

    def add_nodes(self):
        self.graph_ab.add_nodes()
        self.graph_ba.add_nodes()
        self.n_nodes = self.graph_ab.n_nodes
        self.n_edges = self.graph_ab.n_edges

    def set_corridor(self, factor, corridor, start_inds, dest_inds):
        self.graph_ab.set_corridor(factor, corridor, start_inds, dest_inds)
        self.graph_ba.set_corridor(factor, corridor, dest_inds, start_inds)
        self.factor = factor

    def add_edges(self):
        self.graph_ab.add_edges()
        self.graph_ba.add_edges()

    def add_start_and_dest(self, source, dest):
        # here simply return the indices for start and destination
        return source, dest

    def sum_costs(self):
        pass

    def get_shortest_path(self, start_inds, dest_inds):
        self.path_ab, path_costs, cost_sum = self.graph_ab.get_shortest_path(
            start_inds, dest_inds
        )
        self.path_ba, _, _ = self.graph_ba.get_shortest_path(
            dest_inds, start_inds
        )
        assert np.all(
            np.flip(np.asarray(self.path_ba), axis=0) == self.path_ab
        )
        return self.path_ab, path_costs, cost_sum

    def best_in_window_simple(
        self,
        w_xmin,
        w_xmax,
        w_ymin,
        w_ymax,
        start_inds,
        dest_inds,
        margin=0.05
    ):
        """
        margin: percent that it's allowed to be higher than average
        """
        opt = np.min(self.graph_ab.dists[:, dest_inds[0], dest_inds[1]])

        possible_cs = []
        for x in range(w_xmin, w_xmax + 1, 1):
            for y in range(w_ymin, w_ymax + 1, 1):
                # todo here: take into account angle directly
                added_costs = np.min(self.graph_ab.dists[:, x, y]) + np.min(
                    self.graph_ba.dists[:, x, y]
                ) - self.graph_ab.instance[x, y]
                if added_costs < opt + margin * opt:
                    possible_cs.append(np.array([x, y]))
        for c in possible_cs:
            path_ac = self.graph_ab.get_shortest_path(
                start_inds, c, ret_only_path=True
            )
            path_cb = self.graph_ba.get_shortest_path(
                dest_inds, c, ret_only_path=True
            )
            plt.plot(path_ac[:, 0], path_ac[:, 1])
            plt.plot(path_cb[:, 0], path_cb[:, 1])
            plt.show()

    @staticmethod
    def get_sp_start_shift(
        dists, dists_argmin, start_inds, dest_inds, shifts, min_shift
    ):
        if not np.any(dists[:, dest_inds[0], dest_inds[1]] < np.inf):
            raise RuntimeWarning("empty path")
        curr_point = np.asarray(dest_inds)
        my_path = [dest_inds]
        # min_shift = np.argmin(dists[:, dest_inds[0], dest_inds[1]])
        while np.any(curr_point - start_inds):
            new_point = curr_point - shifts[int(min_shift)]
            min_shift = dists_argmin[int(min_shift), curr_point[0],
                                     curr_point[1]]
            my_path.append(new_point)
            curr_point = new_point
        return np.asarray(my_path)

    def best_in_window(
        self,
        w_xmin,
        w_xmax,
        w_ymin,
        w_ymax,
        start_inds,
        dest_inds,
        margin=0.05
    ):
        """
        margin: percent that it's allowed to be higher than average
        """
        tic = time.time()

        ang_weight = self.graph_ba.angle_weight
        ang_norm_factor = self.graph_ba.angle_norm_factor

        possible_cs = []
        c_path_cost = []
        possible_shifts = []

        for x in range(w_xmin, w_xmax + 1, 1):
            for y in range(w_ymin, w_ymax + 1, 1):
                # todo here: take into account angle directly
                cell_val = self.graph_ab.instance[x, y]
                if cell_val < np.inf:
                    min_costs = np.inf
                    min_shifts = [0, 0]
                    for s1 in range(len(self.graph_ab.shifts)):
                        for s2 in range(len(self.graph_ab.shifts)):
                            val_ab = self.graph_ab.dists[s1, x, y]
                            shift_ab = self.graph_ab.shifts[s1]
                            val_ba = self.graph_ba.dists[s2, x, y]
                            shift_ba = self.graph_ba.shifts[s2]
                            ang = angle(
                                np.asarray(shift_ab),
                                np.asarray(shift_ba) * (-1)
                            )
                            added_costs = (
                                val_ab + val_ba - cell_val +
                                ang_weight * ang / ang_norm_factor
                            )
                            if added_costs < min_costs:
                                min_costs = added_costs
                                min_shifts = [s1, s2]
                    possible_shifts.append(min_shifts)
                    added_costs = min_costs
                    # np.min(self.graph_ab.dists[:, x, y]) +
                    # np.min(self.graph_ba.dists[:, x, y]) -
                    # self.graph_ab.instance[x,y]
                else:
                    possible_shifts.append([0, 0])
                    added_costs = np.inf
                possible_cs.append(np.array([x, y]))
                c_path_cost.append(min_costs)

        # get best one
        best_c = np.argmin(c_path_cost)
        print(best_c)
        c = possible_cs[best_c]
        s1, s2 = possible_shifts[best_c]
        # stick together the path
        path_ac = self.get_sp_start_shift(
            self.graph_ab.dists, self.graph_ab.dists_argmin, start_inds, c,
            self.graph_ab.shifts, s1
        )
        path_cb = self.get_sp_start_shift(
            self.graph_ba.dists, self.graph_ba.dists_argmin, dest_inds, c,
            self.graph_ba.shifts, s2
        )
        # plt.plot(path_ac[:, 0], path_ac[:, 1])
        # plt.plot(path_cb[:, 0], path_cb[:, 1])
        # plt.show()
        together = np.concatenate(
            (np.flip(np.array(path_ac), axis=0), np.array(path_cb)), axis=0
        )

        path_costs = []
        for p in together:
            path_costs.append(self.graph_ab.instance_layers[:, p[0], p[1]])

        self.graph_ab.time_logs["best_in_window"] = round(time.time() - tic, 3)
        return together, path_costs, c_path_cost[best_c]

    def k_shortest_paths(self, source, dest, k, overlap=0.5):

        def flat_ind_to_inds(flat_ind, arr_shape):
            _, len2, len3 = arr_shape
            x1 = flat_ind // (len2 * len3)
            x2 = (flat_ind % (len2 * len3)) // len3
            x3 = (flat_ind % (len2 * len3)) % len3
            return (x1, x2, x3)

        tic = time.time()

        best_paths = [self.path_ab]
        tuple_path = [tuple(p) for p in self.path_ab]
        sp_set = set(tuple_path)
        # sum both dists_ab and dists_ba, subtract inst because counted twice
        summed_dists = (
            self.graph_ab.dists + self.graph_ba.dists - self.graph_ab.instance
        )
        # argsort
        e_shortest = np.argsort(summed_dists.flatten())
        # iterate over edges from least to most costly
        for e in e_shortest:
            # compute start and dest v
            x1, x2, x3 = flat_ind_to_inds(e, summed_dists.shape)
            path_ac = self.get_sp_start_shift(
                self.graph_ab.dists, self.graph_ab.dists_argmin, source,
                [x2, x3], self.graph_ab.shifts, x1
            )
            path_cb = self.get_sp_start_shift(
                self.graph_ba.dists, self.graph_ba.dists_argmin, dest,
                [x2, x3], self.graph_ba.shifts, x1
            )
            vertices_path = np.concatenate(
                (np.flip(np.array(path_ac), axis=0), np.array(path_cb)[1:]),
                axis=0
            )
            already = np.array([tuple(u) in sp_set for u in vertices_path])
            if np.sum(already) < len(already) * overlap:
                best_paths.append(vertices_path)
                tup_path = [tuple(p) for p in vertices_path]
                sp_set.update(tup_path)
            if len(best_paths) >= k:
                break

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        return [self.graph_ab.transform_path(path) for path in best_paths]
