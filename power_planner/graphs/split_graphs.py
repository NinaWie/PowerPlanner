from graph_tool.all import shortest_distance
import numpy as np
from .weighted_ksp import WeightedKSP
from .implicit_lg_ksp import ImplicitLgKSP


class SplitWeighted():

    def __init__(self, two_graphs, pix_per_part, margin, padding):
        self.two_graphs = two_graphs
        self.pix_per_part = pix_per_part
        self.margin = margin
        self.padding = padding
        self.deleted_part = pix_per_part - margin - padding

    def construct_critical_zones(self):
        critical_zones = [None, None]
        critical_zones[0] = self.two_graphs[0].pos2node[self.pix_per_part:
                                                        -self.padding, :]
        critical_zones[1] = self.two_graphs[1].pos2node[self.
                                                        padding:self.padding +
                                                        self.margin, :]
        assert len(critical_zones[0]) == self.margin
        self.critical_zones = critical_zones

    def get_sp_trees(self, start_points):
        summed = np.zeros(self.critical_zones[0].shape)
        pred_maps = []
        for i in range(2):
            start_node_ind = self.two_graphs[i].pos2node[start_points[i][0],
                                                         start_points[i][1]]
            dist_map, pred_map = shortest_distance(
                self.two_graphs[i].graph,
                start_node_ind,
                weights=self.two_graphs[i].weight,
                negative_weights=True,
                pred_map=True
            )
            pred_maps.append(pred_map)
            for j in range(self.margin):
                for k in range(summed.shape[1]):
                    node = self.critical_zones[i][j, k]
                    if node >= 0:
                        summed[j, k] += dist_map[node]
                    else:
                        summed[j, k] += np.inf
        self.summed = summed
        self.pred_maps = pred_maps

    def get_concat_path(self, start_points):

        inds0, inds1 = np.where(self.summed == np.min(self.summed))
        assert len(inds0) == 1

        best_nodes = [None, None]
        best_nodes[0] = self.two_graphs[0].pos2node[inds0[0] + self.
                                                    pix_per_part, inds1[0]]
        best_nodes[1] = self.two_graphs[1].pos2node[inds0[0] + self.padding +
                                                    self.margin, inds1[0]]

        concat_path = []
        for i in range(2):
            start_node_ind = self.two_graphs[i].pos2node[start_points[i][0],
                                                         start_points[i][1]]
            vertices_path = WeightedKSP.get_sp_from_preds(
                self.pred_maps[i], best_nodes[i], start_node_ind
            )
            path = np.array(
                [
                    (
                        ind // self.two_graphs[i].y_len,
                        ind % self.two_graphs[i].y_len
                    ) for ind in vertices_path
                ]
            )
            ind = best_nodes[i]
            print(
                ind // self.two_graphs[i].y_len, ind % self.two_graphs[i].y_len
            )
            # -padding is not required because we only pad at the opposite side
            if i == 0:
                path = np.flip(path, axis=0)
            else:
                path[:, 0] = path[:, 0] + self.deleted_part
                path = path[1:]
            # print(path)
            concat_path.extend(path.tolist())
        return concat_path


class SplitLG():

    def __init__(self, two_graphs, pix_per_part, margin, padding):
        self.two_graphs = two_graphs
        self.pix_per_part = pix_per_part
        self.margin = margin
        self.padding = padding
        self.deleted_part = pix_per_part - margin - padding

    def construct_critical_zones(self):

        critical_zone0 = self.two_graphs[0].dists[:, self.pix_per_part:-self.
                                                  padding, :]
        critical_zone1 = self.two_graphs[1].dists[:, self.
                                                  padding:self.padding +
                                                  self.margin, :]
        inst_zone = self.two_graphs[0].instance[self.
                                                pix_per_part:-self.padding, :]

        # iterate over all edges
        summed_dists = (critical_zone0 + critical_zone1 - inst_zone)
        summed_dists[np.isnan(summed_dists)] = np.inf

        # get actual inds in smaller window
        best_path_ind = np.argmin(summed_dists.flatten())
        self.best_path_ind = best_path_ind
        self.summed_shape = summed_dists.shape

    def get_sp_trees(self, start_points):
        pass

    def get_concat_path(self, start_points):
        # Compute merge node
        best_shift, x, y = ImplicitLgKSP._flat_ind_to_inds(
            self.best_path_ind, self.summed_shape
        )
        print(best_shift, x, y)

        # reconstruct paths from merge node
        path_ac = ImplicitLgKSP.get_sp_start_shift(
            self.two_graphs[0].dists, self.two_graphs[0].dists_argmin,
            start_points[0], [x + self.pix_per_part, y],
            self.two_graphs[0].shifts, best_shift
        )
        path_cb = ImplicitLgKSP.get_sp_start_shift(
            self.two_graphs[1].dists, self.two_graphs[1].dists_argmin,
            start_points[1], [x + self.padding, y], self.two_graphs[1].shifts,
            best_shift
        )

        path_cb = np.array(path_cb)[1:]
        path_cb[:, 0] = path_cb[:, 0] + self.deleted_part

        together = np.concatenate(
            (np.flip(np.array(path_ac), axis=0), path_cb), axis=0
        )
        return together
