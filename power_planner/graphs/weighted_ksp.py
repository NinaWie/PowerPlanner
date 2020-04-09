from graph_tool.all import shortest_distance
from .weighted_graph import WeightedGraph
import numpy as np
import time


class WeightedKSP(WeightedGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(WeightedKSP, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )
        # load graph from file
        # self.graph = graph
        # self.weight = self.graph.ep.weight

    @staticmethod
    def get_sp_from_preds(pred_map, curr_vertex, start_vertex):
        path = [int(curr_vertex)]
        while curr_vertex != start_vertex:
            curr_vertex = pred_map[curr_vertex]
            path.append(curr_vertex)
        return path

    def transform_path(self, vertices_path):
        path = [(ind // self.y_len, ind % self.y_len) for ind in vertices_path]

        out_costs = [self.cost_instance[:, i, j].tolist() for (i, j) in path]
        cost_sum = np.dot(
            self.cost_weights, np.sum(np.array(out_costs), axis=0)
        )
        return path, out_costs, cost_sum

    def get_shortest_path(self, source, target):
        tic = time.time()

        self.dist_map_ab, self.pred_map_ab = shortest_distance(
            self.graph,
            source,
            weights=self.weight,
            negative_weights=True,
            directed=self.graph.is_directed(),
            pred_map=True
        )
        # turn around edge directions
        self.graph.set_reversed(is_reversed=True)
        self.dist_map_ba, self.pred_map_ba = shortest_distance(
            self.graph,
            target,
            weights=self.weight,
            negative_weights=True,
            pred_map=True,
            directed=self.graph.is_directed()
        )
        self.time_logs["shortest_path"] = round(time.time() - tic, 3)

        path_ab = self.get_sp_from_preds(self.pred_map_ab, target, source)
        # path_ba = self.get_sp_from_preds(self.pred_map_ba, source, target)
        path_ab.reverse()
        self.best_path = path_ab
        return self.transform_path(path_ab)

    def best_in_window(self, w_xmin, w_xmax, w_ymin, w_ymax, source, dest):
        tic = time.time()
        min_dist = np.inf
        for x in range(w_xmin, w_xmax + 1, 1):
            for y in range(w_ymin, w_ymax + 1, 1):
                v = self.pos2node[x, y]
                if v != -1:
                    dists_v = self.dist_map_ab[v] + self.dist_map_ab[v]
                    if dists_v < min_dist:
                        min_dist = dists_v
                        best_v = v

        path_ac = self.get_sp_from_preds(self.pred_map_ab, best_v, source)
        path_cb = self.get_sp_from_preds(self.pred_map_ba, best_v, dest)
        path_ac.reverse()
        # leave 1 away because otherwise twice
        vertices_path = path_ac + path_cb[1:]

        self.time_logs["best_in_window"] = round(time.time() - tic, 3)

        return self.transform_path(vertices_path)

    def k_shortest_paths(self, source, dest, k, margin=0.05):
        tic = time.time()
        # initialize list of paths
        sp_set = set(self.best_path)
        # iterate over vertices
        best_paths = [self.best_path]  # TODO: include k

        # new version
        v_dists = [
            self.dist_map_ab[v] + self.dist_map_ba[v]
            for v in range(np.max(self.pos2node))
        ]
        v_shortest = np.argsort(v_dists)  # shortest one is first now
        # iterate over vertices starting from shortest paths
        for v in v_shortest:
            if v not in sp_set:
                path_ac = self.get_sp_from_preds(self.pred_map_ab, v, source)
                path_cb = self.get_sp_from_preds(self.pred_map_ba, v, dest)
                path_ac.reverse()
                # concatenate - leave 1 away because otherwise twice
                vertices_path = path_ac + path_cb[1:]
                already = np.array([u in sp_set for u in vertices_path])
                # if less than half already there
                if np.sum(already) < len(already) / 2:
                    best_paths.append(vertices_path)
                    sp_set.update(vertices_path)
            # stop if k paths are sampled
            if len(best_paths) >= k:
                break

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        return [self.transform_path(p) for p in best_paths]
