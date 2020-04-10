from graph_tool.all import shortest_distance
from .weighted_graph import WeightedGraph
import numpy as np
import time
import matplotlib.pyplot as plt


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
        # counter = 0
        while curr_vertex != start_vertex:
            curr_vertex = pred_map[curr_vertex]
            path.append(curr_vertex)
            # if counter > 100:
            #     print(path)
            #     raise RuntimeWarning("while loop for sp not terminating")
            # counter += 1
        return path

    def transform_path(self, vertices_path):
        path = [(ind // self.y_len, ind % self.y_len) for ind in vertices_path]

        out_costs = [self.cost_instance[:, i, j].tolist() for (i, j) in path]
        cost_sum = np.dot(
            self.cost_weights, np.sum(np.array(out_costs), axis=0)
        )
        return path, out_costs, cost_sum

    def get_shortest_path_tree(self, source, target):
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
        # Attention: is_reversed must be True when it has NOT been reversed
        #   before
        self.graph.set_reversed(is_reversed=True)
        self.dist_map_ba, self.pred_map_ba = shortest_distance(
            self.graph,
            target,
            weights=self.weight,
            negative_weights=True,
            pred_map=True,
            directed=self.graph.is_directed()
        )
        # again turn around to recover graph
        self.graph.set_reversed(is_reversed=False)
        self.time_logs["shortest_path"] = round(time.time() - tic, 3)

        path_ab = self.get_sp_from_preds(self.pred_map_ab, target, source)
        assert self.dist_map_ba[source] < np.inf, "s not reachable from t"
        path_ab.reverse()
        self.best_path = path_ab
        # return self.transform_path(path_ab)

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

    def k_shortest_paths(self, source, dest, k, overlap=0.5):
        tic = time.time()
        # initialize list of paths
        sp_set = set(self.best_path)
        # iterate over vertices
        best_paths = [self.best_path]
        # get list of vertices = unique values in pos2node except -1
        vertices = np.unique(self.pos2node)[1:]
        v_dists = [self.dist_map_ab[v] + self.dist_map_ba[v] for v in vertices]
        # sort paths
        v_shortest = np.argsort(v_dists)
        # iterate over vertices starting from shortest paths
        # times_getpath = []
        for j, v_ind in enumerate(v_shortest):
            v = vertices[v_ind]
            # TODO: for runtime scan only every xth one (anyways diverse)
            if v not in sp_set:
                # do not scan unreachable vertices
                if int(self.pred_map_ab[v]
                       ) == int(v) or int(self.pred_map_ba[v]) == int(v):
                    continue
                # tic1 = time.time()
                path_ac = self.get_sp_from_preds(self.pred_map_ab, v, source)
                path_cb = self.get_sp_from_preds(self.pred_map_ba, v, dest)
                # times_getpath.append(time.time() - tic1)
                path_ac.reverse()
                # concatenate - leave 1 away because otherwise twice
                vertices_path = path_ac + path_cb[1:]
                already = np.array([u in sp_set for u in vertices_path])
                # if less than half already there
                if np.sum(already) < len(already) * overlap:
                    best_paths.append(vertices_path)
                    sp_set.update(vertices_path)
                    print("added path, already scanned", j)
            # stop if k paths are sampled
            if len(best_paths) >= k:
                break

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        return [self.transform_path(p) for p in best_paths]
