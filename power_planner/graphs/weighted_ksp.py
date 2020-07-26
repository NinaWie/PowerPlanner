try:
    from graph_tool.all import shortest_distance
    GRAPH_TOOL = 1
except ImportError:
    import networkx as nx
    GRAPH_TOOL = 0
from .weighted_graph import WeightedGraph
from power_planner.utils.utils_ksp import KspUtils
import numpy as np
import time


class WeightedKSP(WeightedGraph):

    def __init__(
        self, cost_instance, hard_constraints, directed=True, verbose=1
    ):
        super(WeightedKSP, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            verbose=verbose
        )
        # load graph from file
        # self.graph = graph
        # self.weight = self.graph.ep.weight

    def transform_path(self, vertices_path):
        path = [(ind // self.y_len, ind % self.y_len) for ind in vertices_path]

        out_costs = [self.cost_instance[:, i, j].tolist() for (i, j) in path]
        cost_sum = np.dot(
            self.cost_weights, np.sum(np.array(out_costs), axis=0)
        )
        return path, out_costs, cost_sum

    def graphtool_sp_tree(self, source, target):
        """
        Interface to graph tool for computing both shortest path trees
        """
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

    def networkx_sp_tree(self, source, target):
        # compute source-rooted SP tree
        (preds_ab, self.dist_map_ab) = nx.dijkstra_predecessor_and_distance(
            self.graph, source, weight="weight"
        )
        # transform because we want only one predecessor for each
        self.pred_map_ab = {
            key: int(next(iter(pred_list), key))
            for key, pred_list in preds_ab.items()
        }
        # reverse edge directions
        self.graph = self.graph.reverse(copy=False)
        # compute target-rooted SP tree
        (preds_ba, self.dist_map_ba) = nx.dijkstra_predecessor_and_distance(
            self.graph, target, weight="weight"
        )
        # fill the ones that are not in dictionary yet
        (self.pred_map_ba, self.pred_map_ba) = {}, {}
        vertices = np.unique(self.pos2node)[1:]
        for v in vertices:
            try:
                self.pred_map_ba[v] = int(next(iter(preds_ba[v]), v))
            except KeyError:
                self.pred_map_ba[v] = v
                self.dist_map_ba[v] = np.inf
            try:
                self.pred_map_ab[v] = int(next(iter(preds_ab[v]), v))
            except KeyError:
                self.pred_map_ab[v] = v
                self.dist_map_ab[v] = np.inf
        self.graph = self.graph.reverse(copy=False)

    def get_shortest_path_tree(self, source, target):
        tic = time.time()

        if self.graphtool:
            self.graphtool_sp_tree(source, target)
        else:
            self.networkx_sp_tree(source, target)

        self.time_logs["shortest_path_tree"] = round(time.time() - tic, 3)

        path_ab = KspUtils.get_sp_from_preds(self.pred_map_ab, target, source)
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

        vertices_path = self.compute_sp(best_v, source, dest)

        return self.transform_path(vertices_path)

    def compute_sp(self, v, source, dest):
        path_ac = KspUtils.get_sp_from_preds(self.pred_map_ab, v, source)
        path_cb = KspUtils.get_sp_from_preds(self.pred_map_ba, v, dest)
        # times_getpath.append(time.time() - tic1)
        path_ac.reverse()
        # concatenate - leave 1 away because otherwise twice
        vertices_path = path_ac + path_cb[1:]
        return vertices_path

    def find_ksp(self, source, dest, k, overlap=0.5, mode="myset"):
        tic = time.time()
        # initialize list of paths
        sp_set = set(self.best_path)
        best_paths = [self.best_path]
        best_path_sets = [set(self.best_path)]
        # get list of vertices = unique values in pos2node except -1
        vertices = np.unique(self.pos2node)[1:]
        v_dists = [self.dist_map_ab[v] + self.dist_map_ba[v] for v in vertices]
        # sort paths
        v_shortest = np.argsort(v_dists)
        # iterate over vertices starting from shortest paths
        # times_getpath = []
        for _, v_ind in enumerate(v_shortest):
            v = vertices[v_ind]
            # TODO: for runtime scan only every xth one (anyways diverse)
            if v not in sp_set:
                # do not scan unreachable vertices
                if int(self.pred_map_ab[v]
                       ) == int(v) or int(self.pred_map_ba[v]) == int(v):
                    continue
                # tic1 = time.time()
                vertices_path = self.compute_sp(v, source, dest)

                # similar = similarity(vertices_path, best_paths, sp_set)
                if mode != "myset":
                    sofar = np.array(
                        [
                            KspUtils.similarity(sp, set(vertices_path), mode)
                            for sp in best_path_sets
                        ]
                    )
                    if np.all(sofar < overlap):
                        best_paths.append(vertices_path)
                        best_path_sets.append(set(vertices_path))
                # mode myset --> my version: set of all paths together
                else:
                    already = np.array([u in sp_set for u in vertices_path])
                    if np.sum(already) < len(already) * overlap:
                        best_paths.append(vertices_path)
                        sp_set.update(vertices_path)
                    # print("added path, already scanned", j)
            # stop if k paths are sampled
            if len(best_paths) >= k:
                break

        self.time_logs["ksp"] = round(time.time() - tic, 3)
        return [self.transform_path(p) for p in best_paths]

    def collect_paths(
        self, sorted_dists, vertices, v_shortest, source_v, target_v,
        max_costs, count_thresh
    ):
        start = 0
        counter = 20
        collected_paths = []
        for c in range(len(v_shortest)):
            new = sorted_dists[c]
            # check whether exactly the same
            if np.isclose(new, start):
                counter += 1
                continue

            # stop to collect paths if costs too high
            if new > max_costs:
                break

            # counter: many new nodes this path has in contrast to the one
            # before --> skip if not very different
            if counter < count_thresh:
                counter = 1
                start = new
                continue

            # elidgible: compute path
            # print(start, counter)
            vertices_path = self.compute_sp(
                vertices[v_shortest[c]], source_v, target_v
            )
            if vertices_path[0] != source_v or vertices_path[-1] != target_v:
                print(vertices[v_shortest[c]])
                print(vertices_path)
                raise RuntimeError("source or target not contained")
            collected_paths.append(vertices_path)

            # renew the current mindist
            start = new
            counter = 1

        return collected_paths

    def simple_transform(self, path):
        return np.array(
            [(ind // self.y_len, ind % self.y_len) for ind in path]
        )

    def dispersion_ksp(
        self,
        source,
        dest,
        k,
        cost_thresh=1.01,
        dist_mode="jaccard",
        count_thresh=5
    ):
        """
        Implement reversed formulation: Given a threshold on the costs,
        compute the k most diverse paths (p-dispersion)
        Arguments:
            cost_thresh: threshold on the costs (1.01 means 1% more than
                            best path costs)
        """
        # compute sorted list of paths
        vertices = np.unique(self.pos2node)[1:]
        v_dists = [self.dist_map_ab[v] + self.dist_map_ba[v] for v in vertices]
        v_shortest = np.argsort(v_dists)
        sorted_dists = np.array(v_dists)[v_shortest]
        # set maximum on costs
        best_path_cells, _, best_cost = self.transform_path(self.best_path)
        correction = 0.5 * (
            self.instance[tuple(best_path_cells[0])] +
            self.instance[tuple(best_path_cells[-1])]
        )
        assert np.isclose(best_cost, sorted_dists[0] + correction)
        max_costs = best_cost * cost_thresh - correction

        # enumerate paths and collect
        collected_paths = self.collect_paths(
            sorted_dists, vertices, v_shortest, source, dest, max_costs,
            count_thresh
        )

        # transform into coordinates
        collected_coords = [self.simple_transform(p) for p in collected_paths]

        # compute all pairwise distances
        dists = KspUtils.pairwise_dists(collected_coords, mode=dist_mode)

        # find the two which are most diverse (following 2-approx)
        max_dist_pair = np.argmax(dists)
        div_ksp = [max_dist_pair // len(dists), max_dist_pair % len(dists)]
        # greedily add the others
        for _ in range(k - 2):
            min_dists = []
            for i in range(len(dists)):
                min_dists.append(np.min([dists[i, div_ksp]]))
            div_ksp.append(np.argmax(min_dists))

        return [self.transform_path(collected_paths[p]) for p in div_ksp]
