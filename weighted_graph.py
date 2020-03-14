from constraints import convolve, get_kernel, convolve_faster
from power_planner.utils import shift_surface, get_donut_vals
from general_graph import GeneralGraph

import numpy as np
from graph_tool.all import Graph, shortest_path
import time
import networkx as nx
from collections import deque


class WeightedGraph(GeneralGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        assert cost_instance.shape == hard_constraints.shape
        # , "Cost size must be equal to corridor definition size!"

        # time logs
        self.time_logs = {}
        tic = time.time()

        # indicator whether to use networkx or graph tool
        self.graphtool = graphtool

        # cost surface
        self.cost_instance = cost_instance
        self.hard_constraints = hard_constraints
        self.x_len, self.y_len = cost_instance.shape
        # node to pos mapping
        self.node_pos = [
            (i, j) for i in range(self.x_len) for j in range(self.y_len)
            if hard_constraints[i, j]
        ]
        # pos to node mapping
        self.pos_node_dict = {
            self.node_pos[i]: i
            for i in range(len(self.node_pos))
        }
        self.pos2node = np.ones(cost_instance.shape)
        self.pos2node *= -1
        for n, (i, j) in enumerate(self.node_pos):
            self.pos2node[i, j] = n
        print("initialized weighted graph (pos2node and node_pos)")

        # initialize graph:
        GeneralGraph.__init__(
            self, directed=directed, graphtool=graphtool, verbose=verbose
        )

        # print statements
        self.verbose = verbose

        self.time_logs["init_graph"] = round(time.time() - tic, 3)

    def set_shift(self, lower, upper, vec, max_angle):
        GeneralGraph.set_shift(self, lower, upper, vec, max_angle)
        self.shift_vals = get_donut_vals(self.shifts, vec)

    def add_nodes(self):
        # add nodes to graph
        GeneralGraph.add_nodes(self, len(self.node_pos))

    def add_edges_old(self, shifts):
        # Define edges
        tic = time.time()
        edge_list = []
        for n, (i, j) in enumerate(self.node_pos):
            weight_node = self.cost_instance[i, j]
            for (x, y) in shifts:
                new_x = i + x
                new_y = j + y
                if new_x >= 0 and new_x < self.x_len and new_y >= 0 and new_y < self.y_len:
                    if self.hard_constraints[new_x, new_y]:  # inside corridor
                        weight = self.cost_instance[new_x, new_y] + weight_node
                        edge_list.append(
                            [
                                n, self.pos_node_dict[(new_x, new_y)],
                                round(weight, 3)
                            ]
                        )
        if self.verbose:
            print("time to build edge list:", time.time() - tic)
        self.time_logs["edge_list"] = round(
            (time.time() - tic) / len(shifts), 3
        )

        tic = time.time()
        # add edges and properties to the graph
        if self.graphtool:
            self.graph.add_edge_list(edge_list, eprops=[self.weight])
        else:
            nx_edge_list = [(e[0], e[1], {"weight": e[2]}) for e in edge_list]
            self.graph.add_edges_from(nx_edge_list)
        # verbose
        if self.verbose:
            print("added edges:", len(edge_list))

            print("finished adding edges", time.time() - tic)
        self.time_logs["add_edges"] = round(
            (time.time() - tic) / len(shifts), 3
        )

    def add_edges(self):
        tic_function = time.time()
        inds_orig = self.pos2node[self.hard_constraints > 0]

        self.cost_rest = self.cost_instance * (self.hard_constraints >
                                               0).astype(int)
        n_edges = 0

        kernels, posneg = get_kernel(self.shifts, self.shift_vals)

        times_edge_list = []
        times_add_edges = []

        edge_array = []

        for i in range(len(self.shifts)):

            tic_edges = time.time()
            costs_shifted = shift_surface(self.cost_rest, self.shifts[i])

            weights = (costs_shifted + self.cost_rest) / 2
            # new version: edge weights
            # weights = convolve_faster(self.cost_rest, kernels[i], posneg[i])
            # weights = weights1 + 2 * weights2
            # print(
            #     "max node weights", np.max(weights1), "max edge weights:",
            #     np.max(weights2), "min node weights", np.min(weights1),
            #     "min edge weights:", np.min(weights2)
            # )

            inds_shifted = self.pos2node[costs_shifted > 0]

            # delete the ones where inds_shifted is zero
            assert len(inds_shifted) == len(inds_orig)
            weights_list = weights[costs_shifted > 0]

            pos_inds = inds_shifted >= 0
            out = np.swapaxes(
                np.asarray([inds_orig, inds_shifted, weights_list]), 1, 0
            )[pos_inds]

            n_edges += len(out)
            times_edge_list.append(round(time.time() - tic_edges, 3))

            # add edges to graph
            tic_graph = time.time()
            if self.graphtool:
                self.graph.add_edge_list(out, eprops=[self.weight])
            else:
                nx_edge_list = [(e[0], e[1], {"weight": e[2]}) for e in out]
                self.graph.add_edges_from(nx_edge_list)
            times_add_edges.append(round(time.time() - tic_graph, 3))

            # alternative: collect edges here and add alltogether
            # edge_array.append(out)

        # # alternative: add edges all in one go
        # tic_concat = time.time()
        # edge_lists_concat = np.concatenate(edge_array, axis=0)
        # self.time_logs["concatenate"] = round(time.time() - tic_concat, 3)
        # print("time for concatenate:", self.time_logs["concatenate"])
        # tic_graph = time.time()
        # self.graph.add_edge_list(edge_lists_concat, eprops=[self.weight])
        # self.time_logs["add_edges"] = round(
        #     (time.time() - tic_graph) / len(shifts), 3
        # )
        self.time_logs["add_edges"] = round(np.mean(times_add_edges), 3)
        self.time_logs["add_edges_times"] = times_add_edges

        self.time_logs["edge_list"] = round(np.mean(times_edge_list), 3)
        self.time_logs["edge_list_times"] = times_edge_list

        if self.verbose:
            print("DONE adding", n_edges, "edges:", time.time() - tic_function)

    def add_start_and_dest(self, start_inds, dest_inds):
        start_node_ind = self.pos2node[start_inds[0], start_inds[1]]
        dest_node_ind = self.pos2node[dest_inds[0], dest_inds[1]]
        if self.graphtool:
            return self.graph.vertex(start_node_ind
                                     ), self.graph.vertex(dest_node_ind)
        else:
            return start_node_ind, dest_node_ind

    def old_add_start_end_vertices(self, start_list=None, end_list=None):
        """
        old version: connect to a list of start and a list of end vertices
        """
        tic = time.time()
        # defaults if no start and end list are given:
        topbottom, leftright = np.where(self.hard_constraints)
        if start_list is None:
            nr_start = len(topbottom) // 1000
            start_list = zip(topbottom[:nr_start], leftright[:nr_start])
        if end_list is None:
            nr_end = len(topbottom) // 1000
            end_list = zip(topbottom[-nr_end:], leftright[-nr_end:])

        # iterate over start and end and over neighbors
        neighbor_lists = [start_list, end_list]
        start_and_end = []

        for k in [0, 1]:
            # add start and end vertex
            if self.graphtool:
                v = self.graph.add_vertex()
                v_index = self.graph.vertex_index[v]
                start_and_end.append(v)
            else:
                v_index = len(self.node_pos) + k
                self.graph.add_node(v_index)
                start_and_end.append(v_index)
            print("index of start/end vertex", v_index)
            # compute edges to neighbors
            edges = []
            for (i, j) in neighbor_lists[k]:
                neighbor_ind = self.pos2node[i, j]
                edges.append(
                    [v_index, neighbor_ind, 1]
                )  # weight zero funktioniert nicht
            # add edges from start and end to neighbors
            if self.graphtool:
                self.graph.add_edge_list(edges, eprops=[self.weight])
            else:
                nx_edges = [(e[0], e[1], {"weight": e[2]}) for e in edges]
                self.graph.add_edges_from(nx_edges)
        self.time_logs["start_end_vertex"] = round(time.time() - tic, 3)
        return start_and_end[0], start_and_end[1]

    def get_shortest_path(self, source, target):
        """
        Compute shortest path from source vertex to target vertex
        """
        tic = (time.time())
        # #if source and target are given as indices:
        if self.graphtool:
            vertices_path, _ = shortest_path(
                self.graph,
                source,
                target,
                weights=self.weight,
                negative_weights=True
            )
            # exclude auxiliary start and end
            # actual_path = vertices_path[1:-1]
            path = [
                self.node_pos[self.graph.vertex_index[v]]
                for v in vertices_path
            ]
        else:
            vertices_path = nx.dijkstra_path(self.graph, source, target)
            # vertices_path = nx.bellman_ford_path(self.graph, source, target)
            path = [self.node_pos[int(v)] for v in vertices_path]  # [1:-1]]

        if self.verbose:
            print("time for shortest path", time.time() - tic)

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        return path, []

    def get_shortest_path_nx(self, source, target):
        # define cutoff
        cutoff = 4 * np.linalg.norm(vec) / PYLON_DIST_MIN

        tic = time.time()
        if source == target:
            return (0, [source])

        weight = lambda u, v, data: data.get("weight", 1)

        # paths = {source: [source]}  # dictionary of paths
        # dist = self._bellman_ford(
        #     self.graph, [source], weight, cutoff, paths=paths, target=target
        # )
        # if target is None:
        #     return (dist, paths)
        # try:
        #     vertices_path = paths[target]
        # except KeyError:
        #     msg = f"Node {target} not reachable from {source}"
        #     raise nx.NetworkXNoPath(msg)
        vertices_path = self.bellman_ford(source, target, cutoff)

        print(vertices_path)

        out_path = [self.node_pos[int(v)] for v in vertices_path]  # [1:-1]]

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        return out_path, []

    def bellman_ford(self, source, target, cutoff):
        """
        Actual BF algorithm, not SPFA
        """
        pred = {}
        dist = {source: 0}

        inf = float('inf')

        for i in range(int(cutoff)):
            print(i)
            for (u, v, w_dict) in self.graph.edges(data=True):
                w = w_dict["weight"]
                if dist.get(u, inf) + w < dist.get(v, inf):
                    dist[v] = dist[u] + w
                    pred[v] = u
        path = [target]
        curr = target
        while curr != source:
            curr = pred[curr]
            path.append(curr)
        path.append(source)
        return list(reversed(path))

    def _bellman_ford(
        self,
        G,
        source,
        weight,
        cutoff,
        pred=None,
        paths=None,
        dist=None,
        target=None
    ):
        """Relaxation loop for Bellman–Ford algorithm.
        This is an implementation of the SPFA variant.
        See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm
        Parameters
        SEE https://github.com/networkx/networkx/blob/02a1721276b3a84d3be8558e4
        79a9cb6b0715488/networkx/algorithms/shortest_paths/weighted.py#L1203
        """
        for s in source:
            if s not in G:
                raise nx.NodeNotFound(f"Source {s} not in G")

        if pred is None:
            pred = {v: [] for v in source}

        if dist is None:
            dist = {v: 0 for v in source}

        G_succ = G.succ if G.is_directed() else G.adj
        inf = float('inf')
        n = len(G)

        # count = {}
        q = deque(source)
        in_q = set(source)
        iteration = 0
        while q:
            u = q.popleft()
            in_q.remove(u)

            # Skip relaxations if any of the predecessors of u is in the queue.
            if all(pred_u not in in_q for pred_u in pred[u]):
                dist_u = dist[u]
                for v, e in G_succ[u].items():
                    dist_v = dist_u + weight(u, v, e)  # TODO:replace function

                    if dist_v < dist.get(v, inf):
                        if v not in in_q:
                            q.append(v)
                            in_q.add(v)
                            # count_v = count.get(v, 0) + 1
                            # if count_v == n:
                            #     raise nx.NetworkXUnbounded(
                            #         "Negative cost cycle detected."
                            #     )
                            # count[v] = count_v
                        dist[v] = dist_v
                        pred[v] = [u]

                    elif dist.get(v) is not None and dist_v == dist.get(v):
                        pred[v].append(u)

            iteration += 1

            # TODO
            # if u == target:
            #     print("early stopping")
            #     break
            if iteration > cutoff and dist.get(target, inf) < inf:
                # print("iteration more than cutoff")
                # if dist.get(target, inf) < inf:
                print("early stopping")
                break

        if paths is not None:
            dsts = [target] if target is not None else pred
            for dst in dsts:

                path = [dst]
                cur = dst

                while pred[cur]:
                    cur = pred[cur][0]
                    path.append(cur)

                path.reverse()
                paths[dst] = path

        return dist
