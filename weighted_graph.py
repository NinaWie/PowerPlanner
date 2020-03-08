import numpy as np
from graph_tool.all import Graph, shortest_path, load_graph
import time

from constraints import convolve, get_kernel
from power_planner.utils import shift_surface


class WeightedGraph():

    def __init__(self, cost_instance, hard_constraints, verbose=1):
        assert cost_instance.shape == hard_constraints.shape
        # , "Cost size must be equal to corridor definition size!"

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

        # Declare graph
        self.graph = Graph(directed=False)
        self.weight = self.graph.new_edge_property("float")

        # print statements:
        self.verbose = verbose

    def add_nodes(self):
        tic = time.time()
        # add nodes to graph
        vlist = self.graph.add_vertex(len(self.node_pos))
        self.n_vertices = len(list(vlist))
        if self.verbose:
            print(
                "Added nodes:", self.n_vertices, "in time:",
                time.time() - tic
            )

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

        tic = time.time()
        # add edges and properties to the graph
        self.graph.add_edge_list(edge_list, eprops=[self.weight])
        if self.verbose:
            print("added edges:", len(list(self.graph.edges())))

            print("finished adding edges", time.time() - tic)

    def add_edges(self, shifts):
        tic_function = time.time()
        inds_orig = self.pos2node[self.hard_constraints > 0]

        self.cost_rest = self.cost_instance * (self.hard_constraints >
                                               0).astype(int)
        n_edges = 0

        kernels, posneg = get_kernel(shifts)

        for i in range(len(shifts)):

            tic_edges = time.time()
            costs_shifted = shift_surface(self.cost_rest, shifts[i])

            # weights = (costs_shifted + self.cost_rest) / 2
            # new version: edge weights
            weights = convolve(self.cost_rest, kernels[i], posneg[i])

            inds_shifted = self.pos2node[costs_shifted > 0]

            # delete the ones where inds_shifted is zero
            assert len(inds_shifted) == len(inds_orig)
            weights_list = weights[costs_shifted > 0]

            pos_inds = inds_shifted >= 0
            out = np.swapaxes(
                np.asarray([inds_orig, inds_shifted, weights_list]), 1, 0
            )[pos_inds]

            if self.verbose:
                print(
                    "finished number", i, "of defining edge list",
                    time.time() - tic_edges
                )
            n_edges += len(out)

            # add edges
            tic_graph = time.time()
            self.graph.add_edge_list(out, eprops=[self.weight])
            if self.verbose:
                print(
                    "finished", i, "of adding edges to graph",
                    time.time() - tic_graph
                )
        if self.verbose:
            print("DONE adding", n_edges, "edges:", time.time() - tic_function)

    def add_start_end_vertices(self, start_list=None, end_list=None):
        # defaults if no start and end list are given:
        topbottom, leftright = np.where(self.hard_constraints)
        if start_list is None:
            nr_start = len(topbottom) // 100
            start_list = zip(topbottom[:nr_start], leftright[:nr_start])
        if end_list is None:
            nr_end = len(topbottom) // 100
            end_list = zip(topbottom[-nr_end:], leftright[-nr_end:])

        # iterate over start and end and over neighbors
        neighbor_lists = [start_list, end_list]
        start_and_end = []

        for k in [0, 1]:
            v = self.graph.add_vertex()
            v_index = self.graph.vertex_index[v]
            start_and_end.append(v)
            print("index of start/end vertex", v_index)
            edges = []
            for (i, j) in neighbor_lists[k]:
                neighbor_ind = self.pos2node[i, j]
                edges.append([v_index, neighbor_ind, 0])
            self.graph.add_edge_list(edges, eprops=[self.weight])

        return start_and_end[0], start_and_end[1]

    def shortest_path(self, source, target):
        """
        Compute shortest path from source vertex to target vertex
        """
        tic = (time.time())
        # #if source and target are given as indices:
        # source = self.graph.vertex(source)
        # target = self.graph.vertex(target)
        vertices_path, _ = shortest_path(
            self.graph,
            source,
            target,
            weights=self.weight,
            negative_weights=True
        )

        # exclude auxiliary start and end
        actual_path = vertices_path[1:-1]

        path = [self.node_pos[self.graph.vertex_index[v]] for v in actual_path]
        if self.verbose:
            print("time for shortest path", time.time() - tic)

        return path

    def save_graph(self, OUT_PATH):
        self.graph.save(OUT_PATH + ".xml.gz")

    def load_graph(self, IN_PATH):
        self.graph = load_graph(IN_PATH + ".xml.gz")
        self.weight = self.graph.ep.weight
        # weight = G2.ep.weight[G2.edge(66, 69)]
