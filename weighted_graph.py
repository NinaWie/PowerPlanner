import numpy as np
from graph_tool.all import Graph, shortest_path
import time


class WeightedGraph():

    def __init__(self, cost_instance, hard_constraints):
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

    def add_nodes(self):
        tic = time.time()
        # add nodes to graph
        vlist = self.graph.add_vertex(len(self.node_pos))
        self.n_vertices = len(list(vlist))
        print("Added nodes:", self.n_vertices, "in time:", time.time() - tic)

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
        print("time to build edge list:", time.time() - tic)

        tic = time.time()
        # add edges and properties to the graph
        self.graph.add_edge_list(edge_list, eprops=[self.weight])
        print("added edges:", len(list(self.graph.edges())))

        print("finished adding edges", time.time() - tic)

    def add_edges(self, shifts, shift_tuples):
        inds_orig = self.pos2node[self.hard_constraints > 0]

        self.cost_rest = self.cost_instance * (self.hard_constraints >
                                               0).astype(int)
        print("cost_rest", self.cost_rest.shape)

        for i in range(len(shift_tuples)):
            costs_shifted = np.pad(
                self.cost_rest, shift_tuples[i], mode='constant'
            )
            shift = shifts[i]
            if shift[0] > 0 and shift[1] > 0:
                costs_shifted = costs_shifted[:-shift[0], :-shift[1]]
            elif shift[0] > 0 and shift[1] <= 0:
                costs_shifted = costs_shifted[:-shift[0], -shift[1]:]
            elif shift[0] <= 0 and shift[1] > 0:
                costs_shifted = costs_shifted[-shift[0]:, :-shift[1]]
            elif shift[0] <= 0 and shift[1] <= 0:
                costs_shifted = costs_shifted[-shift[0]:, -shift[1]:]

            weights = (costs_shifted + self.cost_rest) / 2
            inds_shifted = self.pos2node[costs_shifted > 0]

            # delete the ones where inds_shifted is zero
            assert len(inds_shifted) == len(inds_orig)
            weights_list = weights[costs_shifted > 0]

            pos_inds = inds_shifted >= 0
            out = np.swapaxes(
                np.asarray([inds_orig, inds_shifted, weights_list]), 1, 0
            )[pos_inds]

            # add edges
            self.graph.add_edge_list(out, eprops=[self.weight])

    def shortest_path(self, source_ind, target_ind):
        # ### Compute shortest path
        tic = (time.time())
        vertices_path, _ = shortest_path(
            self.graph,
            self.graph.vertex(source_ind),
            self.graph.vertex(target_ind),
            weights=self.weight,
            negative_weights=True
        )

        path = [
            self.node_pos[self.graph.vertex_index[v]] for v in vertices_path
        ]
        print("time for shortest path", time.time() - tic)
        return path
