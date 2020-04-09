import numpy as np
import time
from graph_tool.all import Graph, shortest_path, load_graph

from power_planner.utils import angle, get_lg_donut
from power_planner.constraints import ConstraintUtils
from power_planner.utils_instance import CostUtils

from .general_graph import GeneralGraph


class LineGraphFromGraph():
    """
    Class to build a line graph from a given weighted graph
    """

    def __init__(
        self,
        prev_graph,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        tic = time.time()
        assert cost_instance.shape == hard_constraints.shape
        self.cost_instance = cost_instance
        self.hard_constraints = hard_constraints

        # Load graph
        GeneralGraph.load_graph(self, prev_graph)
        # self.weight_prev = prev_graph.ep.weight
        self.n_edges = len(list(self.g_prev.edges()))

        # node to pos mapping
        x_len, y_len = cost_instance.shape
        self.node_pos = [
            (i, j) for i in range(x_len) for j in range(y_len)
            if hard_constraints[i, j]
        ]
        # pos to node mapping
        self.pos2node = np.ones(cost_instance.shape)
        self.pos2node *= -1
        for n, (i, j) in enumerate(self.node_pos):
            self.pos2node[i, j] = n
        print("initialized weighted graph (pos2node and node_pos)")

        # edge to node mapping
        max_shape = (
            int(np.max(self.pos2node)) + 1, int(np.max(self.pos2node)) + 1
        )
        self.edge_to_node = np.ones(max_shape)
        self.edge_to_node *= -1
        for k, edge in enumerate(self.g_prev.edges()):
            (i, j) = tuple(edge)
            self.edge_to_node[int(i), int(j)] = k

        # initilize graph
        GeneralGraph.__init__(
            self, directed=directed, graphtool=graphtool, verbose=verbose
        )
        self.verbose = verbose

        self.time_logs = {}
        self.time_logs["init_graph"] = round(time.time() - tic, 3)

    def add_nodes(self):
        tic_function = time.time()
        GeneralGraph.add_nodes(self, self.n_edges)
        self.time_logs["add_nodes"] = round(time.time() - tic_function, 3)

    def add_edges(self, max_angle=0.5 * np.pi):
        tic_edges = time.time()
        edges = []
        for i, v in enumerate(self.g_prev.vertices()):
            for in_nb in v.in_neighbours():
                for out_nb in v.out_neighbours():
                    in_nb_ind = self.node_pos[int(in_nb)]
                    out_nb_ind = self.node_pos[int(out_nb)]
                    pos = self.node_pos[i]
                    # vector between: subtract two pos tuples
                    vec1 = np.subtract(in_nb_ind, pos)
                    vec2 = np.subtract(pos, out_nb_ind)
                    angle_cost = angle(vec1, vec2) / (max_angle)
                    if angle_cost <= 1:
                        v1_line = self.edge_to_node[int(in_nb), i]
                        v2_line = self.edge_to_node[i, int(out_nb)]
                        cost_before = self.cost_instance[pos[0], pos[1]]
                        edges.append(
                            [v1_line, v2_line, 0.5 * angle_cost + cost_before]
                        )
        toc_edges = time.time()

        tic = time.time()
        self.graph.add_edge_list(edges, eprops=[self.weight])

        # time logs
        self.time_logs["add_edges"] = round(time.time() - tic, 3)
        self.time_logs["add_edges_times"] = 0
        self.time_logs["edge_list"] = round(toc_edges - tic_edges, 3)
        self.time_logs["edge_list_times"] = 0

        self.time_logs["add_all_edges"] = round(time.time() - tic_edges, 3)

    def add_start_and_dest(self, source_pos, dest_pos):
        tic = time.time()

        source = self.pos2node[source_pos[0], source_pos[1]]
        dest = self.pos2node[dest_pos[0], dest_pos[1]]
        source_line = self.graph.add_vertex()
        dest_line = self.graph.add_vertex()

        source_dest_edges = []
        for e_out in self.g_prev.vertex(source).out_edges():
            e_out = tuple(e_out)
            node_line = self.edge_to_node[int(e_out[0]), int(e_out[1])]
            source_dest_edges.append(
                [self.graph.vertex_index[source_line], node_line, 0]
            )

        for e_out in self.g_prev.vertex(dest).in_edges():
            e_out = tuple(e_out)
            node_line = self.edge_to_node[int(e_out[0]), int(e_out[1])]
            source_dest_edges.append(
                [node_line, self.graph.vertex_index[dest_line], 0]
            )

        self.graph.add_edge_list(source_dest_edges, eprops=[self.weight])

        self.time_logs["add_start_end"] = round(time.time() - tic, 3)

        return source_line, dest_line

    def get_shortest_path(self, source, dest):
        vertices_path = GeneralGraph.get_shortest_path(self, source, dest)
        path_line = []
        for i, v in enumerate(vertices_path[1:-1]):
            v_ind_line = self.graph.vertex_index[v]
            edge_actual = tuple(list(self.g_prev.edges())[v_ind_line])
            if i == 0:
                path_line.append(
                    self.node_pos[self.g_prev.vertex_index[edge_actual[0]]]
                )
            path_line.append(
                self.node_pos[self.g_prev.vertex_index[edge_actual[1]]]
            )
        return path_line, []
