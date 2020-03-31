import numpy as np
import time
from graph_tool.all import Graph, shortest_path, load_graph

from power_planner.utils import angle, get_lg_donut
from power_planner.constraints import ConstraintUtils
from power_planner.utils_instance import CostUtils

from .general_graph import GeneralGraph


class LineGraph(GeneralGraph):
    """
    Build a line graph for incorporating angle costs
    """

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        tic = time.time()
        # assert cost_instance.shape == hard_constraints.shape
        self.cost_instance = cost_instance
        self.hard_constraints = hard_constraints

        # initilize graph
        GeneralGraph.__init__(
            self, directed=directed, graphtool=graphtool, verbose=verbose
        )

        self.x_len, self.y_len = hard_constraints.shape
        self.pos2node = np.arange(1, self.x_len * self.y_len + 1).reshape(
            (self.x_len, self.y_len)
        )

        self.time_logs = {}
        self.time_logs["init_graph"] = round(time.time() - tic, 3)

    def _edge2node(self, v1_arr, shift):
        """
        binary arrays of same shape as pos2v1
        """
        neighbor_ind = self.shift_dict[tuple(shift)]
        return self.pos2node[v1_arr] * self.n_neighbors + neighbor_ind
        # return self.pos2node[v1_arr] * self.n_entries + self.pos2node[v2_arr]

    # def set_corridor(self, factor, dist_surface, start_inds, dest_inds):
    #     # original pos2node: all filled except for hard constraints
    #     GeneralGraph.set_corridor(
    #         self, factor, dist_surface, start_inds, dest_inds
    #     )

    def set_shift(self, lower, upper, vec, max_angle):
        """
        Get donut tuples (for vertices) and angle tuples (for edges)
        """
        GeneralGraph.set_shift(self, lower, upper, vec, max_angle)
        self.shift_tuples = get_lg_donut(lower, upper, vec)
        self.n_neighbors = len(self.shifts)
        self.shift_dict = {tuple(s): i for i, s in enumerate(self.shifts)}

    def set_edge_costs(self, classes, weights=None):
        """
        Initialize edge properties as in super, but add angle costs
        """
        # classes = ["angle", "env", "urban"]  # data.layer_classes + ["angle"]
        classes = ["angle"] + classes
        if len(weights) < len(classes):
            print("insert weight 1 for angle costs")
            weights = [1] + list(weights)  # append angle weight
        print("edge costs classes:", classes)
        GeneralGraph.set_edge_costs(self, classes, weights=weights)

    def add_nodes(self):
        GeneralGraph.add_nodes(
            self, self.x_len * self.y_len * self.n_neighbors
        )

    def set_cost_rest(self):
        # in case of the non-random graph, we don't have to set cost_rest
        pass

    def _compute_edges(self, shift):
        """
        Get all valid edges given a certain shift
        :param mask: binary 2d array marking forbidden areas
        """
        # get all angles that are possible
        in_node = ConstraintUtils.shift_surface(
            self.hard_constraints,
            np.asarray(shift[0]) * (-1)
        )
        out_node = ConstraintUtils.shift_surface(
            self.hard_constraints,
            np.asarray(shift[1]) * (-1)
        )
        stacked = np.asarray([self.hard_constraints, in_node, out_node])
        all_angles = np.all(stacked, axis=0)

        # shift again, andersrum
        in_node = ConstraintUtils.shift_surface(
            all_angles, np.asarray(shift[0])
        )
        out_node = ConstraintUtils.shift_surface(
            all_angles, np.asarray(shift[1])
        )

        e1 = self._edge2node(
            in_node,
            np.array(shift[0]) * (-1)
        )  # in_node, all_angles,
        e2 = self._edge2node(all_angles, shift[1])

        # new version TODO
        node_cost_arr = np.array(
            [cost_surface[all_angles] for cost_surface in self.cost_rest]
        )

        pos = (np.sum(node_cost_arr, axis=0) > 0).astype(bool)

        weights_arr = node_cost_arr[:, pos]

        angle_weight = [shift[2] for _ in range(sum(pos.astype(int)))]
        inds_arr = np.asarray([e1[pos], e2[pos], angle_weight])

        inds_weights = np.concatenate((inds_arr, weights_arr), axis=0)
        edges_lg = np.swapaxes(inds_weights, 1, 0)

        return edges_lg

    def add_start_and_dest(self, source, dest):
        """
        start and dest are no vertices in line graph, so need to add them
        seperately
        --> get all outgoing edges from source
        --> create new start vertex, connect to all outgoing edges
        :returns: newly created source and dest vertices
        """
        tic = time.time()
        possible_start_edges = [
            self.pos2node[source[0], source[1]] * self.n_neighbors +
            self.shift_dict[tuple(shift)] for shift in self.shifts
        ]

        possible_dest_edges = []
        for shift in self.shifts:
            shifted_dest = np.asarray(dest) - np.asarray(shift)
            possible_dest_edges.append(
                self.pos2node[shifted_dest[0], shifted_dest[1]] *
                self.n_neighbors + self.shift_dict[tuple(shift)]
            )

        start_v = self.graph.add_vertex()
        dest_v = self.graph.add_vertex()
        start_ind = self.graph.vertex_index[start_v]
        dest_ind = self.graph.vertex_index[dest_v]

        mean_costs = np.mean(self.cost_instance, axis=(1, 2)).tolist()
        mean_costs.insert(0, 0)  # insert zero angle cost
        print("mean costs:", mean_costs)  # TODO: leave this?

        start_edges = [
            [start_ind, u] + mean_costs for u in possible_start_edges
        ]
        dest_edges = [[u, dest_ind] + mean_costs for u in possible_dest_edges]
        self.graph.add_edge_list(start_edges, eprops=self.cost_props)
        self.graph.add_edge_list(dest_edges, eprops=self.cost_props)

        self.time_logs["add_start_end"] = round(time.time() - tic, 3)

        return [start_v, dest_v]

    def get_shortest_path(self, source, dest):
        """
        Compute shortest path and convert from line graph representation to 
        coordinates
        """
        vertices_path = GeneralGraph.get_shortest_path(self, source, dest)
        out_path = []
        for v in vertices_path[1:-1]:
            start_node = int(v) // self.n_neighbors
            shift_ind = int(v) % self.n_neighbors
            start_pos = [start_node // self.y_len, start_node % self.y_len]
            out_path.append(start_pos)

        # append last on
        out_path.append(list(np.array(start_pos) + self.shifts[shift_ind]))

        out_costs = []
        for (i, j) in out_path:
            out_costs.append(self.cost_instance[:, i, j].tolist())

        return out_path, out_costs


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
