import numpy as np
import time
from graph_tool.all import Graph, shortest_path, load_graph

from power_planner.utils import angle, shift_surface, get_lg_donut, get_half_donut

from general_graph import GeneralGraph


class LineGraph(GeneralGraph):

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
        self.verbose = verbose

        self.time_logs = {}
        self.time_logs["init_graph"] = round(time.time() - tic, 3)

        # # set different costs:
        # self.angle_cost = self.graph.new_edge_property("float")
        # self.env_cost = self.graph.new_edge_property("float")
        # self.cost_props = [self.angle_cost, self.env_cost]
        # self.cost_weights = [0.25, 1]

    def set_shift(self, lower, upper, vec, max_angle):
        GeneralGraph.set_shift(self, lower, upper, vec, max_angle)
        self.angle_tuples = get_lg_donut(lower, upper, vec)

    def set_edge_costs(self, classes, weights=None):
        # classes = ["angle", "env", "urban"]  # data.layer_classes + ["angle"]
        classes.insert(0, "angle")
        if len(weights) < len(classes):
            print("insert weight 1 for angle costs")
            weights.insert(0, 1)  # append angle weight
        print("edge costs classes:", classes)
        GeneralGraph.set_edge_costs(self, classes, weights=weights)

    def add_nodes(self):
        tic = time.time()
        # Build edge dictionary
        edge_array = []
        for i in range(len(self.shifts)):
            out_edges = shift_surface(
                self.hard_constraints,
                np.asarray(self.shifts[i]) * (-1)
            )
            possible_out_edges = np.where(self.hard_constraints * out_edges)
            out_edge_2 = np.swapaxes(np.vstack(possible_out_edges), 1, 0)
            out_edge_1 = out_edge_2 + np.array(self.shifts[i])
            out_edge = np.concatenate(
                [np.expand_dims(out_edge_2, 1),
                 np.expand_dims(out_edge_1, 1)],
                axis=1
            )
            edge_array.append(out_edge)
        edge_lists_concat = np.concatenate(edge_array, axis=0)
        self.edge_dict = {
            (tuple(edge_lists_concat[i, 0]), tuple(edge_lists_concat[i, 1])): i
            for i in range(len(edge_lists_concat))
        }
        if self.verbose:
            print("Added ", len(edge_lists_concat), "vertices")
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)

    def _valid_edges(self, mask, shift):
        in_node = shift_surface(mask, np.asarray(shift[0]) * (-1))
        out_node = shift_surface(mask, np.asarray(shift[1]) * (-1))
        stacked = np.asarray([mask, in_node, out_node])
        return np.all(stacked, axis=0)

    def add_edges(self):
        tic_function = time.time()

        times_edge_list = []
        times_add_edges = []
        if self.verbose:
            print(
                "Start adding edges...", len(self.angle_tuples), "iterations"
            )
        # for every angle in the new angle tuples
        for shift in self.angle_tuples:
            tic_edges = time.time()
            # get cost for angle
            angle_weight = shift[2]
            # get all angles that are possible
            all_angles = self._valid_edges(self.hard_constraints, shift)
            node_inds = np.swapaxes(np.vstack(np.where(all_angles)), 1, 0)
            in_node = node_inds + shift[0]
            out_node = node_inds + shift[1]
            # cost at edge in lg is cost of node tower inbetween
            # node_cost = [self.cost_instance[all_angles]]
            # new version TODO
            node_cost_list = [
                cost_surface[all_angles] for cost_surface in self.cost_instance
            ]
            node_cost_arr = np.swapaxes(np.array(node_cost_list), 1, 0)

            # iterate over edges and map to node position
            edges_lg = []
            for i in range(len(node_inds)):
                e1 = self.edge_dict[(tuple(in_node[i]), tuple(node_inds[i]))]
                e2 = self.edge_dict[(tuple(node_inds[i]), tuple(out_node[i]))]
                edges_lg.append(
                    [e1, e2, angle_weight] + node_cost_arr[i].tolist()
                )
            # save time
            times_edge_list.append(round(time.time() - tic_edges, 3))
            # add to graph
            tic_graph = time.time()
            self.graph.add_edge_list(edges_lg, eprops=self.cost_props)
            times_add_edges.append(round(time.time() - tic_graph, 3))

        self.time_logs["add_edges"] = round(np.mean(times_add_edges), 3)
        self.time_logs["add_edges_times"] = times_add_edges

        self.time_logs["edge_list"] = round(np.mean(times_edge_list), 3)
        self.time_logs["edge_list_times"] = times_edge_list

        self.time_logs["add_all_edges"] = round(time.time() - tic_function, 3)

        if self.verbose:
            print("Done adding edges:", len(list(self.graph.edges())))

    def add_start_and_dest(self, source, dest):
        tic = time.time()
        possible_start_edges = []
        for shift in self.shifts:
            neighbor = np.asarray(source) + np.asarray(shift)
            node_val = self.edge_dict.get((tuple(source), tuple(neighbor)), -1)
            if node_val > 0:
                possible_start_edges.append(node_val)

        possible_dest_edges = []
        for shift in self.shifts:
            neighbor = np.asarray(dest) - np.asarray(shift)
            node_val = self.edge_dict.get((tuple(neighbor), tuple(dest)), -1)
            if node_val > 0:
                possible_dest_edges.append(node_val)

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
        vertices_path = GeneralGraph.get_shortest_path(self, source, dest)
        edge_mapping = [
            k for k, _ in
            sorted(self.edge_dict.items(), key=lambda item: item[1])
        ]
        # convert
        out_path = [
            edge_mapping[self.graph.vertex_index[v]][0]
            for v in vertices_path[1:-1]
        ]
        # append last one
        out_path.append(
            edge_mapping[self.graph.vertex_index[vertices_path[-2]]][1]
        )
        # version with costs
        out_costs = []
        for i in range(len(vertices_path) - 1):
            edge = self.graph.edge(vertices_path[i], vertices_path[i + 1])
            out_costs.append([c[edge] for c in self.cost_props])
            # costs.append(self.weight[edge]) # TODO: add all costs ?
        return out_path, out_costs


class LineGraphFromGraph():

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
