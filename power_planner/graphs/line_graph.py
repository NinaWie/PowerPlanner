import numpy as np
import time

from power_planner.utils.utils import get_lg_donut
from power_planner.utils.utils_constraints import ConstraintUtils
from power_planner.utils.utils_costs import CostUtils

from .general_graph import GeneralGraph


class LineGraph(GeneralGraph):
    """
    Build a line graph for incorporating angle costs
    """

    def __init__(
        self, cost_instance, hard_constraints, directed=True, verbose=1
    ):
        tic = time.time()
        # assert cost_instance.shape == hard_constraints.shape
        self.cost_instance = cost_instance
        self.hard_constraints = hard_constraints
        self.cost_rest = self.cost_instance * (self.hard_constraints >
                                               0).astype(int)

        # initilize graph
        GeneralGraph.__init__(self, directed=directed, verbose=verbose)

        self.x_len, self.y_len = hard_constraints.shape
        self.pos2node = np.arange(self.x_len * self.y_len).reshape(
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

    def set_shift(
        self,
        start,
        dest,
        pylon_dist_min=3,
        pylon_dist_max=5,
        max_angle=np.pi / 2,
        max_angle_lg=np.pi / 4,
        **kwargs
    ):
        # self, lower, upper, vec, max_angle, max_angle_lg=np.pi / 4):
        """
        Get donut tuples (for vertices) and angle tuples (for edges)
        """
        GeneralGraph.set_shift(
            self,
            start,
            dest,
            pylon_dist_min=pylon_dist_min,
            pylon_dist_max=pylon_dist_max,
            max_angle=max_angle
        )
        self.max_angle_lg = max_angle_lg
        self.shift_tuples = get_lg_donut(
            pylon_dist_min,
            pylon_dist_max,
            dest - start,
            max_angle,
            max_angle_lg=max_angle_lg
        )
        self.n_neighbors = len(self.shifts)
        self.shift_dict = {tuple(s): i for i, s in enumerate(self.shifts)}

    def set_edge_costs(
        self,
        layer_classes=["resistance"],
        class_weights=[1],
        angle_weight=0.5,
        **kwargs
    ):
        """
        Initialize edge properties as in super, but add angle costs
        """
        classes_w_ang = ["angle"] + layer_classes
        GeneralGraph.set_edge_costs(self, classes_w_ang, class_weights)
        weights_norm = np.array(
            [angle_weight * np.sum(class_weights)] + list(class_weights)
        )
        self.cost_weights = weights_norm / np.sum(weights_norm)
        if self.verbose:
            print(self.cost_classes, self.cost_weights)

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
        # print("add", len(edges_lg))
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
            try:
                possible_dest_edges.append(
                    self.pos2node[shifted_dest[0], shifted_dest[1]] *
                    self.n_neighbors + self.shift_dict[tuple(shift)]
                )
            except IndexError:
                pass

        if self.graphtool:
            start_v = self.graph.add_vertex()
            dest_v = self.graph.add_vertex()
            start_ind = self.graph.vertex_index[start_v]
            dest_ind = self.graph.vertex_index[dest_v]
        else:
            start_v = self.n_nodes
            dest_v = self.n_nodes + 1
            self.graph.add_nodes_from([start_v, dest_v])

        mean_costs = np.mean(self.cost_instance, axis=(1, 2)).tolist()
        mean_costs.insert(0, 0)  # insert zero angle cost

        if self.graphtool:
            start_edges = [
                [start_ind, u] + mean_costs for u in possible_start_edges
            ]
            dest_edges = [
                [u, dest_ind] + mean_costs for u in possible_dest_edges
            ]
            self.graph.add_edge_list(start_edges, eprops=self.cost_props)
            self.graph.add_edge_list(dest_edges, eprops=self.cost_props)
        else:
            nx_edge_list = [
                (start_v, u, {
                    "weight": 0
                }) for u in possible_start_edges
            ] + [(u, dest_v, {
                "weight": 0
            }) for u in possible_dest_edges]
            self.graph.add_edges_from(nx_edge_list)

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
        if len(out_path) > 0:
            out_path.append(list(np.array(start_pos) + self.shifts[shift_ind]))

        # compute costs: angle costs
        ang_costs = CostUtils.compute_angle_costs(out_path, self.max_angle_lg)
        out_costs = list()
        for k, (i, j) in enumerate(out_path):
            out_costs.append(
                [ang_costs[k]] + self.cost_instance[:, i, j].tolist()
            )
        # for i in range(len(vertices_path) - 1):
        #     edge = self.graph.edge(vertices_path[i], vertices_path[i + 1])
        #     out_costs.append([c[edge] for c in self.cost_props])

        weighted_sum = np.dot(
            self.cost_weights, np.sum(np.array(out_costs), axis=0)
        )
        return out_path, out_costs, weighted_sum

    # LG Utils
    # @staticmethod
    # def edge_tuple_2_pos_tuples(n1, n2, graph):
    #     (x,y) = (n1//graph.n_neighbors, n1 % graph.n_neighbors)
    #     tup1 = ((x // graph.y_len, x % graph.y_len), (y // graph.y_len,
    #               y % graph.y_len))
    #     (x,y) = (n2//graph.n_neighbors, n2 % graph.n_neighbors)
    #     tup2 = ((x // graph.y_len, x % graph.y_len), (y // graph.y_len,
    #               y % graph.y_len))
    #     return tup1, tup2
    # edge_tuple_2_pos_tuples(6716500,1062881, graph)
    # # graph.angle_tuples[0]
