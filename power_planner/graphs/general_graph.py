import numpy as np
from graph_tool.all import Graph, shortest_path, load_graph
import time
import networkx as nx
from power_planner.utils import get_half_donut
from power_planner.plotting import plot_pareto_scatter_2d, plot_pareto_scatter_3d, plot_pareto_3d
from power_planner.utils_instance import CostUtils


class GeneralGraph():
    """
    General wrapper for graph-tool or networkx graphs to add edges and nodes
    according to constraints
    """

    def __init__(self, graphtool=1, directed=True, verbose=1):
        # Initialize graph
        if graphtool:
            self.graph = Graph(directed=directed)
            self.weight = self.graph.new_edge_property("float")
        else:
            if directed:
                print("directed graph")
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.Graph()
        # set metaparameter
        self.time_logs = {}
        self.verbose = verbose
        self.graphtool = graphtool

    def set_edge_costs(self, classes, weights, angle_weight=0):
        """
        Initialize edge cost variables
        :param classes: list of cost categories
        :param weights: list of weights for cost categories - must be of same 
                        shape as classes (if None, then equal weighting)
        """
        weights = np.array(weights)
        # set different costs:
        self.cost_classes = classes
        self.cost_props = [
            self.graph.new_edge_property("float") for _ in range(len(classes))
        ]
        self.cost_weights = weights / np.sum(weights)
        print(self.cost_classes, self.cost_weights)
        # save weighted instance for plotting
        self.instance = np.sum(
            np.moveaxis(self.cost_instance, 0, -1) * self.cost_weights, axis=2
        ) * self.hard_constraints

    def set_shift(self, lower, upper, vec, max_angle):
        """
        Initialize shift variable by getting the donut values
        :param lower, upper: min and max distance of pylons
        :param vec: vector of diretion of edges
        :param max_angle: Maximum angle of edges to vec
        """
        self.shifts = get_half_donut(lower, upper, vec, angle_max=max_angle)
        self.shift_tuples = self.shifts

    def set_corridor(
        self, factor, dist_surface, start_inds, dest_inds, sample_func,
        sample_method
    ):
        # set new corridor
        corridor = (dist_surface > 0).astype(int)

        self.factor = factor
        self.cost_rest = self.cost_instance * (self.hard_constraints >
                                               0).astype(int) * corridor
        # downsample
        tic = time.time()
        if factor > 1:
            self.cost_rest = CostUtils.downsample(
                self.cost_rest, factor, mode=sample_method, func=sample_func
            )

        self.time_logs["downsample"] = round(time.time() - tic, 3)

        # repeat because edge artifacts
        self.cost_rest = self.cost_rest * (self.hard_constraints >
                                           0).astype(int) * corridor

        # add start and end TODO ugly
        self.cost_rest[:, dest_inds[0],
                       dest_inds[1]] = self.cost_instance[:, dest_inds[0],
                                                          dest_inds[1]]
        self.cost_rest[:, start_inds[0],
                       start_inds[1]] = self.cost_instance[:, start_inds[0],
                                                           start_inds[1]]

    def add_nodes(self, nodes):
        """
        Add vertices to the graph
        param nodes: list of node names if networkx, integer if graphtool
        """
        tic = time.time()
        # add nodes to graph
        if self.graphtool:
            _ = self.graph.add_vertex(nodes)
            self.n_nodes = len(list(self.graph.vertices()))
        else:
            self.graph.add_nodes_from(np.arange(nodes))
            self.n_nodes = len(self.graph.nodes())
        # verbose
        if self.verbose:
            print("Added nodes:", nodes, "in time:", time.time() - tic)
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)

    def add_edges(self):
        tic_function = time.time()

        n_edges = 0
        # kernels, posneg = ConstraintUtils.get_kernel(self.shifts,
        # self.shift_vals)
        # edge_array = []

        times_edge_list = []
        times_add_edges = []

        if self.verbose:
            print("n_neighbors:", len(self.shift_tuples))

        for i in range(len(self.shift_tuples)):

            tic_edges = time.time()

            # set cost rest if necessary (random graph)
            self.set_cost_rest()

            # compute shift and weights
            out = self._compute_edges(self.shift_tuples[i])

            # Error if -1 entries because graph-tool crashes with -1 nodes
            if np.any(out[:2].flatten() == -1):
                print(np.where(out[:2] == -1))
                raise RuntimeError

            n_edges += len(out)
            times_edge_list.append(round(time.time() - tic_edges, 3))

            # add edges to graph
            tic_graph = time.time()
            if self.graphtool:
                self.graph.add_edge_list(out, eprops=self.cost_props)
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
        self.n_edges = len(list(self.graph.edges()))
        self._update_time_logs(times_add_edges, times_edge_list, tic_function)
        if self.verbose:
            print("DONE adding", n_edges, "edges:", time.time() - tic_function)

    def _update_time_logs(
        self, times_add_edges, times_edge_list, tic_function
    ):
        self.time_logs["add_edges"] = round(np.mean(times_add_edges), 3)
        self.time_logs["add_edges_times"] = times_add_edges

        self.time_logs["edge_list"] = round(np.mean(times_edge_list), 3)
        self.time_logs["edge_list_times"] = times_edge_list

        self.time_logs["add_all_edges"] = round(time.time() - tic_function, 3)

        if self.verbose:
            print("Done adding edges:", len(list(self.graph.edges())))

    def sum_costs(self):
        """
        Additive weighting of costs
        Take the individual edge costs, compute weighted sum --> self.weight
        """
        # add sum of all costs
        tic = time.time()
        summed_costs_arr = np.zeros(self.cost_props[0].get_array().shape)
        for i in range(len(self.cost_props)):
            prop = self.cost_props[i].get_array()
            summed_costs_arr += prop * self.cost_weights[i]
        self.weight.a = summed_costs_arr

        self.time_logs["sum_of_costs"] = round(time.time() - tic, 3)

    def remove_vertices(self, dist_surface, delete_padding=0):
        """
        Remove edges in a certain corridor (or all) to replace them by
        a refined surface

        @param dist_surface: a surface where each pixel value corresponds to 
        the distance of the pixel to the shortest path
        @param delete_padding: define padding in which part of the corridor to 
        delete vertices (cannot delete all because then graph unconnected)
        """
        tic = time.time()
        self.graph.clear_edges()
        self.graph.shrink_to_fit()
        self.time_logs["remove_edges"] = round(time.time() - tic, 3)

    def get_pareto(
        self, vary, source, dest, out_path=None, compare=[0, 1], plot=1
    ):
        """
        Arguments:
            vary: how many weights to explore
                    e.g 3 --> each cost class can have weight 0, 0.5 or 1
            source, dest: as always the source and destination vertex
            out_path: where to save the pareto figure(s)
            compare: indices of cost classes to compare
        Returns:
            paths: All found paths
            pareto: The costs for each combination of weights
        """
        tic = time.time()
        # initialize lists
        pareto = list()
        paths = list()
        cost_sum = list()
        # get the edge costs
        cost_arrs = [cost.get_array() for cost in self.cost_props]
        # [self.cost_props[comp].get_array() for comp in compare]

        # get vary weights between 0 and 1
        var_weights = np.around(np.linspace(0, 1, vary), 2)

        # construct weights array
        if len(compare) == 2:
            weights = [[v, 1 - v] for v in var_weights]
        elif len(compare) == 3:
            weights = list()
            for w0 in var_weights:
                for w1 in var_weights[var_weights <= 1 - w0]:
                    weights.append([w0, w1, 1 - w0 - w1])
        else:
            raise ValueError("argument compare can only have length 2 or 3")

        # w_avail: keep weights of non-compare classes, get leftover amount
        w_avail = np.sum(np.asarray(self.cost_weights)[compare])
        # compute paths for each combination of weights
        for j in range(len(weights)):
            # option 2: np.zeros(len(cost_arrs)) + non_compare_weight
            w = self.cost_weights.copy()
            # replace the ones we want to compare
            w[compare] = np.array(weights[j]) * w_avail

            # weighted sum of edge costs
            self.weight.a = np.sum(
                [cost_arrs[i] * w[i] for i in range(len(cost_arrs))], axis=0
            )
            # get shortest path
            path, path_costs, _ = self.get_shortest_path(source, dest)
            # don't take cost_sum bc this is sum of original weighting
            pareto.append(np.sum(path_costs, axis=0)[compare])
            paths.append(path)
            # take overall sum of costs (unweighted) that this w leads to
            cost_sum.append(np.sum(path_costs))

        # print best weighting
        best_weight = np.argmin(cost_sum)
        w = self.cost_weights.copy()
        w[compare] = np.array(weights[best_weight]) * w_avail
        print("Best weights:", w, "with (unweighted) costs:", np.min(cost_sum))

        self.time_logs["pareto"] = round(time.time() - tic, 3)

        pareto = np.array(pareto)
        classes = [self.cost_classes[comp] for comp in compare]
        # Plotting
        if plot:
            if len(compare) == 2:
                plot_pareto_scatter_2d(
                    pareto, weights, classes, out_path=out_path
                )
            elif len(compare) == 3:
                # plot_pareto_3d(pareto, weights, classes)
                plot_pareto_scatter_3d(
                    pareto, weights, classes, out_path=out_path
                )
        return paths, weights, cost_sum

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
        else:
            vertices_path = nx.dijkstra_path(self.graph, source, target)

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        return vertices_path

    def save_graph(self, OUT_PATH):
        """
        Save the graph in OUT_PATH
        """
        if self.graphtool:
            for i, cost_class in enumerate(self.cost_classes):
                self.graph.edge_properties[cost_class] = self.cost_props[i]
            self.graph.edge_properties["weight"] = self.weight
            self.graph.save(OUT_PATH + ".xml.gz")
        else:
            nx.write_weighted_edgelist(
                self.graph, OUT_PATH + '.weighted.edgelist'
            )

    def load_graph(self, IN_PATH):
        """
        Retrieve graph from IN_PATH
        """
        if self.graphtool:
            self.g_prev = load_graph(IN_PATH + ".xml.gz")
            self.weight_prev = self.g_prev.ep.weight
            # weight = G2.ep.weight[G2.edge(66, 69)]
        else:
            self.g_prev = nx.read_edgelist(
                IN_PATH + '.weighted.edgelist',
                nodetype=int,
                data=(('weight', float), )
            )
