import numpy as np
from graph_tool.all import Graph, shortest_path, load_graph, find_edge
import time
import networkx as nx
from power_planner.utils import get_half_donut
from power_planner.plotting import plot_pareto


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

    def set_edge_costs(self, classes, weights=None):
        """
        Initialize edge cost variables
        :param classes: list of cost categories
        :param weights: list of weights for cost categories - must be of same 
                        shape as classes (if None, then equal weighting)
        """
        if weights is None:
            weights = [1 for i in range(len(classes))]
        weights = np.array(weights)
        # set different costs:
        self.cost_classes = classes
        self.cost_props = [
            self.graph.new_edge_property("float") for _ in range(len(classes))
        ]
        self.cost_weights = weights / np.sum(weights)
        print(self.cost_classes, self.cost_weights)

    def set_shift(self, lower, upper, vec, max_angle):
        """
        Initialize shift variable by getting the donut values
        :param lower, upper: min and max distance of pylons
        :param vec: vector of diretion of edges
        :param max_angle: Maximum angle of edges to vec
        """
        self.shifts = get_half_donut(lower, upper, vec, angle_max=max_angle)

    def add_nodes(self, nodes):
        """
        Add vertices to the graph
        param nodes: list of node names if networkx, integer if graphtool
        """
        tic = time.time()
        # add nodes to graph
        if self.graphtool:
            _ = self.graph.add_vertex(nodes)
        else:
            self.graph.add_nodes_from(np.arange(nodes))
        # verbose
        if self.verbose:
            print("Added nodes:", nodes, "in time:", time.time() - tic)
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)

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

    def get_pareto(self, vary, source, dest, out_path=None, compare=[0, 1]):
        """
        compute shortest paths with varied weights
        :param vary: list of weights to compute
        :param source, dest: source and dest vertex to compute SP
        :param out_path: Plot pareto fronteir and save plot in out_path
        :param compare: indices of cost surface to compare -->
                ONLY LEN 2 possible so far!
        """
        pareto = list()
        paths = list()
        cost0 = self.cost_props[compare[0]].get_array()
        cost1 = self.cost_props[compare[1]].get_array()
        class0 = self.cost_classes[compare[0]]
        class1 = self.cost_classes[compare[1]]
        # test_edge = find_edge(self.graph, self.graph.edge_index, 44)[0]
        for w in vary:
            self.weight.a = cost0 * w + cost1 * (1 - w)
            # print("test weight", self.weight[test_edge])
            path, path_costs = self.get_shortest_path(source, dest)
            # print(
            #     class0, "weight:", w, class1, "weight:", 1 - w, "costs:",
            #     np.mean(path_costs, axis=0)
            # )
            pareto.append(np.sum(path_costs, axis=0))
            paths.append(path)

        if out_path is not None:
            pareto = np.asarray(pareto)
            pareto0 = pareto[:, compare[0]]
            pareto1 = pareto[:, compare[1]]
            plot_pareto(
                pareto0,
                pareto1,
                paths,
                vary, [class0, class1],
                out_path=out_path
            )
        return paths

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
