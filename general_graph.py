import numpy as np
from graph_tool.all import Graph, shortest_path, load_graph
import time
import networkx as nx
from power_planner.utils import get_half_donut


class GeneralGraph():

    def __init__(self, graphtool=1, directed=True, verbose=1):
        if graphtool:
            self.graph = Graph(directed=directed)
            self.weight = self.graph.new_edge_property("float")
        else:
            self.graph = nx.Graph()
        self.time_logs = {}
        self.verbose = verbose
        self.graphtool = graphtool

    def set_shift(self, lower, upper, vec, max_angle):
        self.shifts = get_half_donut(lower, upper, vec, angle_max=max_angle)

    def add_nodes(self, nodes):
        """
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

        if self.verbose:
            print("time for shortest path", time.time() - tic)

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        return vertices_path

    def save_graph(self, OUT_PATH):
        self.graph.edge_properties["weight"] = self.weight
        self.graph.save(OUT_PATH + ".xml.gz")

    def load_graph(self, IN_PATH):
        self.g_prev = load_graph(IN_PATH + ".xml.gz")
        self.weight_prev = self.g_prev.ep.weight
        # weight = G2.ep.weight[G2.edge(66, 69)]
