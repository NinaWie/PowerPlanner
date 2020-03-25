from constraints import convolve, get_kernel, convolve_faster
from power_planner.utils import shift_surface, get_donut_vals
from weighted_graph import WeightedGraph

import numpy as np
from graph_tool.all import Graph, shortest_path
import time
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd

from skimage.segmentation import watershed
from skimage import data, util, filters, color


class ReducedGraph(WeightedGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        cluster_scale,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(ReducedGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )

        # cluster_scale = 2 --> half of the nodes

        self.cluster_scale = cluster_scale
        self.mode = "center"

        # overwrites pos2node
        self._watershed_transform(mode=self.mode)
        print("number of nodes watershed:", len(np.unique(self.pos2node)))
        self.node_pos = []

    def _get_seeds(self, greater_zero):
        lab = 0
        x_len, y_len = greater_zero.shape
        seeds = np.zeros(greater_zero.shape)
        omitted = 0
        for i in np.arange(0, x_len, self.cluster_scale):
            for j in np.arange(0, y_len, self.cluster_scale):
                if greater_zero[i, j]:
                    seeds[i, j] = lab
                    lab += 1
                else:
                    omitted += 1
        print("omitted:", omitted)
        return seeds

    def _watershed_transform(self, compact=0.01, mode="all"):
        """
        :param mode: all = all combinations in one cluster possible
                --> leading to larger distances
                center = only the center of each cluster can be connected
        """
        tic = time.time()
        img = np.mean(self.cost_rest, axis=0)

        greater_zero = (img > 0).astype(int)
        edges = filters.sobel(img)

        seeds = self._get_seeds(greater_zero)
        if self.verbose:
            print("number seeds: ", np.sum(seeds > 0))

        w1 = watershed(edges, seeds, compactness=compact)
        # w1 is full watershed --> labels spread over corridor borders
        # but: label 0 also included --> +1 before corridor
        w1_g_zero = (w1 + 1) * greater_zero
        # labels: 0 is forbidden, 1 etc is watershed labels
        labels = np.unique(w1_g_zero)
        self.pos2node = np.zeros(w1.shape).astype(int)
        self.nr_members = np.zeros(w1.shape).astype(int)
        for i, lab in enumerate(labels):
            inds = w1_g_zero == lab
            self.pos2node[inds] = i
            self.nr_members[inds] = np.sum(inds)

        self.pos2node[self.pos2node == 0] = -1

       if mode == "center":
            new_cost_rest = np.zeros(self.cost_rest.shape)
            for lab in np.unique(self.pos2node)[1:]:
                # , lab in enumerate(labels):
                x_inds, y_inds = np.where(self.pos2node == lab)
                for j in range(len(self.cost_rest)):
                    new_cost_rest[j,
                                  int(np.mean(x_inds)),
                                  int(np.mean(y_inds))] = np.mean(
                                      self.cost_rest[j, x_inds, y_inds]
                                  )
            self.cost_rest = new_cost_rest

        self.cost_rest *= self.hard_constraints
        self.time_logs["watershed"] = round(time.time() - tic, 3)

    def _split_to_shape(self, a, chunk_shape, start_axis=0):
        """
        compute splits of array in both dimensions - return patches
        """
        if len(chunk_shape) != len(a.shape):
            raise ValueError(
                'chunk length does not match array number of axes'
            )

        if start_axis == len(a.shape):
            return a

        num_sections = math.ceil(a.shape[start_axis] / chunk_shape[start_axis])
        # print(num_sections)
        split = numpy.array_split(a, num_sections, axis=start_axis)
        return [
            split_to_shape(split_a, chunk_shape, start_axis + 1)
            for split_a in split
        ]

    def get_shortest_path(self, source, target):
        tic = time.time()

        # compute shortest path
        vertices_path, _ = shortest_path(
            self.graph,
            source,
            target,
            weights=self.weight,
            negative_weights=True
        )
        # else:
        #     vertices_path = nx.dijkstra_path(self.graph, source, target)

        path_map = np.zeros(self.hard_constraints.shape)
        col = 1

        # transform path
        path = []
        out_costs = []
        for v in vertices_path:
            v_ind = self.graph.vertex_index[v]
            x_inds, y_inds = np.where(self.pos2node == v_ind)
            # find minimum value field out of possible
            if self.mode == "all":
                min_val = 1
                for (i, j) in zip(x_inds, y_inds):
                    val_cost = np.mean(self.cost_instance[:, i, j])
                    if val_cost < min_val:
                        min_val = val_cost
                        min_ind_x = i
                        min_ind_y = j
                    # current plotting
                    path_map[i, j] = col
                    col += 1
            elif self.mode == "center":
                min_ind_x = int(np.mean(x_inds))
                min_ind_y = int(np.mean(y_inds))

            path.append((min_ind_x, min_ind_y))
            out_costs.append(self.cost_rest[:, min_ind_x, min_ind_y].tolist())
        plt.imshow(path_map, origin="upper")
        plt.savefig("path_map.png")

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)

        return path, out_costs
