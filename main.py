# import matplotlib.pyplot as plt
import numpy as np
import time
from graph_tool.all import Graph, shortest_path  # *
import os

from data_reader import read_in_tifs, get_hard_constraints
from utils import reduce_instance, normalize, get_half_donut, plot_path

# define paths:
PATH_FILES = "/Users/ninawiedemann/Downloads/tif_ras_buf"
HARD_CONS_PATH = os.path.join(PATH_FILES, "hard_constraints")
CORR_PATH = os.path.join(PATH_FILES, "corridor")

# define hyperparameters:
RASTER = 10
PYLON_DIST_MIN = 150
PYLON_DIST_MAX = 250

DOWNSCALE = True
SCALE_PARAM = 5

# get tifs and constraints
tifs, files = read_in_tifs(PATH_FILES)
instance_corr = get_hard_constraints(CORR_PATH, HARD_CONS_PATH)

# TODO: next part is not finalized
instance = np.sum(tifs, axis=0)
print("shappe of summed tifs", instance.shape)

# scale down to simplify
if DOWNSCALE:
    instance = reduce_instance(instance, SCALE_PARAM)
    instance_corr = reduce_instance(instance_corr, SCALE_PARAM)

instance_norm = normalize(instance)

x_len, y_len = instance_norm.shape

# node to pos mapping
node_pos = [
    (i, j) for i in range(x_len) for j in range(y_len) if instance_corr[i, j]
]
# pos to node mapping
pos_node = {node_pos[i]: i for i in range(len(node_pos))}

# ### Define edges
tic = time.time()
donut_tuples = get_half_donut(2.5, 5)
edge_list = []

for n, (i, j) in enumerate(node_pos):
    # n is the name of the node in the graph (=index), (i,j) the position
    weight_node = 1 - instance_norm[i, j]
    for (x, y) in donut_tuples:
        new_x = i + x
        new_y = j + y
        if new_x >= 0 and new_x < x_len and new_y >= 0 and new_y < y_len:
            if instance_corr[new_x, new_y]:  # inside corridor
                weight = 1 - instance_norm[new_x, new_y] + weight_node
                edge_list.append(
                    [n, pos_node[(new_x, new_y)],
                     round(weight, 3)]
                )
print("time to build edge list:", time.time() - tic)

# ### Add nodes and edges to graph
G = Graph(directed=False)
weight = G.new_edge_property("float")

# add nodes to graph
vlist = G.add_vertex(len(node_pos))
print("added nodes:", len(list(vlist)))

# add edges and properties to the graph
G.add_edge_list(edge_list, eprops=[weight])
print("added edges:", len(list(G.edges())))

# ### Compute shortest path
tic = (time.time())
SOURCE = 0
TARGET = len(node_pos) - 1
vertices_path, edges_path = shortest_path(
    G,
    G.vertex(SOURCE),
    G.vertex(TARGET),
    weights=weight,
    negative_weights=True
)
path = [node_pos[G.vertex_index[v]] for v in vertices_path]
print("time for shortest path", time.time() - tic)

# plot the result
plot_path(instance_norm, path)
