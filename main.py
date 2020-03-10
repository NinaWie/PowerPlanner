# import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from power_planner.data_reader import DataReader
from power_planner.utils import normalize, get_half_donut, plot_path
from weighted_graph import WeightedGraph

import matplotlib.pyplot as plt

import json

# define paths:
PATH_FILES = "/Users/ninawiedemann/Downloads/tifs_new"
HARD_CONS_PATH = "hard_constraints"
CORR_PATH = "corridor/Corridor_BE.tif"
COST_PATH = "COSTSURFACE.tif"
START_PATH = "start_point/Start"
DEST_PATH = "dest_point/Destination"

OUT_PATH = "outputs/path_" + str(round(time.time()))[-5:]

# define hyperparameters:
RASTER = 10
PYLON_DIST_MIN = 150
PYLON_DIST_MAX = 250
MAX_ANGLE = 0.5 * np.pi

VERBOSE = 1
GTNX = 1

SCALE_PARAM = 2

# compute pylon distances:
PYLON_DIST_MIN /= RASTER
PYLON_DIST_MAX /= RASTER
if SCALE_PARAM > 1:
    PYLON_DIST_MIN /= SCALE_PARAM
    PYLON_DIST_MAX /= SCALE_PARAM
print("defined pylon distances in raster:", PYLON_DIST_MIN, PYLON_DIST_MAX)

# read in files
data = DataReader(PATH_FILES, CORR_PATH, SCALE_PARAM)
# prev version: read all tif files in main folder and sum up:
# tifs, files = data.read_in_tifs(PATH_FILES)
# instance = np.sum(tifs, axis=0)
instance_corr = data.get_hard_constraints(HARD_CONS_PATH)

instance = data.get_corridor()  # data.get_cost_surface(COST_PATH)
start_inds = data.get_shape_point(START_PATH)
dest_inds = data.get_shape_point(DEST_PATH)
print("shape of instance", instance.shape)
assert instance.shape == instance_corr.shape
print("start cells:", start_inds, "dest cells:", dest_inds)

# normalize instance values (0-1)
instance_norm = normalize(instance)

# get donut around each cell
vec = dest_inds - start_inds
print("start-dest-vec", vec)
donut_tuples = get_half_donut(
    PYLON_DIST_MIN, PYLON_DIST_MAX, vec, angle_max=MAX_ANGLE
)
# TESTING:
# print("len donut tuples", donut_tuples)
# img_size = int(PYLON_DIST_MAX) + 1
# ar = np.zeros((2 * img_size, 2 * img_size))
# for tup in donut_tuples:
#     ar[tup[0] + img_size, tup[1] + img_size] = 1
# plt.imshow(ar)
# plt.savefig("test.png")

# Define graph
graph = WeightedGraph(
    instance_norm, instance_corr, graphtool=GTNX, verbose=VERBOSE
)
graph.add_nodes()
# old version: graph.add_edges_old(donut_tuples)
graph.add_edges(donut_tuples)

# compute shortest path
source = graph.cells_to_vertices(start_inds)
target = graph.cells_to_vertices(dest_inds)
path = graph.shortest_path(source, target)
path.insert(0, start_inds)
path.append(dest_inds)
# # Alternative: with list of start and end nodes:
# source, target = graph.add_start_end_vertices()
# path = graph.shortest_path(source, target)

# plot the result
plot_path(instance_norm, path, out_path=OUT_PATH + ".png")

# save the path as a json:
data.save_json(
    path, OUT_PATH, scale_factor=SCALE_PARAM, time_logs=graph.time_logs
)

if VERBOSE:
    del graph.time_logs['edge_list_times']
    del graph.time_logs['add_edges_times']
    print(graph.time_logs)
