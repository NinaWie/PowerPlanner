# import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

from power_planner.data_reader import DataReader
from power_planner.utils import reduce_instance, normalize, get_half_donut, plot_path
from weighted_graph import WeightedGraph

import json

# define paths:
PATH_FILES = "/Users/ninawiedemann/Downloads/tif_ras_buf"
HARD_CONS_PATH = "hard_constraints"
CORR_PATH = "corridor/Corridor_BE.tif"
COST_PATH = "corridor/COSTSURFACE.tif"

OUT_PATH = "outputs/path_" + str(round(time.time()))[-5:]

# define hyperparameters:
RASTER = 10
PYLON_DIST_MIN = 150
PYLON_DIST_MAX = 250

VERBOSE = 1

SCALE_PARAM = 5

PYLON_DIST_MIN /= RASTER
PYLON_DIST_MAX /= RASTER
if SCALE_PARAM > 1:
    PYLON_DIST_MIN /= SCALE_PARAM
    PYLON_DIST_MAX /= SCALE_PARAM
print("defined pylon distances in raster:", PYLON_DIST_MIN, PYLON_DIST_MAX)

# READ IN FILES
data = DataReader(PATH_FILES, CORR_PATH)
# prev version: read all tif files in main folder and sum up:
# tifs, files = data.read_in_tifs(PATH_FILES)
# instance = np.sum(tifs, axis=0)
instance_corr = data.get_hard_constraints(HARD_CONS_PATH)
instance = data.get_corridor()  # data.get_cost_surface(COST_PATH)
print("shape of instance", instance.shape)

# scale down to simplify
if SCALE_PARAM > 1:
    print("Image downscaled by ", SCALE_PARAM)
    instance = reduce_instance(instance, SCALE_PARAM)
    instance_corr = reduce_instance(instance_corr, SCALE_PARAM)

instance_norm = normalize(instance)

donut_tuples = get_half_donut(PYLON_DIST_MIN, PYLON_DIST_MAX)

# Define graph
graph = WeightedGraph(instance_norm, instance_corr, verbose=VERBOSE)
graph.add_nodes()

# old version: graph.add_edges_old(donut_tuples)
graph.add_edges(donut_tuples)

# Compute path
# SOURCE_IND = 0
# TARGET_IND = graph.n_vertices - 1
source, target = graph.add_start_end_vertices()  # give lists as parameters
path = graph.shortest_path(source, target)

# plot the result
plot_path(instance_norm, path, out_path=OUT_PATH + ".png")

# save the path as a json:
data.save_json(path, OUT_PATH, SCALE_PARAM)
