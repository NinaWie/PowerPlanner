# import matplotlib.pyplot as plt
import numpy as np
import time
import os
import datetime
import json

from data_reader import DataReader
# read_in_tifs, get_hard_constraints, get_cost_surface
from utils import reduce_instance, normalize, get_half_donut, plot_path, get_shift_transformed

from weighted_graph import WeightedGraph

# define paths:
PATH_FILES = "/Users/ninawiedemann/Downloads/tif_ras_buf"
HARD_CONS_PATH = "hard_constraints"
CORR_PATH = "corridor/Corridor_BE.tif"

OUT_PATH = "outputs/path_result" + str(datetime.datetime.utcnow())

# define hyperparameters:
RASTER = 10
PYLON_DIST_MIN = 150
PYLON_DIST_MAX = 250

DOWNSCALE = False
SCALE_PARAM = 5

PYLON_DIST_MIN /= RASTER
PYLON_DIST_MAX /= RASTER
if DOWNSCALE:
    PYLON_DIST_MIN /= SCALE_PARAM
    PYLON_DIST_MAX /= SCALE_PARAM
print("defined pylon distances in raster:", PYLON_DIST_MIN, PYLON_DIST_MAX)

# READ IN FILES
data = DataReader(PATH_FILES)
## read all tif files in main folder and sum up:
# tifs, files = data.read_in_tifs(PATH_FILES)
# instance = np.sum(tifs, axis=0)
instance_corr = data.get_hard_constraints(CORR_PATH, HARD_CONS_PATH)
instance = data.get_cost_surface("corridor/COSTSURFACE.tif")
print("shape of instance", instance.shape)

# scale down to simplify
if DOWNSCALE:
    print("Image downscaled by ", SCALE_PARAM)
    instance = reduce_instance(instance, SCALE_PARAM)
    instance_corr = reduce_instance(instance_corr, SCALE_PARAM)

instance_norm = normalize(instance)

donut_tuples = get_half_donut(PYLON_DIST_MIN, PYLON_DIST_MAX)

# Define graph
graph = WeightedGraph(instance_norm, instance_corr)
graph.add_nodes()

# old version: graph.add_edges_old(donut_tuples)
shift_tuples = get_shift_transformed(donut_tuples)
graph.add_edges(donut_tuples, shift_tuples)

# Compute path
SOURCE_IND = 0
TARGET_IND = graph.n_vertices - 1
path = graph.shortest_path(SOURCE_IND, TARGET_IND)

# plot the result
plot_path(instance_norm, path, out_path=OUT_PATH + ".png")

# save the path as a json:
with open(OUT_PATH + ".json", "w") as outfile:
    json.dump(path, outfile)
