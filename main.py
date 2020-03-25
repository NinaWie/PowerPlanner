# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from power_planner.data_reader import DataReader
from power_planner.plotting import plot_path, plot_path_costs
from power_planner.utils import time_test_csv
from weighted_graph import WeightedGraph
from line_graph import LineGraph, LineGraphFromGraph
from weighted_reduced_graph import ReducedGraph
import numpy as np
import time
import os
import pickle

# define paths:
PATH_FILES = "/Users/ninawiedemann/Downloads/tifs_new"
HARD_CONS_PATH = "hard_constraints"
CORR_PATH = "corridor/Corridor_BE.tif"
COST_PATH = "COSTSURFACE.tif"
START_PATH = "start_point/Start"
DEST_PATH = "dest_point/Destination"
WEIGHT_CSV = "layer_weights.csv"
CSV_TIMES = "outputs/time_tests.csv"

ID = str(round(time.time() / 60))[-5:]
OUT_PATH = "outputs/path_" + ID

# define hyperparameters:
RASTER = 10
PYLON_DIST_MIN = 150
PYLON_DIST_MAX = 250
MAX_ANGLE = 0.5 * np.pi
SCENARIO = 1

VERBOSE = 1
GTNX = 1

SCALE_PARAM = 5
CLUSTER_SCALE = 2  # segmentation --> reducing number of nodes by this factor

GRAPH_TYPE = "REDUCED"

NOTES = "None"

LOAD = 0
SAVE_PICKLE = 0
IOPATH = os.path.join(PATH_FILES, "data_dump_" + str(SCALE_PARAM) + ".dat")

print("graph type:", GRAPH_TYPE)

# compute pylon distances:
PYLON_DIST_MIN /= RASTER
PYLON_DIST_MAX /= RASTER
if SCALE_PARAM > 1:
    PYLON_DIST_MIN /= SCALE_PARAM
    PYLON_DIST_MAX /= SCALE_PARAM
print("defined pylon distances in raster:", PYLON_DIST_MIN, PYLON_DIST_MAX)

# READ DATA
if LOAD:
    # load from pickle
    with open(IOPATH, "rb") as infile:
        (instance, instance_corr, start_inds, dest_inds) = pickle.load(infile)
else:
    # read in files
    data = DataReader(PATH_FILES, CORR_PATH, WEIGHT_CSV, SCENARIO, SCALE_PARAM)
    instance, instance_corr, start_inds, dest_inds = data.get_data(
        START_PATH, DEST_PATH, emergency_dist=PYLON_DIST_MAX
    )

    if SAVE_PICKLE:
        out_tuple = (instance, instance_corr, start_inds, dest_inds)
        with open(IOPATH, "wb") as outfile:
            pickle.dump(out_tuple, outfile)
        print("successfully saved data")

vec = dest_inds - start_inds
print("start-dest-vec", vec)

# DEFINE GRAPH AND ALGORITHM
if GRAPH_TYPE == "NORM":
    # Define normal weighted graph
    graph = WeightedGraph(
        instance, instance_corr, graphtool=GTNX, verbose=VERBOSE
    )

elif GRAPH_TYPE == "LINE":
    # Define LINE GRAPH
    graph = LineGraph(instance, instance_corr, graphtool=GTNX, verbose=VERBOSE)

elif GRAPH_TYPE == "LINE_FILE":
    # Load file and derive line graph
    graph_file = "outputs/path_02852_graph"
    graph = LineGraphFromGraph(
        graph_file, instance, instance_corr, graphtool=GTNX, verbose=VERBOSE
    )
elif GRAPH_TYPE == "REDUCED":
    graph = ReducedGraph(
        instance,
        instance_corr,
        CLUSTER_SCALE,
        graphtool=GTNX,
        verbose=VERBOSE
    )

# BUILD GRAPH:
graph.set_edge_costs(data.layer_classes, data.class_weights)
graph.set_shift(PYLON_DIST_MIN, PYLON_DIST_MAX, vec, MAX_ANGLE)
# add nodes and vertices
graph.add_nodes()
graph.add_edges()
# weighted sum of all costs
graph.sum_costs()

# SHORTEST PATH
# # Alternative: with list of start and end nodes:
# source, target = graph.add_start_end_vertices()
# path = graph.shortest_path(source, target)
source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
print("start and end:", source_v, target_v)
path, path_costs = graph.get_shortest_path(source_v, target_v)
# PARETO FRONTEIR
# _ = graph.get_pareto(
#     np.arange(0, 1.1, 0.1),
#     source_v,
#     target_v,
#     out_path=OUT_PATH,
#     compare=[2, 3]
# )

# PLOT RESULT
# plot_path(instance * instance_corr, path, buffer=1, out_path=OUT_PATH+".png")
plot_path_costs(
    instance * instance_corr,
    path,
    path_costs,
    graph.cost_classes,
    buffer=2,
    out_path=OUT_PATH + ".png"
)

# SAVE timing test
time_test_csv(
    ID, CSV_TIMES, SCALE_PARAM, GTNX, GRAPH_TYPE, graph, path_costs, NOTES
)

# SAVE graph
# graph.save_graph(OUT_PATH + "_graph")
np.save(OUT_PATH + "_pos2node.npy", graph.pos2node)

# data.save_coordinates(path, OUT_PATH, scale_factor=SCALE_PARAM)
DataReader.save_json(OUT_PATH, path, path_costs, graph.time_logs, SCALE_PARAM)

if VERBOSE:
    del graph.time_logs['edge_list_times']
    del graph.time_logs['add_edges_times']
    print(graph.time_logs)

#TESTING:

# print("len donut tuples", donut_tuples)
# img_size = int(PYLON_DIST_MAX) + 1
# ar = np.zeros((2 * img_size, 2 * img_size))
# for tup in donut_tuples:
#     ar[tup[0] + img_size, tup[1] + img_size] = 1
# plt.imshow(ar)
# plt.savefig("test.png")

# TESTING:
# a = np.zeros((20, 20))
# a[:5, 1] = 1
# a[4, 1:8] = 1
# a[4, 10] = 1
# a[7:15, 10] = 1
# a[14, 10] = 1
# a[14, 13:15] = 1
# a[14:, 14] = 1
# start_inds = np.asarray([4, 5])
# dest_inds = np.asarray([23, 18])
# instance = np.pad(a, ((4, 4), (4, 4)))
# instance_corr = instance
# PYLON_DIST_MIN = 1.5
# PYLON_DIST_MAX = 3