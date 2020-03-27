# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from power_planner.data_reader import DataReader
from power_planner.plotting import plot_path_costs, plot_pipeline_paths
from power_planner.utils import time_test_csv, get_distance_surface
from weighted_graph import WeightedGraph
from line_graph import LineGraph, LineGraphFromGraph
from weighted_reduced_graph import ReducedGraph
import numpy as np
import time
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
parser.add_argument('scale', help="how much to downsample", type=int)
args = parser.parse_args()

if args.cluster:
    PATH_FILES = os.path.join("..", "data")
else:
    PATH_FILES = "/Users/ninawiedemann/Downloads/tifs_new"

# define paths:
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

SCALE_PARAM = args.scale
PIPELINE = [(4, 80), (2, 50), (1, 0)]  # [(1, 0)]  # (8, 100), (4, 80),

GRAPH_TYPE = "NORM"

NOTES = "parcels"

LOAD = 1
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
        data = pickle.load(infile)
        (instance, instance_corr, start_inds, dest_inds) = data.data
else:
    # read in files
    data = DataReader(PATH_FILES, CORR_PATH, WEIGHT_CSV, SCENARIO, SCALE_PARAM)
    instance, instance_corr, start_inds, dest_inds = data.get_data(
        START_PATH, DEST_PATH, emergency_dist=PYLON_DIST_MAX
    )

    if SAVE_PICKLE:
        data.data = (instance, instance_corr, start_inds, dest_inds)
        with open(IOPATH, "wb") as outfile:
            pickle.dump(data, outfile)
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
else:
    raise NotImplementedError

# BUILD GRAPH:
graph.set_edge_costs(data.layer_classes, data.class_weights)
graph.set_shift(PYLON_DIST_MIN, PYLON_DIST_MAX, vec, MAX_ANGLE)
# add vertices
graph.add_nodes()

# START PIPELINE
tic = time.time()
corridor = np.ones(instance_corr.shape)  # beginning: everything is included
output_paths = []
plot_surfaces = []
time_infos = []

for (factor, dist) in PIPELINE:
    print("----------- PIPELINE", factor, dist, "---------------")
    graph.set_cost_rest(factor, corridor, start_inds, dest_inds)
    print(
        "1) set cost rest, nonzero:",
        np.sum(np.mean(graph.cost_rest, axis=0) > 0)
    )
    graph.add_edges()
    print("2) added edges", len(list(graph.graph.edges())))
    print("number of vertices:", len(list(graph.graph.vertices())))

    # weighted sum of all costs
    graph.sum_costs()
    source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
    print("3) summed cost, get source and dest")
    # get actual best path
    path, path_costs = graph.get_shortest_path(source_v, target_v)
    print("4) shortest path")
    # save for inspection
    output_paths.append((path, path_costs))
    plot_surfaces.append(graph.cost_rest[2])  # TODO: mean makes black
    # get several paths --> here: pareto paths
    paths = [path]
    # graph.get_pareto(
    #     np.arange(0, 1.1, 0.1), source_v, target_v, compare=[2, 3]
    # )

    # PRINT AND SAVE timing test
    time_test_csv(
        ID, CSV_TIMES, SCALE_PARAM, GTNX, GRAPH_TYPE, graph, path_costs, dist,
        0, NOTES
    )
    time_infos.append(graph.time_logs)

    if VERBOSE:
        del graph.time_logs['edge_list_times']
        del graph.time_logs['add_edges_times']
        print(graph.time_logs)

    if dist > 0:
        # do specified numer of dilations
        dist_surface = get_distance_surface(
            graph.pos2node.shape, paths, mode="dilation", n_dilate=dist
        )
        print("5) compute distance surface")
        # remove the edges of vertices in the corridor (to overwrite)
        graph.remove_vertices(dist_surface, delete_padding=PYLON_DIST_MAX)
        print("6) remove edges")
        # set new corridor
        corridor = (dist_surface > 0).astype(int)

time_pipeline = round(time.time() - tic, 3)
print("FINISHED PIPELINE:", time_pipeline)

# SAVE timing test
time_test_csv(
    ID, CSV_TIMES, SCALE_PARAM, GTNX, GRAPH_TYPE, graph, path_costs, 0,
    time_pipeline, NOTES
)

# PLOT RESULT
plot_pipeline_paths(
    plot_surfaces, output_paths, buffer=2, out_path=OUT_PATH + "_pipeline.png"
)
# plot_path(instance * instance_corr, path, buffer=1, out_path=OUT_PATH+".png")
plot_path_costs(
    instance * instance_corr,
    path,
    path_costs,
    graph.cost_classes,
    buffer=2,
    out_path=OUT_PATH + ".png"
)

# SAVE graph
# graph.save_graph(OUT_PATH + "_graph")
# np.save(OUT_PATH + "_pos2node.npy", graph.pos2node)

# data.save_coordinates(path, OUT_PATH, scale_factor=SCALE_PARAM)
# DataReader.save_json(OUT_PATH, path, path_costs, graph.time_logs,SCALE_PARAM)
DataReader.save_pipeline_infos(
    OUT_PATH, output_paths, time_infos, PIPELINE, SCALE_PARAM
)

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

# PIPELINE PREV

# # add edges in corridor
# corridor = np.ones(instance_corr.shape)
# graph.set_cost_rest(1, corridor, start_inds, dest_inds)
# graph.add_edges()
# # weighted sum of all costs
# graph.sum_costs()

# # SHORTEST PATH
# # # Alternative: with list of start and end nodes:
# # source, target = graph.add_start_end_vertices()
# # path = graph.shortest_path(source, target)
# source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
# print("start and end:", source_v, target_v)
# path, path_costs = graph.get_shortest_path(source_v, target_v)
# # PARETO FRONTEIR
# _ = graph.get_pareto(
#     np.arange(0, 1.1, 0.1),
#     source_v,
#     target_v,
#     out_path=OUT_PATH,
#     compare=[2, 3]
# )
