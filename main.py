import argparse
import os
import pickle
import time
# import warnings
import numpy as np
import json
from types import SimpleNamespace
import matplotlib.pyplot as plt

# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.plotting import (
    plot_path_costs, plot_pipeline_paths, plot_path, plot_k_sp,
    plot_pareto_paths
)
from power_planner.utils.utils import (
    get_distance_surface, time_test_csv, compute_pylon_dists
)

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
# parser.add_argument('scale', help="downsample", type=int, default=1)
args = parser.parse_args()

# define out save name
ID = "de_inst_2_forb_uberspannen"  # str(round(time.time() / 60))[-5:]
OUT_DIR = os.path.join("..", "outputs")
OUT_PATH = os.path.join(OUT_DIR, ID)

# DEFINE CONFIGURATION
SCALE_PARAM = 2  # args.scale
# normal graph pipeline
# PIPELINE = [(2, 30), (1, 0)]  # [(1, 0)]  # [(4, 80), (2, 50), (1, 0)]  #
# random graph pipeline
PIPELINE = [(1, 0)]
# PIPELINE = [(4, 200), (2, 50), (1, 0)]  # (2, 200),
# PIPELINE = [(0.8, 100), (0.5, 50), (0, 0)]  # nonauto random
# PIPELINE = [(5000000, 100), (5000000, 0)]  # auto pipeline
USE_KSP = 0

GRAPH_TYPE = graphs.ImplicitKSP
# LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph, ImplicitLG
# ImplicitLgKSP, WeightedKSP
print("graph type:", GRAPH_TYPE)
# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"

LOAD = 1
if args.cluster:
    LOAD = 1
SAVE_PICKLE = 1

# define IO paths
if LOAD:
    PATH_FILES = os.path.join("data")
else:
    # PATH_FILES = "/Volumes/Nina Backup/data_master_thesis/large_instance"
    PATH_FILES = "../data/instance_DE.nosync"  #  "../data/Instance_CH.nosync" "../data/belgium.nosync"
IOPATH = os.path.join(PATH_FILES, "de_dump_" + str(SCALE_PARAM) + ".dat")

# LOAD CONFIG
with open("config.json", "r") as infile:
    cfg_dict = json.load(infile)  # Config(SCALE_PARAM)
    cfg = SimpleNamespace(**cfg_dict)
    cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX = compute_pylon_dists(
        cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX, cfg.RASTER, SCALE_PARAM
    )

# READ DATA
if LOAD:
    # load from pickle
    with open(IOPATH, "rb") as infile:
        data = pickle.load(infile)
        (instance, instance_corr, start_inds, dest_inds) = data.data
else:
    # read in files
    data = DataReader(
        PATH_FILES, cfg.CORR_PATH, cfg.WEIGHT_CSV, cfg.SCENARIO, SCALE_PARAM
    )
    instance, instance_corr, start_inds, dest_inds = data.get_data(
        cfg.START_PATH,
        cfg.DEST_PATH,
        percent_padding=None,  # cfg.PERC_PAD,
        emergency_dist=None,  # cfg.PYLON_DIST_MAX
        oneclass=True
    )

    if SAVE_PICKLE:
        data.data = (instance, instance_corr, start_inds, dest_inds)
        with open(IOPATH, "wb") as outfile:
            pickle.dump(data, outfile)
        print("successfully saved data")

# DEFINE GRAPH AND ALGORITHM
graph = GRAPH_TYPE(
    instance, instance_corr, graphtool=cfg.GTNX, verbose=cfg.VERBOSE
)

# START PIPELINE
tic = time.time()
corridor = np.ones(instance_corr.shape) * 0.5  # start with all
output_paths = []
plot_surfaces = []
time_infos = []

for (factor, dist) in PIPELINE:
    print("----------- PIPELINE", factor, dist, "---------------")
    graph.set_shift(
        cfg.PYLON_DIST_MIN,
        cfg.PYLON_DIST_MAX,
        dest_inds - start_inds,
        cfg.MAX_ANGLE,
        max_angle_lg=cfg.MAX_ANGLE_LG
    )
    graph.set_corridor(
        corridor, start_inds, dest_inds, factor_or_n_edges=factor
    )
    print("1) set shift and corridor")
    graph.set_edge_costs(
        data.layer_classes,
        data.class_weights,
        angle_weight=cfg.ANGLE_WEIGHT,
        cab_forb=cfg.CABLE_FORBIDDEN
    )
    # add vertices
    graph.add_nodes()
    print("1.2) set shift, edge costs and added nodes")
    graph.add_edges(edge_weight=cfg.EDGE_WEIGHT)
    print("2) added edges", graph.n_edges)
    print("number of vertices:", graph.n_nodes)

    # weighted sum of all costs
    graph.sum_costs()
    source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
    print("3) summed cost, get source and dest")
    # get actual best path
    path, path_costs, cost_sum = graph.get_shortest_path(source_v, target_v)
    print("4) shortest path", cost_sum)
    # save for inspection
    output_paths.append((path, path_costs))
    plot_surfaces.append(graph.instance.copy())
    # get several paths --> possible to replace by pareto_out[0]
    # paths = [path]
    time_infos.append(graph.time_logs.copy())

    if cfg.VERBOSE:
        graph.time_logs.pop('edge_list_times', None)
        graph.time_logs.pop('add_edges_times', None)
        print(graph.time_logs)

    if dist > 0:
        # PRINT AND SAVE timing test
        time_test_csv(
            ID, cfg.CSV_TIMES, SCALE_PARAM, cfg.GTNX, GRAPH_TYPE, graph,
            path_costs, cost_sum, dist, 0, NOTES
        )
        # Define paths around which to place corridor
        if USE_KSP:
            graph.get_shortest_path_tree(source_v, target_v)
            ksp = graph.k_shortest_paths(source_v, target_v, 3, overlap=0.2)
            paths = [k[0] for k in ksp]
            flat_list = [item for sublist in paths for item in sublist]
            del output_paths[-1]
            output_paths.append((flat_list, path_costs))
            plot_k_sp(
                ksp,
                graph.instance * (corridor > 0).astype(int),
                out_path=OUT_PATH + str(factor)
            )
        else:
            paths = [path]

        # do specified numer of dilations
        corridor = get_distance_surface(
            graph.hard_constraints.shape,
            paths,
            mode="dilation",
            n_dilate=dist
        )
        print("5) compute distance surface")
        # remove the edges of vertices in the corridor (to overwrite)
        graph.remove_vertices(corridor, delete_padding=cfg.PYLON_DIST_MAX)
        print("6) remove edges")

# BEST IN WINDOW
# path_window, path_window_cost, cost_sum_window = graph.best_in_window(
#     30, 35, 60, 70, source_v, target_v
# )
# print("cost actually", cost_sum, "cost_new", cost_sum_window)

# COMPUTE KSP
graph.get_shortest_path_tree(source_v, target_v)
ksp = graph.laplace(source_v, target_v, cfg.KSP, radius=20, cost_add=0.01)
plot_k_sp(ksp, graph.instance * (corridor > 0).astype(int), out_path=OUT_PATH)
# ksp = graph.k_diverse_paths(
#     source_v,
#     target_v,
#     cfg.KSP,
#     cost_thresh=1.01,
#     dist_mode="eucl_mean",
#     count_thresh=5
# )

# PARETO
# pareto_out = graph.get_pareto(
#     10,
#     source_v,
#     target_v,
#     compare=[0, 2, 3],
#     non_compare_weight=0.2,
#     out_path=OUT_PATH
# )
# plot_pareto_paths(pareto_out, graph.instance, out_path=OUT_PATH)

time_pipeline = round(time.time() - tic, 3)
print("FINISHED PIPELINE:", time_pipeline)
print("path length", len(path))
# SAVE timing test
time_test_csv(
    ID, cfg.CSV_TIMES, SCALE_PARAM * cfg.RASTER, cfg.GTNX, GRAPH_TYPE, graph,
    path_costs, cost_sum, dist, time_pipeline, NOTES
)

# -------------  PLOTTING: ----------------------

# FOR PIPELINE
plot_pipeline_paths(
    plot_surfaces, output_paths, buffer=2, out_path=OUT_PATH + "_pipeline.png"
)
# FOR KSP:
# with open(OUT_PATH + "_ksp.json", "w") as outfile:
#     json.dump(ksp, outfile)
# plot_k_sp(ksp, graph.instance * (corridor > 0).astype(int), out_path=OUT_PATH)

# FOR WINDOW
# plot_path(
#     graph.instance, path_window, buffer=0, out_path=OUT_PATH + "_window.png"
# )
# SIMPLE
# plot_path(graph.instance, path, buffer=0, out_path=OUT_PATH + ".png")

# FOR COST COMPARISON
plot_path_costs(
    instance * instance_corr,
    path,
    path_costs,
    data.layer_classes,
    buffer=2,
    out_path=OUT_PATH + "_costs.png"
)

# -------------  SAVE INFOS: ----------------------

# SAVE graph
# graph.save_graph(OUT_PATH + "_graph")
# np.save(OUT_PATH + "_pos2node.npy", graph.pos2node)

# SAVE JSON WITH INFOS
# DataReader.save_pipeline_infos(
#     OUT_PATH, output_paths, time_infos, PIPELINE, SCALE_PARAM
# )

# save just coordinates of path
# data.save_original_path(OUT_PATH, [path])
data.save_original_path(OUT_PATH, [k[0] for k in ksp], output_coords=True)

# LINE GRAPH FROM FILE:
# elif GRAPH_TYPE == "LINE_FILE":
#     # Load file and derive line graph
#     graph_file = "outputs/path_02852_graph"
#     graph = LineGraphFromGraph(
#         graph_file, instance, instance_corr, graphtool=GTNX, verbose=VERBOSE
#     )
