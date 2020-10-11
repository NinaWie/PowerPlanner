import argparse
import os
import pickle
import time
# import warnings
import numpy as np
import warnings
import matplotlib.pyplot as plt
import psutil

# utils imports
try:
    from power_planner.data_reader import DataReader
except ImportError:
    warnings.warn("DATA READER CANNOT BE USED - IMPORTS")
from power_planner import graphs
from power_planner.ksp import KSP
from power_planner.alternative_paths import AlternativePaths
from power_planner.evaluate_path import save_path_cost_csv
from power_planner.plotting import (
    plot_path_costs, plot_pipeline_paths, plot_path, plot_k_sp,
    plot_pareto_paths
)
from power_planner.utils.utils import (
    get_distance_surface, time_test_csv, load_config
)
print(
    "memory start:",
    round(
        psutil.virtual_memory().percent / 100000000 *
        psutil.virtual_memory().available, 2
    )
)

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
parser.add_argument('-i', '--instance', type=str, default="test")
parser.add_argument('-s', '--scale', help="resolution", type=int, default=2)
args = parser.parse_args()

# define out save name
ID = "de_new_data_20_laplace_" + args.instance  # str(round(time.time() / 60))[-5:]
OUT_DIR = os.path.join("..", "outputs")
OUT_PATH = os.path.join(OUT_DIR, ID)

SCALE_PARAM = args.scale
SCENARIO = 1
INST = args.instance  # "belgium"
height_resistance_path = None  # "../data/Instance_CH.nosync/dtm_10m.tif"

# DEFINE CONFIGURATION
# normal graph pipeline
# PIPELINE = [(2, 30), (1, 0)]  # [(1, 0)]  # [(4, 80), (2, 50), (1, 0)]  #
# random graph pipeline
PIPELINE = [(1, 0)]
# PIPELINE = [(4, 200), (2, 50), (1, 0)]  # (2, 200),
# PIPELINE = [(0.8, 100), (0.5, 50), (0, 0)]  # nonauto random
# PIPELINE = [(5000000, 100), (5000000, 0)]  # auto pipeline
USE_KSP = 0

GRAPH_TYPE = graphs.ImplicitLG
# LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph, ImplicitLG
# ImplicitLgKSP, WeightedKSP
print("graph type:", GRAPH_TYPE)
# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"

PATH_FILES = os.path.join("data")
IOPATH = os.path.join(PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat")

with open(IOPATH, "rb") as infile:
    data = pickle.load(infile)
    try:
        (instance, edge_cost, instance_corr, config) = data
        cfg = config.graph
        start_inds = cfg.start_inds
        dest_inds = cfg.dest_inds
    except ValueError:
        warnings.warn("Edge weights not available - taking normal costs")
        (instance, instance_corr, start_inds, dest_inds) = data.data
        edge_cost = instance.copy()

# DEFINE GRAPH AND ALGORITHM
graph = GRAPH_TYPE(
    instance, instance_corr, edge_instance=edge_cost, verbose=cfg.verbose
)

# START PIPELINE
tic = time.time()
# path, path_costs, cost_sum = graph.single_sp(**vars(cfg))
path, path_costs, cost_sum = graph.sp_trees(**vars(cfg))

ksp = KSP(graph)
alternatives = ksp.laplace(
    5,
    thresh=70,
    cost_add=0.05,
)

# gives a single  value
print("edges in graph (in mio):", graph.n_edges / 1000000)

time_pipeline = round(time.time() - tic, 3)
print("FINISHED PIPELINE:", time_pipeline)
# print("path length", len(path))
# SAVE timing test
# time_test_csv(
#     ID, cfg.csv_times, SCALE_PARAM, 1, GRAPH_TYPE, graph,
#     [round(s, 3) for s in np.sum(path_costs, axis=0)], cost_sum, 0,
#     time_pipeline, NOTES
# )

# save the costs
save_path_cost_csv(
    OUT_PATH, [k[0] for k in alternatives], instance, **vars(cfg)
)

# KSP
plot_k_sp(alternatives, graph.instance, out_path=OUT_PATH)

# SIMPLE
plot_path(graph.instance, path, buffer=2, out_path=OUT_PATH + ".png")

# FOR COST COMPARISON
plot_path_costs(
    instance * instance_corr,
    path,
    path_costs,
    cfg.layer_classes,
    buffer=2,
    out_path=OUT_PATH + "_costs.png"
)

# -------------------------- LONG FLEXIBLE VERSION ---------------------

# corridor = np.ones(instance_corr.shape) * 0.5  # start with all
# output_paths = []
# plot_surfaces = []
# time_infos = []

# for (factor, dist) in PIPELINE:
#     print("----------- PIPELINE", factor, dist, "---------------")
#     graph.set_corridor(corridor, start_inds, dest_inds, factor_or_n_edges=1)
#     print(cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX)
#     graph.set_shift(
#         cfg.PYLON_DIST_MIN,
#         cfg.PYLON_DIST_MAX,
#         dest_inds - start_inds,
#         cfg.MAX_ANGLE,
#         max_angle_lg=cfg.MAX_ANGLE_LG
#     )
#     print("neighbors:", len(graph.shifts))
#     print("1) set shift and corridor")
#     graph.set_edge_costs(
#         cfg.layer_classes, cfg.class_weights, angle_weight=cfg.ANGLE_WEIGHT
#     )
#     # add vertices
#     graph.add_nodes()
#     if height_resistance_path is not None:
#         graph.init_heights(height_resistance_path, 60, 80, SCALE_PARAM)
#     print("1.2) set shift, edge costs and added nodes")
#     graph.add_edges(edge_weight=cfg.EDGE_WEIGHT, height_weight=0)
#     print("2) added edges", graph.n_edges)
#     print("number of vertices:", graph.n_nodes)

#     # weighted sum of all costs
#     graph.sum_costs()
#     source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
#     print("3) summed cost, get source and dest")
#     # get actual best path
#     path, path_costs, cost_sum = graph.get_shortest_path(source_v, target_v)
#     print("4) shortest path", cost_sum)
#     # save for inspection
#     output_paths.append((path, path_costs))
#     plot_surfaces.append(graph.instance.copy())
#     # get several paths --> possible to replace by pareto_out[0]
#     # paths = [path]
#     time_infos.append(graph.time_logs.copy())

#     if cfg.verbose:
#         graph.time_logs.pop('edge_list_times', None)
#         graph.time_logs.pop('add_edges_times', None)
#         print(graph.time_logs)

#     if dist > 0:
#         # PRINT AND SAVE timing test
#         time_test_csv(
#             ID, cfg.csv_times, SCALE_PARAM, GRAPH_TYPE, graph, path_costs,
#             cost_sum, dist, 0, NOTES
#         )
#         # Define paths around which to place corridor
#         if USE_KSP:
#             graph.get_shortest_path_tree(source_v, target_v)
#             ksp = graph.find_ksp(source_v, target_v, 3, overlap=0.2)
#             paths = [k[0] for k in ksp]
#             flat_list = [item for sublist in paths for item in sublist]
#             del output_paths[-1]
#             output_paths.append((flat_list, path_costs))
#             plot_k_sp(
#                 ksp,
#                 graph.instance * (corridor > 0).astype(int),
#                 out_path=OUT_PATH + str(factor)
#             )
#         else:
#             paths = [path]

#         # do specified numer of dilations
#         corridor = get_distance_surface(
#             graph.hard_constraints.shape,
#             paths,
#             mode="dilation",
#             n_dilate=dist
#         )
#         print("5) compute distance surface")
#         # remove the edges of vertices in the corridor (to overwrite)
#         graph.remove_vertices(corridor, delete_padding=cfg.PYLON_DIST_MAX)
#         print("6) remove edges")

# necessary for ALL further computations:
# graph.get_shortest_path_tree(source_v, target_v)

# BEST IN WINDOW
# path_window, path_window_cost, cost_sum_window = graph.best_in_window(
#     30, 35, 60, 70, source_v, target_v
# )
# path_window, path_window_cost, cost_sum_window = graph.path_through_window(
#     50, 100, 200, 300
# )
# alt = AlternativePaths(graph)
# path_window, path_window_cost, cost_sum_window = alt.replace_window(
#     150, 200, 250, 300
# )
# print("cost actually", cost_sum, "cost window", cost_sum_window)
# plot_path_costs(
#     instance * instance_corr,
#     path_window,
#     path_window_cost,
#     data.layer_classes,
#     buffer=2,
#     out_path=OUT_PATH + "_window.png"
# )

# COMPUTE KSP
#
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

# -------------  PLOTTING: ----------------------

# FOR PIPELINE
# plot_pipeline_paths(
#     plot_surfaces, output_paths, buffer=2, out_path=OUT_PATH + "_pipeline.png"
# )
# FOR KSP:
# with open(OUT_PATH + "_ksp.json", "w") as outfile:
#     json.dump(ksp, outfile)
# plot_k_sp(ksp, graph.instance * (corridor > 0).astype(int), out_path=OUT_PATH)

# FOR WINDOW
# plot_path(
#     graph.instance, path_window, buffer=0, out_path=OUT_PATH + "_window.png"
# )

# -------------  SAVE INFOS: ----------------------

# SAVE graph
# graph.save_graph(OUT_PATH + "_graph")
# np.save(OUT_PATH + "_pos2node.npy", graph.pos2node)

# SAVE JSON WITH INFOS
# DataReader.save_pipeline_infos(
#     OUT_PATH, output_paths, time_infos, PIPELINE, SCALE_PARAM
# )

# with open(
#     os.path.join(PATH_FILES, f"{INST}_data_{SCENARIO}_1.dat"), "rb"
# ) as infile:
#     (big_inst, _, _, _) = pickle.load(infile)
#     print(big_inst.shape)
# big_inst = np.sum(
#     np.moveaxis(big_inst, 0, -1) * np.asarray(cfg.class_weights), axis=2
# )
# print(big_inst.shape)

# save just coordinates of path
#  [k[0] for k in ksp_out]
# data.save_original_path(OUT_PATH, [k[0] for k in ksp], output_coords=True)

# LINE GRAPH FROM FILE:
# elif GRAPH_TYPE == "LINE_FILE":
#     # Load file and derive line graph
#     graph_file = "outputs/path_02852_graph"
#     graph = LineGraphFromGraph(
#         graph_file, instance, instance_corr, graphtool=GTNX, verbose=verbose
#     )
