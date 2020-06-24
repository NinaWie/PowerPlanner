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
ID = "ch_"  # str(round(time.time() / 60))[-5:]
OUT_DIR = os.path.join("..", "outputs")

SCALE_PARAM = 2  # args.scale
SCENARIO = 1
INST = "ch"
if args.cluster:
    height_resistance_path = "data/dtm_10m.tif"
else:
    height_resistance_path = "../data/Instance_CH.nosync/dtm_10m.tif"
PIPELINE = [(1, 0)]
USE_KSP = 0
MIN_H = 40

GRAPH_TYPE = graphs.HeightGraph
# LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph, ImplicitLG
# ImplicitLgKSP, WeightedKSP
print("graph type:", GRAPH_TYPE)
# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"
PATH_FILES = os.path.join("data")
IOPATH = os.path.join(PATH_FILES, f"{INST}_dump_w{SCENARIO}_{SCALE_PARAM}.dat")

# LOAD CONFIG
with open(os.path.join(PATH_FILES, f"{INST}_config.json"), "r") as infile:
    cfg_dict = json.load(infile)  # Config(SCALE_PARAM)
    cfg = SimpleNamespace(**cfg_dict)
    cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX = compute_pylon_dists(
        cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX, cfg.RASTER, SCALE_PARAM
    )

# READ DATA
with open(IOPATH, "rb") as infile:
    data = pickle.load(infile)
    (instance, instance_corr, start_inds, dest_inds) = data.data

COMPARISONS = []
for a_w in [0.1, 0.2, 0.4]:
    for e_w in [0, 0.3, 0.5]:
        for h_w in [0, 0.2, 1]:
            for b_w in [0, 1 / 3, 2 / 3]:
                for p_w in [0, 1 / 3, 2 / 3]:
                    if p_w + b_w > 1 or p_w + b_w == 0:
                        continue
                    COMPARISONS.append([a_w, e_w, h_w, b_w, p_w])
print("Number comparisons", len(COMPARISONS))
shortcut = ["a", "e", "h", "b", "p"]
# for angle_weight in
for COMP in COMPARISONS:
    (a_w, e_w, h_w, b_w, p_w) = COMP
    u_w = 1 - p_w - b_w
    ID_list = [
        shortcut[i] + str(round(COMP[i] * 10)) for i in range(len(COMP))
    ]
    ID_list.append("u" + str(round(u_w * 10)))
    ID = "ch_" + ("_").join(ID_list)
    # f"{ID}_a{angle_weight}_e{edge_weight}_h{height_weight}_b{round(
    # bau_weight*10, 0)}_p{round(planung_weight, 1)}_u
    # {round(umwelt_weight,1)}"
    OUT_PATH = os.path.join(OUT_DIR, ID)
    print("-------------- ", ID, "-------------")
    CLASS_WEIGHTS = [b_w, p_w, u_w]

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
            CLASS_WEIGHTS,
            angle_weight=a_w,
            cab_forb=cfg.CABLE_FORBIDDEN
        )
        # add vertices
        graph.add_nodes()
        if height_resistance_path is not None:
            graph.init_heights(height_resistance_path, MIN_H, 80, SCALE_PARAM)
        print("1.2) set shift, edge costs and added nodes")
        graph.add_edges(edge_weight=e_w, height_weight=h_w)
        print("2) added edges", graph.n_edges)
        print("number of vertices:", graph.n_nodes)

        # weighted sum of all costs
        graph.sum_costs()
        source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
        print("3) summed cost, get source and dest")
        # get actual best path
        path, path_costs, cost_sum = graph.get_shortest_path(
            source_v, target_v
        )
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

    time_pipeline = round(time.time() - tic, 3)
    print("FINISHED PIPELINE:", time_pipeline)
    print("path length", len(path))
    # SAVE timing test
    time_test_csv(
        ID, cfg.CSV_TIMES, SCALE_PARAM * cfg.RASTER, cfg.GTNX, GRAPH_TYPE,
        graph, path_costs, cost_sum, dist, time_pipeline, NOTES
    )

    # -------------  PLOTTING: ----------------------

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

    # save just coordinates of path
    raw_costs, names = graph.raw_path_costs(path)
    data.save_original_path(OUT_PATH, [path], [raw_costs], names)
