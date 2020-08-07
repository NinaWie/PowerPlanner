import argparse
import os
import pickle
import time
# import warnings
import numpy as np
import warnings

# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.evaluate_path import save_path_cost_csv
from power_planner.plotting import (
    plot_path_costs, plot_pipeline_paths, plot_path, plot_k_sp,
    plot_pareto_paths
)
from power_planner.utils.utils import (
    get_distance_surface, time_test_csv, load_config
)

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
parser.add_argument('-i', '--instance', type=str, default="ch")
parser.add_argument('-s', '--scale', help="resolution", type=int, default=1)
args = parser.parse_args()

# define out save name
ID = "power_analysis_angle_" + args.instance  # str(round(time.time() / 60))[-5:]
OUT_DIR = os.path.join("..", "outputs")
OUT_PATH = os.path.join(OUT_DIR, ID)

SCALE_PARAM = args.scale
SCENARIO = 1
INST = args.instance
height_resistance_path = None  # "../data/Instance_CH.nosync/dtm_10m.tif"
PIPELINE = [(1, 0)]
USE_KSP = 0

GRAPH_TYPE = graphs.ImplicitLG
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
    PATH_FILES = f"../data/instance_{INST}.nosync"
IOPATH = os.path.join(PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat")

# LOAD CONFIGURATION
if not LOAD:
    cfg = load_config(
        os.path.join(PATH_FILES, f"{INST}_config.json"),
        scale_factor=SCALE_PARAM
    )

# READ DATA
if LOAD:
    # load from pickle
    with open(IOPATH, "rb") as infile:
        data = pickle.load(infile)
        try:
            (instance, edge_cost, instance_corr, config) = data
            cfg = config.graph
            start_inds = config.graph.start_inds
            dest_inds = config.graph.dest_inds
        except ValueError:
            warnings.warn("Edge weights not available - taking normal costs")
            (instance, instance_corr, start_inds, dest_inds) = data.data
            edge_cost = instance.copy()
else:
    # read in files
    data = DataReader(PATH_FILES, SCENARIO, SCALE_PARAM, cfg)
    instance, edge_cost, instance_corr, config = data.get_data()
    # get graph processing specific cfg
    cfg = config.graph
    start_inds = cfg.start_inds
    dest_inds = cfg.dest_inds
    # save
    if SAVE_PICKLE:
        data_out = (instance, edge_cost, instance_corr, config)
        with open(IOPATH, "wb") as outfile:
            pickle.dump(data_out, outfile)
        print("successfully saved data")

print("INSTANCE", np.mean(instance), np.min(instance), np.max(instance))
save_path_costs = []

cfg.PYLON_DIST_MIN = 350 / (10 * SCALE_PARAM)  # RESOLUTION is 50
cfg.PYLON_DIST_MAX = 500 / (10 * SCALE_PARAM)
OUT_PATH_orig = OUT_PATH

COMPARISONS = []
for a_w in [0.3, 0.6]:
    for e_w in [0.2, 0.5]:
        for p_w in [1, 2]:
            COMPARISONS.append([a_w, e_w, p_w])
print("Number comparisons", len(COMPARISONS))
shortcut = ["a", "e", "p"]
# for angle_weight in
for COMP in COMPARISONS:
    (a_w, e_w, p_w) = COMP
    # u_w = 1 - p_w - b_w
    ID_list = [
        shortcut[i] + str(round(COMP[i] * 10)) for i in range(len(COMP))
    ]
    ID = "sensitivity_" + "_".join(ID_list)
    OUT_PATH = OUT_PATH_orig + ID
    cfg.ANGLE_WEIGHT = a_w
    cfg.EDGE_WEIGHT = e_w
    power = p_w
    # DEFINE GRAPH AND ALGORITHM
    graph = GRAPH_TYPE(
        instance, instance_corr, edge_instance=edge_cost, verbose=cfg.VERBOSE
    )
    tic = time.time()

    # PROCESS
    path, path_costs, cost_sum = graph.single_sp(power=power, **vars(cfg))

    # initialize normal graph:
    if power == 1:
        first_path = path
        graph_normal = graph
    _, actual_costs, actual_cost_sum = graph_normal.transform_path(path)
    # max_costs = np.max(actual_costs, axis=0)
    # mean_costs = np.mean(actual_costs, axis=0) * 10
    # print("maxima:", max_costs)
    # print("sum:", actual_cost_sum)
    # compute weighted costs: (1: because no angle costs considered)
    weighted_cost = np.dot(
        np.asarray(actual_costs)[:, 1:], graph.cost_weights[1:]
    )
    max_costs = round(np.max(weighted_cost), 3)
    mean_costs = round(np.mean(weighted_cost), 3)
    save_path_costs.append(weighted_cost)

    time_pipeline = round(time.time() - tic, 3)
    print("FINISHED :", time_pipeline)
    print("----------------------------")

    # SAVE timing test
    time_test_csv(
        ID, cfg.CSV_TIMES, SCALE_PARAM * 10, cfg.GTNX, "impl_lg_" + INST,
        graph, max_costs, actual_cost_sum, power, time_pipeline, mean_costs
    )

    # -------------  PLOTTING: ----------------------
    save_path_cost_csv(OUT_PATH, [path], instance, **vars(cfg))

    # SIMPLE
    plot_path(
        graph.instance,
        path,
        buffer=2,
        out_path=OUT_PATH + "_" + str(power) + ".png"
    )

with open(OUT_PATH + "_paths.dat", "wb") as outfile:
    pickle.dump(save_path_costs, outfile)
