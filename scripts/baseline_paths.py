import argparse
import os
import pickle
import time
# import warnings
import numpy as np
from csv import writer
import warnings
import matplotlib.pyplot as plt
# utils imports
from power_planner.utils.utils_ksp import KspUtils
from power_planner.utils.utils_costs import CostUtils
from power_planner.evaluate_path import save_path_cost_csv
from power_planner import graphs


def convert_instance(instance, instance_corr):
    tuned_inst_corr = np.ones(instance_corr.shape)
    for i in range(instance_corr.shape[0]):
        if not np.any(instance_corr[i, :]):
            tuned_inst_corr[i, :] = 0
    for i in range(instance_corr.shape[1]):
        if not np.any(instance_corr[:, i]):
            tuned_inst_corr[:, i] = 0
    # put high costs in the edge area
    tuned_inst = instance.copy()
    inverted = np.absolute(instance_corr - 1).astype("bool")
    tuned_inst[:, inverted] = 1
    return tuned_inst, tuned_inst_corr


def logging(ID, graph, path, path_costs, cfg, time_pipeline, comp_path=None):
    if comp_path is None:
        max_eucl = 0
        mean_eucl = 0
    else:
        # compute path distances and multiply with resolution to get meters
        max_eucl = (
            KspUtils.path_distance(path, comp_path, mode="eucl_max") *
            cfg.scale * 10
        )
        mean_eucl = (
            KspUtils.path_distance(path, comp_path, mode="eucl_mean") *
            cfg.scale * 10
        )
    # SAVE timing test
    angle_cost = round(np.sum(CostUtils.compute_angle_costs(path)), 2)
    n_categories = len(cfg.class_weights)
    path_costs = np.asarray(path_costs)
    summed_costs = np.around(np.sum(path_costs[:, -n_categories:], axis=0), 2)
    weighted_sum = round(np.dot(summed_costs, cfg.class_weights), 2)
    n_pixels = np.sum(instance_corr > 0)

    # csv_header = ["ID", "instance", "resolution", "graph", "number pixels"
    # "space edges", "overall time",
    # "time vertex adding", "time edge adding",  "time shortest path",
    # "angle cost", "category costs", "sum of costs"]
    logs = [
        ID, INST, SCALE_PARAM * 10, n_pixels, graphtype, graph.n_nodes,
        graph.n_edges, time_pipeline, graph.time_logs["add_nodes"],
        graph.time_logs["add_all_edges"], graph.time_logs["shortest_path"],
        cfg.angle_weight, angle_cost, summed_costs, weighted_sum, mean_eucl,
        max_eucl
    ]
    with open(cfg.csv_times, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(logs)


parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
parser.add_argument('-i', '--instance', type=str, default="ch")
parser.add_argument('-s', '--scale', help="resolution", type=int, default=1)
args = parser.parse_args()

# define out save name
ID = "results_" + args.instance  # str(round(time.time() / 60))[-5:]
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

# define IO paths
PATH_FILES = "data"
IOPATH = os.path.join(PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat")

# LOAD DATA
with open(IOPATH, "rb") as infile:
    data = pickle.load(infile)
    (instance, edge_cost, instance_corr, config) = data
    cfg = config.graph
    start_inds = config.graph.start_inds
    dest_inds = config.graph.dest_inds

OUT_PATH_orig = OUT_PATH

orig_pylon_dist_min = cfg.pylon_dist_min
orig_pylon_dist_max = cfg.pylon_dist_max

graph_names = ["Implicit line graph", "Normal graph"]
for g, GRAPH_TYPE in enumerate([graphs.ImplicitLG, graphs.WeightedGraph]):
    for a, angle_weight in enumerate([0.05, 0.1, 0.3]):
        cfg.edge_weight = 0
        cfg.angle_weight = angle_weight

        # ID
        graphtype = graph_names[g]
        ID = f"_{graph_names[g]}_{SCALE_PARAM}_{INST}_{int(angle_weight*100)}"
        OUT_PATH = OUT_PATH_orig + ID

        # if cable is not forbidden, we need to design helper corridor and inst
        if not config.data.cable_forbidden:
            tuned_inst, tuned_inst_corr = convert_instance(
                instance, instance_corr
            )
        else:
            tuned_inst_corr = instance_corr.copy()
            tuned_inst = instance.copy()

        # run first time in RASTER mode
        # (edge inst doesn't matter cause edge weight must be zero here)
        graph_bl = GRAPH_TYPE(tuned_inst, tuned_inst_corr, verbose=True)
        cfg.pylon_dist_min = 1
        cfg.pylon_dist_max = 1.5
        tic_raster = time.time()
        path_raster, _, _ = graph_bl.single_sp(**vars(cfg))
        toc_raster = time.time() - tic_raster

        # construct corridor from path
        pylon_spotting_corr = np.zeros(instance_corr.shape)
        for (i, j) in path_raster:
            if instance_corr[i, j] < np.inf:
                pylon_spotting_corr[i, j] = 1

        # Pylon spotting
        graph_pylon_spotting = GRAPH_TYPE(
            instance,
            pylon_spotting_corr,
            edge_instance=tuned_inst,
            verbose=cfg.verbose
        )
        cfg.pylon_dist_min = orig_pylon_dist_min
        cfg.pylon_dist_max = orig_pylon_dist_max
        path_bl, path_costs_bl, cost_sum_bl = graph_pylon_spotting.single_sp(
            **vars(cfg)
        )
        logging(
            "baseline" + ID, graph_bl, path_bl, path_costs_bl, cfg, toc_raster
        )
        save_path_cost_csv(
            OUT_PATH + "_baseline", [path_bl], instance, **vars(cfg)
        )

        # GROUND TRUTH
        graph_gt = GRAPH_TYPE(
            instance,
            instance_corr,
            edge_instance=tuned_inst,
            verbose=cfg.verbose
        )
        tic = time.time()
        path_gt, path_costs_gt, cost_sum_gt = graph_gt.single_sp(**vars(cfg))

        time_pipeline = round(time.time() - tic, 3)
        print("FINISHED ")
        print("----------------------------")

        logging(
            ID,
            graph_gt,
            path_gt,
            path_costs_gt,
            cfg,
            time_pipeline,
            comp_path=path_bl
        )

        save_path_cost_csv(OUT_PATH, [path_gt], instance, **vars(cfg))

        plt.figure(figsize=(10, 10))
        plt.imshow(graph_gt.instance)
        path_bl = np.array(path_bl)
        path_gt = np.array(path_gt)
        plt.plot(path_bl[:, 1], path_bl[:, 0])
        plt.plot(path_gt[:, 1], path_gt[:, 0])
        plt.savefig(OUT_PATH + "_paths.png")

        if g == 1 and a == 0:
            # angle weight only needs to be varied for the implicit lg
            break
