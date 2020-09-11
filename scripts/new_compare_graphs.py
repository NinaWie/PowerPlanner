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
    # get unnormalized costs
    angle_cost = round(np.sum(CostUtils.compute_angle_costs(path)), 2)
    n_categories = len(cfg.class_weights)
    path_costs = np.asarray(path_costs)
    # get normalization weights
    ang_weight_norm = cfg.angle_weight * np.sum(cfg.class_weights)
    all_cost_weights = np.array([ang_weight_norm] + list(cfg.class_weights))
    all_cost_weights = all_cost_weights / np.sum(all_cost_weights)
    print([ang_weight_norm] + list(cfg.class_weights), all_cost_weights)
    # compute normalized path costs
    summed_costs = np.around(np.sum(path_costs[:, -n_categories:], axis=0), 2)
    weighted_sum = round(np.dot(summed_costs, all_cost_weights[1:]), 2)
    together = all_cost_weights[0] * angle_cost + weighted_sum
    n_pixels = np.sum(instance_corr > 0)

    # csv_header = ["ID", "instance", "resolution", "graph", "number pixels"
    # "space edges", "overall time",
    # "time vertex adding", "time edge adding",  "time shortest path",
    # "angle cost", "category costs", "sum of costs"]
    logs = [
        ID, INST, SCALE_PARAM * 10, n_pixels, graphtype, graph.n_nodes,
        graph.n_edges, time_pipeline, graph.time_logs["add_nodes"],
        graph.time_logs["add_all_edges"], graph.time_logs["shortest_path"],
        cfg.angle_weight, angle_cost, summed_costs, weighted_sum, together,
        mean_eucl, max_eucl
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

# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"

# define IO paths
PATH_FILES = "data"

# Iterate overall graphs
graph_names = ["Normal graph", "Implicit line graph", "Line graph"]
for INST in ["belgium", "de", "ch"]:
    for g, GRAPH_TYPE in enumerate(
        [graphs.WeightedGraph, graphs.ImplicitLG, graphs.LineGraph]
    ):
        for SCALE_PARAM in [5, 2, 1]:  # TODO
            print("")
            print("---------------------------------------------------")
            print("---------------", INST, SCALE_PARAM, "-------------")
            IOPATH = os.path.join(
                PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat"
            )

            # LOAD DATA
            with open(IOPATH, "rb") as infile:
                data = pickle.load(infile)
                (instance, edge_cost, instance_corr, config) = data
                cfg = config.graph
                start_inds = config.graph.start_inds
                dest_inds = config.graph.dest_inds

            instance_vertices = len(np.where(instance_corr > 0)[0])

            OUT_PATH_orig = OUT_PATH

            # for a, angle_weight in enumerate([0.2 ]):  # TODO
            # print("PROCESS ", graph_names[g], angle_weight)
            cfg.edge_weight = 0
            # cfg.angle_weight = angle_weight
            cfg.csv_times = "../outputs/graph_compare.csv"
            # ID
            graphtype = graph_names[g]
            ID = f"{graph_names[g]}_{SCALE_PARAM}_{INST}_{int(cfg.angle_weight*100)}"
            OUT_PATH = os.path.join(OUT_DIR, ID + ".csv")

            graph_gt = GRAPH_TYPE(instance, instance_corr, verbose=0)
            graph_gt.set_shift(cfg.start_inds, cfg.dest_inds, **vars(cfg))

            estimated_edges = instance_vertices * len(graph_gt.shift_tuples)
            print("will have ", estimated_edges, "edges")
            # ABORT if too many edges
            if estimated_edges > 1000000000:
                print("ABORT bc of memory!")
                logs = [
                    ID, INST, SCALE_PARAM * 10, instance_vertices, graphtype,
                    instance_vertices, estimated_edges
                ]
                with open(cfg.csv_times, 'a+', newline='') as write_obj:
                    # Create a writer object from csv module
                    csv_writer = writer(write_obj)
                    # Add contents of list as last row in the csv file
                    csv_writer.writerow(logs)
                break
            tic = time.time()
            path_gt, path_costs_gt, cost_sum_gt = graph_gt.single_sp(
                **vars(cfg)
            )

            print("vertices:", graph_gt.n_nodes, "edges", graph_gt.n_edges)

            time_pipeline = round(time.time() - tic, 3)
            print("DONE SP")

            if g == 0:
                path_bl = path_gt

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

            # if g == 0 and a == 0:
            # angle weight only needs to be varied for the implicit lg
            #    break
