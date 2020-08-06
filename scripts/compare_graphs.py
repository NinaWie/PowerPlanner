import argparse
import os
import pickle
import time
# import warnings
import numpy as np
from csv import writer
import warnings
# utils imports
from power_planner.utils.utils_costs import CostUtils
from power_planner import graphs

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
parser.add_argument('-i', '--instance', type=str, default="ch")
parser.add_argument('-s', '--scale', help="resolution", type=int, default=1)
args = parser.parse_args()

# define out save name
ID = "sensitivity_" + args.instance  # str(round(time.time() / 60))[-5:]
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

graph_names = ["Implicit line graph", "Normal graph", "Line graph"]
for angle_weight in [0.05, 0.1, 0.3]:
    for g, GRAPH_TYPE in enumerate(
        [graphs.ImplicitLG, graphs.WeightedGraph, graphs.LineGraph]
    ):
        cfg.edge_weight = 0
        cfg.angle_weight = angle_weight

        # ID
        graphtype = graph_names[g]
        ID = f"{graph_names[g]}_{SCALE_PARAM}_{INST}_{angle_weight}"
        OUT_PATH = OUT_PATH_orig + ID

        # DEFINE GRAPH AND ALGORITHM
        graph = GRAPH_TYPE(instance, instance_corr, verbose=cfg.VERBOSE)
        tic = time.time()

        # PROCESS
        path, path_costs, cost_sum = graph.single_sp(**vars(cfg))

        time_pipeline = round(time.time() - tic, 3)
        print("FINISHED :", graph)
        print("----------------------------")

        # SAVE timing test
        angle_cost = round(np.sum(CostUtils.compute_angle_costs(path)), 3)
        n_categories = len(cfg.class_weights)
        path_costs = np.asarray(path_costs)
        summed_costs = np.around(
            np.sum(path_costs[:, -n_categories:], axis=0), 3
        )
        weighted_sum = round(np.dot(summed_costs, cfg.class_weights), 3)
        n_pixels = np.sum(instance_corr > 0)

        # csv_header = ["ID", "instance", "resolution", "graph", "number pixels"
        # "space edges", "overall time",
        # "time vertex adding", "time edge adding",  "time shortest path",
        # "angle cost", "category costs", "sum of costs"]
        logs = [
            ID, INST, SCALE_PARAM * 10, n_pixels, graphtype, graph.n_nodes,
            graph.n_edges, time_pipeline, graph.time_logs["add_nodes"],
            graph.time_logs["add_all_edges"], graph.time_logs["shortest_path"],
            cfg.angle_weight, angle_cost, summed_costs, weighted_sum
        ]
        with open(cfg.csv_times, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(logs)

        # -------------  PLOTTING: ----------------------
        # graph.save_path_cost_csv(OUT_PATH, [path], **vars(cfg))
