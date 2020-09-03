import argparse
import os
import pickle
import time
# import warnings
import numpy as np
from power_planner.utils.utils import get_distance_surface
from csv import writer
import warnings
import matplotlib.pyplot as plt
# utils imports
from power_planner.utils.utils_ksp import KspUtils
from power_planner.utils.utils_costs import CostUtils
from power_planner.evaluate_path import save_path_cost_csv
from power_planner import graphs


def logging(
    ID, graph, path, path_costs, cfg, N_EDGES, time_pipeline, comp_path=None
):
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
    n_pixels = np.sum(belgium_inst_corr > 0)

    # csv_header = ["ID", "instance", "resolution", "graph", "number pixels"
    # "space edges", "overall time",
    # "time vertex adding", "time edge adding",  "time shortest path",
    # "angle cost", "category costs", "sum of costs"]
    logs = [
        ID, INST, SCALE_PARAM * 10, n_pixels, graphtype, graph.n_nodes,
        N_EDGES, time_pipeline, graph.time_logs["add_nodes"],
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
# ID = "results_" + args.instance  # str(round(time.time() / 60))[-5:]
OUT_DIR = os.path.join("..", "outputs")

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

# PIPE = [(MAX_EDGES, D1), (MAX_EDGES, D2), (MAX_EDGES, 0)]
PIPELINES = [[1], [2, 1], [4, 2, 1], [3, 1]]
print("PIPELINES:", PIPELINES)

mult_factor = 13
random = 0
graph_names = ["Normal graph", "Implicit line graph", "Line graph"]

for INST, SCALE_PARAM in zip(["belgium", "de", "ch"], [1, 2, 2]):
    print("")
    print("---------------------------------------------------")
    print(INST, SCALE_PARAM)
    # LOAD DATA
    IOPATH = os.path.join(
        PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat"
    )
    with open(IOPATH, "rb") as infile:
        data = pickle.load(infile)
        (
            belgium_inst, belgium_edge_inst, belgium_inst_corr, belgium_config
        ) = data
        cfg = belgium_config.graph
        start_inds = belgium_config.graph.start_inds
        dest_inds = belgium_config.graph.dest_inds
    # iterate over pipelines
    ground_truth_paths = [[], []]
    for pipe_kind, PIPE in enumerate(PIPELINES):
        ID = str(PIPE)
        print("------------- NEW PIPELINE ", PIPE, "-----------------------")

        for g, GRAPH in enumerate([graphs.WeightedGraph, graphs.ImplicitLG]):
            print("")
            print(GRAPH)
            print("")

            graphtype = graph_names[g]

            graph = GRAPH(belgium_inst, belgium_inst_corr, verbose=False)
            corridor = np.ones(belgium_inst_corr.shape) * 0.5

            tic = time.time()
            actual_pipe = []
            edge_numbers = []

            for pipe_step, factor in enumerate(PIPE):
                if random:
                    factor = 1 - (1 / factor**2)
                graph.set_corridor(
                    corridor,
                    cfg.start_inds,
                    cfg.dest_inds,
                    factor_or_n_edges=factor,
                    sample_method="simple"
                )
                # main path computation
                path_gt, path_costs_gt, cost_sum_wg = graph.single_sp(
                    **vars(cfg)
                )

                edge_numbers.append(graph.n_edges)

                if factor == 1 or factor == 0:
                    actual_pipe.append((1, 0))
                    break
                corridor = get_distance_surface(
                    graph.hard_constraints.shape,
                    [path_gt],
                    mode="dilation",
                    n_dilate=10  # dist
                )
                # estimated edges are pixels times neighbors
                # divided by resolution squared
                estimated_edges_10 = len(np.where(corridor > 0)[0]) * len(
                    graph.shifts
                ) / ((PIPE[pipe_step + 1])**2)
                now_dist = (mult_factor * graph.n_edges) / estimated_edges_10
                # print("reduce corridor:", dist)
                corridor = get_distance_surface(
                    graph.hard_constraints.shape, [path_gt],
                    mode="dilation",
                    n_dilate=int(np.ceil(now_dist))
                )
                # print(
                #     "estimated with distance ", int(np.ceil(now_dist)),
                #     len(np.where(corridor > 0)[0]) * len(graph.shifts) /
                #     ((PIPE[pipe_step + 1])**2)
                # )
                actual_pipe.append([factor, int(np.ceil(now_dist))])
                graph.remove_vertices(corridor)

            time_pipeline = time.time() - tic
            print("OVERALL TIME:", time_pipeline)

            nr_edges = np.max(edge_numbers)

            if pipe_kind == 0:
                ground_truth_paths[g] = path_gt
            path_bl = ground_truth_paths[g]

            logging(
                ID,
                graph,
                path_gt,
                path_costs_gt,
                cfg,
                nr_edges,
                time_pipeline,
                comp_path=path_bl
            )
