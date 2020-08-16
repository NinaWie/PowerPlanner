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


def logging(dat_fp, csv_fp, ID):
    # MAX_EDGES = 5000000
    with open(dat_fp, "rb") as outfile:
        (output, max_nr_edges, times_pipeline, correct) = pickle.load(outfile)

    edge_factor = np.mean(max_nr_edges) / edges_gt
    print("Reducing the number of edges by a factor of ", edge_factor)
    print("Using time ", np.mean(times_pipeline), "compared to", time_gt)
    percent_correct = np.around(
        np.sum(np.array(correct).astype(int)) / len(correct), 2
    )
    print("Percentage correct:", percent_correct)

    unique_path_set = []
    unique_costs = []
    paths_computed = [np.array(o[0]) for o in output]
    for (new_path, _, cost) in output:
        already = [
            len(path) == len(new_path) and np.all(path == new_path)
            for path in unique_path_set
        ]
        if not np.any(already):
            unique_path_set.append(new_path)
            unique_costs.append(cost)
    print("Number of unique produced paths:", len(unique_path_set))
    intersection_w_gt = 1 - np.mean(
        [
            np.around(KspUtils.path_distance(path_gt, p2), 2)
            for p2 in paths_computed
        ]
    )
    print("Intersections with ground truth", intersection_w_gt)
    eucl_w_gt = np.mean(
        [
            np.around(KspUtils.path_distance(path_gt, p2, "eucl_mean"), 2)
            for p2 in paths_computed
        ]
    )
    print("Mean eucledian distance of paths and ground truth", eucl_w_gt)
    print("New costs", np.mean(unique_costs), "vs cost gt:", cost_sum_gt)

    res_list = np.around(
        np.array(
            [
                np.mean(times_pipeline), edge_factor, percent_correct,
                np.mean(unique_costs),
                np.std(unique_costs), intersection_w_gt, eucl_w_gt
            ]
        ), 2
    ).tolist()
    list_of_elem = [ID, INST] + res_list
    with open(csv_fp, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


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
    (belgium_inst, belgium_edge_inst, belgium_inst_corr, belgium_config) = data
    cfg = belgium_config.graph
    start_inds = belgium_config.graph.start_inds
    dest_inds = belgium_config.graph.dest_inds

# GROUND TRUTH PATH:
graph = graphs.WeightedGraph(belgium_inst, belgium_inst_corr, verbose=False)
tic = time.time()
path_gt, path_costs_gt, cost_sum_gt = graph.single_sp(
    **vars(belgium_config.graph)
)
edges_gt = graph.n_edges
path_groundtruth = np.array(path_gt)
print("number of edges", edges_gt)
time_gt = time.time() - tic

MAX_EDGES = 500000
D1 = 100
D2 = 50

random = True

RAND_PIPES = []
for MAX_EDGES in [4000000, 5000000, 6000000]:
    for D1 in [100, 150, 200]:
        for D2 in [0, 50, 75]:
            RAND_PIPES.append(
                [(MAX_EDGES, D1), (MAX_EDGES, D2), (MAX_EDGES, 0)]
            )

NORM_PIPES = []
for sample_factor in [3, 4]:
    for D1 in [40, 60, 80]:
        for D2 in [0, 15, 30, 40]:
            if D2 == 0:
                second_factor = 1
            else:
                second_factor = 2
            NORM_PIPES.append(
                [(sample_factor, D1), (second_factor, D2), (1, 0)]
            )

PIPELINES = NORM_PIPES + RAND_PIPES
randomness = [0 for _ in range(len(NORM_PIPES))
              ] + [1 for _ in range(len(RAND_PIPES))]

# PIPE = [(MAX_EDGES, D1), (MAX_EDGES, D2), (MAX_EDGES, 0)]
for PIPE, random in zip(PIPELINES, randomness):
    print(" ------------------- NEW PIPELINE -------------------")
    # PIPE = [(4,100), (2,25), (1,0)] did not work, and for 50 instead of 50 there were nans
    print(PIPE)
    max_nr_edges = []
    times_pipeline = []
    correct = []
    output = []

    if random:
        nr_iters = 3
        graphclass = graphs.RandomWeightedGraph
    else:
        nr_iters = 1
        graphclass = graphs.WeightedGraph

    # COMPUTE STATISTICS
    for _ in range(nr_iters):

        graph = graphclass(belgium_inst, belgium_inst_corr, verbose=False)
        # set shift necessary in case of random graph automatic
        # probability estimation by edge bound
        graph.set_shift(cfg.start_inds, cfg.dest_inds, **vars(cfg))

        corridor = np.ones(belgium_inst_corr.shape) * 0.5

        edge_numbers = list()

        tic = time.time()

        for factor, dist in PIPE:
            graph.set_corridor(
                corridor,
                cfg.start_inds,
                cfg.dest_inds,
                factor_or_n_edges=factor
            )
            path_wg = []
            while len(path_wg) == 0:
                path_wg, path_costs_wg, cost_sum_wg = graph.single_sp(
                    **vars(cfg)
                )

            edge_numbers.append(graph.n_edges)

            if dist == 0:
                break
            corridor = get_distance_surface(
                graph.hard_constraints.shape, [path_wg],
                mode="dilation",
                n_dilate=dist
            )
            graph.remove_vertices(corridor)
            # plt.imshow(graph.corridor)
            # plt.colorbar()
            # plt.show()

        time_pipeline = time.time() - tic

        times_pipeline.append(time_pipeline)
        max_nr_edges.append(np.max(edge_numbers))
        if len(path_wg) == len(path_gt):
            correct.append(np.all(np.array(path_wg) == np.asarray(path_gt)))
        else:
            correct.append(False)
        print("-----------", factor, dist, "correct:", correct)
        output.append([path_wg, path_costs_wg, cost_sum_wg])

        # print(
        #     np.max(edge_numbers), "equal gt?",
        #     np.all(np.array(path_wg) == np.asarray(path_gt))
        # )

    LEN_PIPE = len(PIPE)
    ID = str(PIPE)  # f"{INST[:2].upper()} {PIPE}"
    # f"random_results_{MAX_EDGES}_{LEN_PIPE}_{D1}_{D2}.dat"
    with open(os.path.join(OUT_DIR, ID), "wb") as outfile:
        pickle.dump((output, max_nr_edges, times_pipeline, correct), outfile)

    logging(
        os.path.join(OUT_DIR, ID), os.path.join(OUT_DIR, "random_results.csv"),
        ID
    )
