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

USE_PRIOR = False

# PIPE = [(MAX_EDGES, D1), (MAX_EDGES, D2), (MAX_EDGES, 0)]
PIPELINES = []
for factor1 in [2, 3, 4, 5]:
    for factor2 in [1, 2]:
        if factor1 <= factor2:
            continue
        if factor2 == 1:
            PIPELINES.append([factor1, factor2])
        else:
            PIPELINES.append([factor1, factor2, 1])
print("PIPELINES:", PIPELINES)

useprior_meaning = [
    ["simple downsampling", "watershed"], ["no prior", "prior"]
]
edge_remain_dict = {
    1: [1],
    2: [0.35, 0.45, 0.5, 0.6],
    3: [0.16, 0.19],
    4: [0.1],
    5: [0.05]
}

for PIPE in PIPELINES:
    for EDGE_REMAIN in edge_remain_dict[PIPE[0]]:
        max_remaining_edges = edges_gt * EDGE_REMAIN
        print("------------- NEW PIPELINE -----------------------------")
        print(PIPE, "max remaining", max_remaining_edges)
        for random in [0, 1]:
            for USE_PRIOR in [0, 1]:
                print(" ------------- new random or prior -------------")
                print(
                    "random:", (["deterministic", "random"])[int(random)],
                    useprior_meaning[random][USE_PRIOR], "at most",
                    max_remaining_edges
                )
                # ID = f"{INST[:2].upper()}_{PIPE}"
                # if os.path.exists(os.path.join(OUT_DIR, ID)):
                #     print("already there")
                #     continue
                max_nr_edges = []
                times_pipeline = []
                correct = []
                output = []

                if random:
                    mult_factor = 10
                    nr_iters = 1  # TODO
                    graphclass = graphs.RandomWeightedGraph
                else:
                    mult_factor = 13
                    nr_iters = 1
                    graphclass = graphs.WeightedGraph

                # COMPUTE STATISTICS
                for iteri in range(nr_iters):
                    actual_pipe = []

                    graph = graphclass(
                        belgium_inst, belgium_inst_corr, verbose=False
                    )
                    # set shift necessary in case of random graph automatic
                    # probability estimation by edge bound
                    graph.set_shift(cfg.start_inds, cfg.dest_inds, **vars(cfg))

                    if random and USE_PRIOR:
                        corridor = get_distance_surface(
                            belgium_inst_corr.shape,
                            [[cfg.start_inds, cfg.dest_inds]],
                            mode="dilation",
                            n_dilate=200
                        )
                    else:
                        corridor = np.ones(belgium_inst_corr.shape) * 0.5

                    if USE_PRIOR:
                        DOWNSAMPLE_METHOD = "watershed"
                    else:
                        DOWNSAMPLE_METHOD = "simple"

                    edge_numbers = list()

                    tic = time.time()

                    for pipe_step, factor in enumerate(PIPE):
                        if random:
                            factor = 1 - (1 / factor**2)
                        graph.set_corridor(
                            corridor,
                            cfg.start_inds,
                            cfg.dest_inds,
                            factor_or_n_edges=factor,
                            mode="squared",
                            sample_method=DOWNSAMPLE_METHOD
                        )
                        path_wg = []
                        while len(path_wg) == 0:
                            path_wg, path_costs_wg, cost_sum_wg = graph.single_sp(
                                **vars(cfg)
                            )

                        edge_numbers.append(graph.n_edges)

                        if factor == 1 or factor == 0:
                            actual_pipe.append((1, 0))
                            break
                        corridor = get_distance_surface(
                            graph.hard_constraints.shape,
                            [path_wg],
                            mode="dilation",
                            n_dilate=10  # dist
                        )
                        # estimated edges are pixels times neighbors
                        # divided by resolution squared
                        estimated_edges_10 = len(
                            np.where(corridor > 0)[0]
                        ) * len(graph.shifts) / ((PIPE[pipe_step + 1])**2)
                        now_dist = (
                            mult_factor * max_remaining_edges
                        ) / estimated_edges_10
                        # print("reduce corridor:", dist)
                        corridor = get_distance_surface(
                            graph.hard_constraints.shape, [path_wg],
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

                    if iteri == 0:
                        print("edge numbers:", edge_numbers)
                    times_pipeline.append(time_pipeline)
                    max_nr_edges.append(np.max(edge_numbers))
                    if len(path_wg) == len(path_gt):
                        correct.append(
                            np.all(np.array(path_wg) == np.asarray(path_gt))
                        )
                    else:
                        correct.append(False)

                    output.append([path_wg, path_costs_wg, cost_sum_wg])

                LEN_PIPE = len(PIPE)
                ID = f"{INST[:2].upper()}_{actual_pipe}"

                # ID = str(PIPE)  # f"{INST[:2].upper()} {PIPE}"
                # f"random_results_{MAX_EDGES}_{LEN_PIPE}_{D1}_{D2}.dat"
                with open(os.path.join(OUT_DIR, ID), "wb") as outfile:
                    pickle.dump(
                        (output, max_nr_edges, times_pipeline, correct),
                        outfile
                    )

                logging(
                    os.path.join(OUT_DIR, ID),
                    os.path.join(OUT_DIR, "random_results.csv"), ID
                )
