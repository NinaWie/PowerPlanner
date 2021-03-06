import argparse
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.plotting import plot_prior_paths
from power_planner.utils.utils import get_distance_surface, time_test_csv
from power_planner.utils.utils_corridor import CorridorUtils
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
# parser.add_argument('scale', help="downsample", type=int, default=1)
args = parser.parse_args()

if args.cluster:
    PATH_FILES = os.path.join("..", "data")
else:
    PATH_FILES = "../data"

# DEFINE CONFIGURATION
ID = "randpipe_2_nina_corrs2"  # str(round(time.time() / 60))[-5:]

OUT_PATH = "outputs/path_" + ID
SCALE_PARAM = 2  # args.scale
# normal graph pipeline
# PIPELINE = [(2, 50), (1, 0)]  # [(1, 0)]  # [(4, 80), (2, 50), (1, 0)]  #
# random graph pipeline
PIPELINE = [(0.8, 150), (0.5, 0)]  # [(0.9, 40), (0, 0)]

GRAPH_TYPE = graphs.RandomWeightedGraph
# LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph
print("graph type:", GRAPH_TYPE)
# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"

LOAD = 1
SAVE_PICKLE = 0
IOPATH = os.path.join(PATH_FILES, "data_dump_" + str(SCALE_PARAM) + ".dat")

cfg = Config(SCALE_PARAM)

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
        cfg.START_PATH, cfg.DEST_PATH, emergency_dist=cfg.PYLON_DIST_MAX
    )

    if SAVE_PICKLE:
        data.data = (instance, instance_corr, start_inds, dest_inds)
        with open(IOPATH, "wb") as outfile:
            pickle.dump(data, outfile)
        print("successfully saved data")

vec = dest_inds - start_inds
print("start-dest-vec", vec)

# DEFINE GRAPH AND ALGORITHM
graph = GRAPH_TYPE(
    instance, instance_corr, graphtool=cfg.GTNX, verbose=cfg.VERBOSE
)

# BUILD GRAPH:
graph.set_edge_costs(data.layer_classes, data.class_weights)
graph.set_shift(cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX, vec, cfg.MAX_ANGLE)
# add vertices
graph.add_nodes()

# start with all
# corrs = [np.ones(instance_corr.shape) * 0.5]
corrs = CorridorUtils.generate_corridors_middle_line(
    instance_corr, start_inds, dest_inds, n_dilate=150
)
# corrs = CorridorUtils.generate_corridors_sample_path(
#     instance, start_inds, dest_inds, n_dilate=50, factor=40
# )
# corrs = CorridorUtils.generate_corridors_from_file(
#     os.path.join(PATH_FILES, cfg.CORR_PATH), scale_param=SCALE_PARAM
# )

corr_plot_surfaces = []
corr_plot_paths = []
for corr_ind, corridor in enumerate(corrs):
    output_paths = []
    plot_surfaces = []
    time_infos = []
    # START PIPELINE
    tic = time.time()
    for (factor, dist) in PIPELINE:
        print("----------- PIPELINE", factor, dist, "---------------")
        graph.set_corridor(factor, corridor, start_inds, dest_inds)
        print("1) set cost rest")
        graph.add_edges()
        print("2) added edges", len(list(graph.graph.edges())))
        print("number of vertices:", len(list(graph.graph.vertices())))

        # weighted sum of all costs
        graph.sum_costs()
        source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
        print("3) summed cost, get source and dest")
        # get actual best path
        path, path_costs, cost_sum = graph.get_shortest_path(
            source_v, target_v
        )
        print("4) shortest path")
        # save for inspection
        output_paths.append((path, path_costs))
        plot_surfaces.append(
            graph.cost_rest[2].copy()
        )  # TODO: mean makes black
        # get several paths --> here: pareto paths
        paths = [path]
        # graph.get_pareto(
        #     np.arange(0, 1.1, 0.1), source_v, target_v, compare=[2, 3]
        # )

        time_infos.append(graph.time_logs.copy())

        if cfg.VERBOSE:
            del graph.time_logs['edge_list_times']
            del graph.time_logs['add_edges_times']
            print(graph.time_logs)

        if dist > 0:
            # do specified numer of dilations
            corridor = get_distance_surface(
                graph.pos2node.shape, paths, mode="dilation", n_dilate=dist
            )
            print("5) compute distance surface")
            # remove the edges of vertices in the corridor (to overwrite)
            graph.remove_vertices(corridor, delete_padding=cfg.PYLON_DIST_MAX)
            print("6) remove edges")

    time_pipeline = round(time.time() - tic, 3)
    print("FINISHED PIPELINE:", time_pipeline)

    # SAVE timing test
    time_test_csv(
        ID, cfg.CSV_TIMES, SCALE_PARAM, cfg.GTNX, GRAPH_TYPE, graph,
        path_costs, cost_sum, str(PIPELINE), time_pipeline, NOTES
    )
    # save information for plotting
    corr_plot_surfaces.append(plot_surfaces)
    corr_plot_paths.append([o for (o, _) in output_paths])

    # clear graph for next prior
    graph.remove_vertices(corridor, delete_padding=cfg.PYLON_DIST_MAX)

plot_prior_paths(
    corr_plot_surfaces,
    corr_plot_paths,
    buffer=2,
    out_path=OUT_PATH + "_corr_pipeline.png"
)
# PLOT RESULT
# plot_pipeline_paths(
#     plot_surfaces,
#     output_paths,
#     buffer=2,
#     out_path=OUT_PATH + str(corr_ind) + "_pipeline.png"
# )
