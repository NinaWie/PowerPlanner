import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.plotting import plot_path_costs, plot_path, plot_k_sp
from power_planner.utils.utils import get_distance_surface, time_test_csv
from power_planner.utils.utils_split import SplitUtils
from power_planner.graphs.split_graphs import SplitLG, SplitWeighted
from config import Config

PATH_FILES = "data"  # os.path.join("..", "data")

# DEFINE CONFIGURATION
ID = "split_path_lg_5"

OUT_PATH = os.path.join("..", "outputs", ID)
SCALE_PARAM = 5  # args.scale
PIPELINE = [(1, 0)]  # [(0.9, 40), (0, 0)]

GRAPH_TYPE = graphs.ImplicitLG
SPLIT_TYPE = SplitLG
# LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph, PowerBF
# TwoPowerBF, WeightedKSP
print("graph type:", GRAPH_TYPE)
# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"

IOPATH = os.path.join(PATH_FILES, "data_dump_" + str(SCALE_PARAM) + ".dat")

cfg = Config(SCALE_PARAM)

# READ DATA
with open(IOPATH, "rb") as infile:
    data = pickle.load(infile)
    (instance, instance_corr, start_inds, dest_inds) = data.data

# SWAPAXES (to avoid problem of along which axis the instance was split)
split_axis = np.argmax(np.absolute(start_inds - dest_inds))
if split_axis == 1:
    instance = np.swapaxes(instance, 2, 1)
    instance_corr = np.swapaxes(instance_corr, 1, 0)
    start_inds = np.flip(start_inds)
    dest_inds = np.flip(dest_inds)

# SPLIT
pix_per_part = int(instance.shape[1] / 2)
margin = int(cfg.PYLON_DIST_MAX)
padding = 20
two_insts, two_corrs = SplitUtils.construct_patches(
    instance, instance_corr, pix_per_part, margin, padding
)

print(
    "check shapes:", two_insts[0].shape, instance.shape, instance_corr.shape,
    two_corrs[0].shape, two_corrs[1].shape
)

# SET START AND DEST
deleted_part = pix_per_part - margin - padding
# if start is in first part and dest in second
if start_inds[0] < dest_inds[0]:
    start_points = [start_inds, dest_inds - [deleted_part, 0]]
# if dest is in first part and start is in second one
else:
    start_points = [dest_inds, start_inds - [deleted_part, 0]]
print("start and dest", deleted_part, start_points)
# make sure we got the correct point
assert two_insts[1][2, start_points[1][0], start_points[1][1]
                    ] == instance[2, dest_inds[0], dest_inds[1]]

# CONSTRUCT GRAPHS

vec = start_points[1] - start_points[0]  # start to dest vector
two_graphs = [None, None]
# do all steps for both seperately
for i in range(2):
    graph = GRAPH_TYPE(
        two_insts[i], two_corrs[i], graphtool=cfg.GTNX, verbose=cfg.VERBOSE
    )

    graph.set_edge_costs(
        data.layer_classes, data.class_weights, angle_weight=cfg.ANGLE_WEIGHT
    )

    # for the second graph, the shifts must be flipped by sign
    if i == 1:
        graph.angle_norm_factor = cfg.MAX_ANGLE_LG
        graph.shifts = np.asarray(two_graphs[0].shifts) * (-1)
        graph.shift_tuples = graph.shifts
    else:
        graph.set_shift(
            cfg.PYLON_DIST_MIN,
            cfg.PYLON_DIST_MAX,
            vec,
            cfg.MAX_ANGLE,
            max_angle_lg=cfg.MAX_ANGLE_LG
        )

    # add vertices
    graph.add_nodes()
    corridor = np.ones(two_corrs[i].shape) * 0.5
    graph.set_corridor(
        corridor,
        start_points[i],
        start_points[i],
        factor_or_n_edges=1  # start_points[(i+1)%2],
    )
    graph.add_edges()
    graph.sum_costs()

    # save the current graph
    two_graphs[i] = graph

# TRANSFORM AND CONCATENATE PATH
splitGraph = SPLIT_TYPE(two_graphs, pix_per_part, margin, padding)
splitGraph.construct_critical_zones()
splitGraph.get_sp_trees(start_points)

path = splitGraph.get_concat_path(start_points)

instance_mean = np.mean(instance, axis=0)
plot_path(instance_mean, path, buffer=0, out_path=OUT_PATH + ".png")
