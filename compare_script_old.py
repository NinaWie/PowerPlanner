import argparse
import os
import pickle
import time
# import warnings
import numpy as np
# import matplotlib.pyplot as plt

# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.plotting import plot_path_costs, plot_pipeline_paths
from power_planner.utils.utils import get_distance_surface, time_test_csv
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
# parser.add_argument('scale', help="downsample", type=int, default=1)
args = parser.parse_args()

if args.cluster:
    PATH_FILES = os.path.join("..", "data")
else:
    PATH_FILES = "/Users/ninawiedemann/Downloads/tifs_new"

# summarize: mean/max/min, remove: all/surrounding, sample: simple/watershed
NOTES = "None"  # "mean-all-simple"
LOAD = 1
SAVE_PICKLE = 0

COMPARISONS = [
    ["norm-1-2-100", 1, "Weighted", [(2, 100), (1, 0)]],
    ["norm-1-4-100-2-50", 1, "Weighted", [(4, 100), (2, 50), (1, 0)]],
    ["norm-1-5-100-3-50", 1, "Weighted", [(5, 100), (3, 50), (1, 0)]],
    [
        "norm-1-5-200-3-100-2-50", 1, "Weighted",
        [(5, 200), (3, 100), (2, 50), (1, 0)]
    ],
    ["norm-1-4-200-2-100", 1, "Weighted", [(4, 200), (2, 100), (1, 0)]],
    ["norm-1-3-150-2-75", 1, "Weighted", [(3, 150), (2, 75), (1, 0)]],
    ["random-1-9-100", 1, "RandomWeighted", [(0.9, 100), (0, 0)]],
    [
        "random-1-9-100-9-50", 1, "RandomWeighted",
        [(0.9, 100), (0.9, 50), (0, 0)]
    ],
    [
        "random-1-95-100-95-50", 1, "RandomWeighted",
        [(0.95, 100), (0.95, 50), (0, 0)]
    ],
    [
        "random-1-95-200-95-100", 1, "RandomWeighted",
        [(0.95, 200), (0.95, 100), (0, 0)]
    ],
    [
        "random-1-9-100-8-50", 1, "RandomWeighted",
        [(0.9, 100), (0.8, 50), (0, 0)]
    ],
    [
        "random-1-8-200-8-100", 1, "RandomWeighted",
        [(0.8, 200), (0.8, 100), (0, 0)]
    ],
]

# COMPARISONS_1 = [
#     ["norm-2-direct", 2, "Weighted", [(1, 0)]],
#     ["norm-2-pipe", 2, "Weighted", [(2, 50), (1, 0)]],
#     ["line-5-direct", 5, "Line", [(1, 0)]],
#     ["line-5-pipe", 5, "Line", [(2, 40), (1, 0)]],
#     ["line-andom-5-pipe", 5, "RandomLine", [(0.9, 40), (0, 0)]],
#     ["norm-random-2-pipe", 2, "RandomWeighted", [(0.9, 50), (0, 0)]],
#     ["norm-1-pipe", 1, "Weighted", [(2, 100), (1, 0)]],
#     ["norm-1-pipe2", 1, "Weighted", [(4, 100), (2, 50), (1, 0)]],
#     [
#         "norm-random-1-pipe", 1, "RandomWeighted",
#         [(0.95, 100), (0.9, 50), (0, 0)]
#     ], ["norm-random-1-pipe2", 1, "RandomWeighted", [(0.95, 100), (0, 0)]],
#     ["norm-1-direct", 1, "Weighted", [(1, 0)]]
# ]

for compare_params in COMPARISONS:
    ID = compare_params[0]
    OUT_PATH = "outputs/path_" + ID
    SCALE_PARAM = compare_params[1]
    graph_name = "graphs." + compare_params[2] + "Graph"
    GRAPH_TYPE = eval(graph_name)
    PIPELINE = compare_params[3]
    # LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph
    print("graph type:", GRAPH_TYPE)

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
            PATH_FILES, cfg.CORR_PATH, cfg.WEIGHT_CSV, cfg.SCENARIO,
            SCALE_PARAM
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

    # START PIPELINE
    tic = time.time()
    corridor = np.ones(instance_corr.shape) * 0.5  # start with all
    output_paths = []
    plot_surfaces = []
    time_infos = []

    for (factor, dist) in PIPELINE:
        print("----------- PIPELINE", factor, dist, "---------------")
        # DEFINE GRAPH AND ALGORITHM
        graph = GRAPH_TYPE(
            instance, instance_corr, graphtool=cfg.GTNX, verbose=cfg.VERBOSE
        )
        # BUILD GRAPH:
        graph.set_edge_costs(data.layer_classes, data.class_weights)
        graph.set_shift(
            cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX, vec, cfg.MAX_ANGLE
        )
        # add vertices
        graph.add_nodes()
        print("0) initialize graph, add nodes")
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
            # PRINT AND SAVE timing test
            time_test_csv(
                ID, cfg.CSV_TIMES, SCALE_PARAM, cfg.GTNX, GRAPH_TYPE, graph,
                path_costs, cost_sum, dist, 0, NOTES
            )
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
        path_costs, cost_sum, dist, time_pipeline, NOTES
    )

    # PLOT RESULT
    plot_pipeline_paths(
        plot_surfaces,
        output_paths,
        buffer=2,
        out_path=OUT_PATH + "_pipeline.png"
    )
    # plot_path(instance * instance_corr, path, buffer=1, out_path=OUT_PATH+".png")
    plot_path_costs(
        instance * instance_corr,
        path,
        path_costs,
        graph.cost_classes,
        buffer=2,
        out_path=OUT_PATH + ".png"
    )

    # SAVE graph
    # graph.save_graph(OUT_PATH + "_graph")
    # np.save(OUT_PATH + "_pos2node.npy", graph.pos2node)

    # data.save_coordinates(path, OUT_PATH, scale_factor=SCALE_PARAM)
    # DataReader.save_json(OUT_PATH, path, path_costs, graph.time_logs,SCALE_PARAM)
    DataReader.save_pipeline_infos(
        OUT_PATH, output_paths, time_infos, PIPELINE, SCALE_PARAM
    )

    graph = None
