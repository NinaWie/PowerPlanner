import argparse
import os
import pickle
import time
# import warnings
import numpy as np
import json
# import matplotlib.pyplot as plt

# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.plotting import plot_path_costs, plot_pipeline_paths, plot_path, plot_k_sp, plot_pareto_paths
from power_planner.utils import get_distance_surface, append_to_csv
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
# parser.add_argument('scale', help="downsample", type=int, default=1)
args = parser.parse_args()

if args.cluster:
    PATH_FILES = os.path.join("..", "data")
else:
    PATH_FILES = "/Users/ninawiedemann/Downloads/tifs_new"

SCALE_PARAM = 2
NOTES = "None"  # "mean-all-simple"

IOPATH = os.path.join(PATH_FILES, "data_dump_" + str(SCALE_PARAM) + ".dat")

cfg = Config(SCALE_PARAM)

# load from pickle
with open(IOPATH, "rb") as infile:
    data = pickle.load(infile)
    (instance, instance_corr, start_inds, dest_inds) = data.data

COMPARISONS = [
    # BEST ONES FROM BEFORE:
    ["2-100-mean", "WeightedGraph", [(2, 100), (1, 0)], "mean", "simple"],
    ["2-100-min", "WeightedGraph", [(2, 100), (1, 0)], "min", "simple"],
    ["2-100-max", "WeightedGraph", [(2, 100), (1, 0)], "max", "simple"],
    [
        "2-100-mean-watershed", "WeightedGraph", [(2, 100), (1, 0)], "mean",
        "watershed"
    ],
    [
        "2-100-min-watershed", "WeightedGraph", [(2, 100), (1, 0)], "min",
        "watershed"
    ],
    [
        "2-100-max-watershed", "WeightedGraph", [(2, 100), (1, 0)], "max",
        "watershed"
    ],
    ["4-100-mean", "WeightedGraph", [(4, 100), (1, 0)], "mean", "simple"],
    [
        "4-100-mean-watershed", "WeightedGraph", [(4, 100), (1, 0)], "mean",
        "watershed"
    ]
    # COMPARISONS = [
    #     # BEST ONES FROM BEFORE:
    #     ["baseline", "WeightedKSP", [(1, 0)], 0.2],
    #     ["2-100-05", "WeightedKSP", [(2, 100), (1, 0)], 0.5],
    #     ["2-100-02", "WeightedKSP", [(2, 100), (1, 0)], 0.2],
    #     ["3-200-02", "WeightedKSP", [(3, 200), (2, 100)], 0.2],
    #     ["09-100-05", "RandomWeightedGraph", [(0.9, 100), (0, 0)], 0.5],
    #     ["09-100-02", "RandomWeightedGraph", [(0.9, 100), (0, 0)], 0.2],
    #     [
    #         "095-100-09-50-02", "RandomWeightedGraph",
    #         [(0.95, 100), (0.9, 50), (0, 0)], 0.2
    #     ],
]

for compare_params in COMPARISONS:
    print("------------------------------------------------")
    print("---- NEW CONFIG:", compare_params, "------------")
    ID = compare_params[0]
    OUT_PATH = "outputs/scale2watershed/" + ID
    graph_name = "graphs." + compare_params[1]
    GRAPH_TYPE = eval(graph_name)
    PIPELINE = compare_params[2]
    # ksp_overlap = compare_params[3]
    sample_func = compare_params[3]
    sample_method = compare_params[4]
    # LineGraph, WeightedGraph, RandomWeightedGraph, RandomLineGraph
    print("graph type:", GRAPH_TYPE)

    # DEFINE GRAPH AND ALGORITHM
    graph = GRAPH_TYPE(
        instance, instance_corr, graphtool=cfg.GTNX, verbose=cfg.VERBOSE
    )

    graph.set_edge_costs(
        data.layer_classes, data.class_weights, angle_weight=cfg.ANGLE_WEIGHT
    )
    graph.set_shift(
        cfg.PYLON_DIST_MIN,
        cfg.PYLON_DIST_MAX,
        dest_inds - start_inds,
        cfg.MAX_ANGLE,
        max_angle_lg=cfg.MAX_ANGLE_LG
    )
    # add vertices
    graph.add_nodes()

    # START PIPELINE
    tic = time.time()
    corridor = np.ones(instance_corr.shape) * 0.5  # start with all
    output_paths = []
    plot_surfaces = []
    time_infos = []

    for (factor, dist) in PIPELINE:
        print("----------- PIPELINE", factor, dist, "---------------")
        graph.set_corridor(
            factor, corridor, start_inds, dest_inds, sample_func, sample_method
        )
        print("1) set cost rest")
        graph.add_edges()
        print("2) added edges", graph.n_edges)
        print("number of vertices:", graph.n_nodes)

        # weighted sum of all costs
        graph.sum_costs()
        source_v, target_v = graph.add_start_and_dest(start_inds, dest_inds)
        print("3) summed cost, get source and dest")
        # get actual best path
        path, path_costs, cost_sum = graph.get_shortest_path(
            source_v, target_v
        )
        print(path)
        print(source_v, target_v)
        print("4) shortest path")
        # save for inspection
        output_paths.append((path, path_costs))
        plot_surfaces.append(
            graph.cost_rest[2].copy()
        )  # TODO: mean makes black
        # get several paths --> possible to replace by pareto_out[0]
        # paths = [path]
        time_infos.append(graph.time_logs.copy())

        if cfg.VERBOSE:
            del graph.time_logs['edge_list_times']
            del graph.time_logs['add_edges_times']
            print(graph.time_logs)

        if dist > 0:

            # Define paths around which to place corridor
            # graph.get_shortest_path_tree(source_v, target_v)
            # ksp = graph.k_shortest_paths(
            #     source_v, target_v, 3, overlap=ksp_overlap
            # )  # cfg.KSP)
            # paths = [k[0] for k in ksp]
            paths = [path]

            # time test csv
            e_c, c_c, p_c, t_c = tuple(
                [round(s, 3) for s in np.sum(path_costs, axis=0)]
            )
            param_list = [
                ID,
                SCALE_PARAM,
                compare_params[1],
                PIPELINE,
                graph.n_nodes,
                graph.n_edges,
                graph.time_logs["add_nodes"],
                graph.time_logs["add_all_edges"],
                graph.time_logs["shortest_path"],
                # graph.time_logs["shortest_path_tree"], graph.time_logs["ksp"],
                0,
                0,
                0,
                graph.time_logs["downsample"],
                e_c,
                c_c,
                p_c,
                t_c,
                cost_sum,
                0
            ]
            append_to_csv(cfg.CSV_TIMES, param_list)

            # do specified numer of dilations
            corridor = get_distance_surface(
                graph.pos2node.shape, paths, mode="dilation", n_dilate=dist
            )
            print("5) compute distance surface")
            # remove the edges of vertices in the corridor (to overwrite)
            graph.remove_vertices(corridor, delete_padding=cfg.PYLON_DIST_MAX)
            print("6) remove edges")

    # BEST IN WINDOW
    # path_window, path_window_cost, cost_sum_window = graph.best_in_window(
    #     30, 35, 60, 70, source_v, target_v
    # )
    # print("cost actually", cost_sum, "cost_new", cost_sum_window)

    # # COMPUTE KSP
    # graph.get_shortest_path_tree(source_v, target_v)
    # ksp = graph.k_shortest_paths(
    #     source_v, target_v, cfg.KSP, overlap=ksp_overlap
    # )
    # plot_k_sp(ksp, graph.instance, out_path=OUT_PATH)

    # # PARETO
    # pareto_out = graph.get_pareto(
    #     10, source_v, target_v, compare=[2, 3], out_path=OUT_PATH
    # )
    # plot_pareto_paths(pareto_out, graph.instance, out_path=OUT_PATH)

    time_pipeline = round(time.time() - tic, 3)
    print("FINISHED PIPELINE:", time_pipeline)

    # SAVE timing test
    e_c, c_c, p_c, t_c = tuple(
        [round(s, 3) for s in np.sum(path_costs, axis=0)]
    )
    param_list = [
        ID,
        SCALE_PARAM,
        compare_params[1],
        PIPELINE,
        graph.n_nodes,
        graph.n_edges,
        graph.time_logs["add_nodes"],
        graph.time_logs["add_all_edges"],
        graph.time_logs["shortest_path"],
        # graph.time_logs["shortest_path_tree"],
        # graph.time_logs["ksp"], graph.time_logs["pareto"],
        0,
        0,
        0,
        graph.time_logs["downsample"],
        e_c,
        c_c,
        p_c,
        t_c,
        cost_sum,
        time_pipeline
    ]
    append_to_csv(cfg.CSV_TIMES, param_list)

    # PLOTTING:
    # FOR PIPELINE
    plot_pipeline_paths(
        plot_surfaces,
        output_paths,
        buffer=2,
        out_path=OUT_PATH + "_pipeline.png"
    )
    # FOR KSP:
    # with open(OUT_PATH + "_ksp.json", "w") as outfile:
    #     json.dump(ksp, outfile)
    # plot_k_sp(ksp, graph.instance * (corridor > 0).astype(int), out_path=OUT_PATH)

    # FOR WINDOW
    # plot_path(
    #     graph.instance, path_window, buffer=0, out_path=OUT_PATH + "_window.png"
    # )
    # SIMPLE
    # plot_path(graph.instance, path, buffer=0, out_path=OUT_PATH + ".png")

    # FOR COST COMPARISON
    # plot_path_costs(
    #     instance * instance_corr,
    #     path,
    #     path_costs,
    #     data.layer_classes,
    #     buffer=2,
    #     out_path=OUT_PATH + "_costs.png"
    # )

    # SAVE graph
    # graph.save_graph(OUT_PATH + "_graph")
    # np.save(OUT_PATH + "_pos2node.npy", graph.pos2node)

    # SAVE JSON WITH INFOS
    DataReader.save_pipeline_infos(
        OUT_PATH, output_paths, time_infos, PIPELINE, SCALE_PARAM
    )

    # LINE GRAPH FROM FILE:
    # elif GRAPH_TYPE == "LINE_FILE":
    #     # Load file and derive line graph
    #     graph_file = "outputs/path_02852_graph"
    #     graph = LineGraphFromGraph(
    #         graph_file, instance, instance_corr, graphtool=GTNX, verbose=VERBOSE
    #     )
