import argparse
import os
import pickle
import time
# import warnings
import numpy as np
import warnings
import pandas as pd
# utils imports
from power_planner.data_reader import DataReader
from power_planner import graphs
from power_planner.evaluate_path import save_path_cost_csv

from power_planner.utils.utils import (time_test_csv, load_config)

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
PATH_FILES = f"../data/instance_{INST}.nosync"
IOPATH = os.path.join(PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat")

# LOAD CONFIGURATION
config = load_config(
    os.path.join(PATH_FILES, f"{INST}_config.json"), scale_factor=SCALE_PARAM
)

OUT_PATH_orig = OUT_PATH

# TODO: select appropriate parameters for sensitivity analysis:
config.graph.angle_weight = 0.3
config.graph.edge_weight = 0.5

change_dict = {
    "I_1422_Wald_ohne_Bedeutung": [2, 0, 1, 3],
    "I_2713_Landschaftspraegende_Denkmaeler_inkl_3000m": [0, 1, 3],
    "I_1111_Wohnumfeldschutz": [0, 1, 2],
    "I_1311_Landschaftl_Vorbehaltsgebiet": [0, 1, 3],
    "I_1421_Bannwald": [0, 1, 2],
    "I_1812_Strassen_ueberregional.tif": [-3, -2, 0],
    "TA_Laerm_Grenzwerte": [0, 1, 2],
    "I_2214_VS_5000m": [0, 1, 3],
    "I_2224_Gesetzl_geschuetzte_Biotope": [0, 1, 3],
    "I_2511_Wald_Bedeutung_Klimaschutz": [0, 1, 3],
    "I_2611_Landschaftsbildeinheit_sehr_und_hohe_Bedeutung": [0, 1, 3],
    "I_2711_Bodendenkmaeler": [0, 2, 3]
}

layer_list = list(change_dict.keys())
layer_combs = []
for i in range(len(layer_list)):
    for j in range(i + 1, len(layer_list)):
        layer_combs.append((layer_list[i], layer_list[j]))
        # print(layer_list[i], layer_list[j])

# for layer, new_weights in change_dict.items():
for (layer1, layer2) in layer_combs:
    weight_csv = pd.read_csv(
        os.path.join(PATH_FILES, config.data.weight_csv[:-4] + "_orig.csv")
    ).set_index("Layer Name")
    # for w in new_weights:
    weight_csv.loc[layer1, "weight_1"] = 0
    weight_csv.loc[layer2, "weight_1"] = 0
    weight_csv.to_csv(os.path.join(PATH_FILES, config.data.weight_csv))

    # CONSTRUCT DATA
    data = DataReader(PATH_FILES, SCENARIO, SCALE_PARAM, config)
    instance, edge_cost, instance_corr, config = data.get_data()
    cfg = config.graph
    start_inds = cfg.start_inds
    dest_inds = cfg.dest_inds

    # ID
    # if layer == "I_1422_Wald_ohne_Bedeutung" and w == 2:
    #     ID = "baseline"
    # else:
    ID = f"-{layer1}-{layer2}"
    OUT_PATH = OUT_PATH_orig + ID

    # DEFINE GRAPH AND ALGORITHM
    graph = GRAPH_TYPE(
        instance, instance_corr, edge_instance=edge_cost, verbose=cfg.verbose
    )
    tic = time.time()

    # PROCESS
    path, path_costs, cost_sum = graph.single_sp(**vars(cfg))

    time_pipeline = round(time.time() - tic, 3)
    print("FINISHED :", time_pipeline)
    print("----------------------------")

    # SAVE timing test
    # time_test_csv(
    #     ID, cfg.csv_times, SCALE_PARAM * 10, 1, "impl_lg_" + INST, graph,
    #     1, cost_sum, 1, time_pipeline, 1
    # )

    # -------------  PLOTTING: ----------------------
    save_path_cost_csv(OUT_PATH, [path], instance, **vars(cfg))
