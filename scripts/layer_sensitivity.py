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
config.graph.ANGLE_WEIGH = 0.3
config.graph.EDGE_WEIGHT = 0.5

change_dict = {
    "I_1422_Wald_ohne_Bedeutung": [0, 1, 3],
    "I_2713_Landschaftspraegende_Denkmaeler_inkl_3000m": [0, 1, 2]
}

for layer, new_weights in change_dict.items():
    weight_csv = pd.read_csv(
        os.path.join(PATH_FILES, config.data.WEIGHT_CSV[:-4] + "_orig.csv")
    ).set_index("Layer Name")
    for w in new_weights:
        weight_csv.loc[layer, "weight_1"] = w
        weight_csv.to_csv(os.path.join(PATH_FILES, config.data.WEIGHT_CSV))

        # CONSTRUCT DATA
        data = DataReader(PATH_FILES, SCENARIO, SCALE_PARAM, config)
        instance, edge_cost, instance_corr, config = data.get_data()
        cfg = config.graph
        start_inds = cfg.start_inds
        dest_inds = cfg.dest_inds

        # ID
        ID = f"_{layer}_{w}"
        OUT_PATH = OUT_PATH_orig + ID

        # DEFINE GRAPH AND ALGORITHM
        graph = GRAPH_TYPE(
            instance,
            instance_corr,
            edge_instance=edge_cost,
            verbose=cfg.VERBOSE
        )
        tic = time.time()

        # PROCESS
        path, path_costs, cost_sum = graph.single_sp(**vars(cfg))

        time_pipeline = round(time.time() - tic, 3)
        print("FINISHED :", time_pipeline)
        print("----------------------------")

        # SAVE timing test
        time_test_csv(
            ID, cfg.CSV_TIMES, SCALE_PARAM * 10, cfg.GTNX, "impl_lg_" + INST,
            graph, 1, cost_sum, 1, time_pipeline, 1
        )

        # -------------  PLOTTING: ----------------------
        graph.save_path_cost_csv(OUT_PATH, [path], **vars(cfg))
