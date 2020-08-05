import argparse
import os
import pickle
# import warnings
import numpy as np
import warnings
import matplotlib.pyplot as plt

# utils imports
try:
    from power_planner.data_reader import DataReader
except ImportError:
    warnings.warn("DATA READER CANNOT BE USED - IMPORTS")
from power_planner import graphs
from power_planner.utils.utils import (load_config)

parser = argparse.ArgumentParser()
parser.add_argument('-cluster', action='store_true')
parser.add_argument('-i', '--instance', type=str, default="test")
parser.add_argument('-s', '--scale', help="resolution", type=int, default=2)
args = parser.parse_args()

# define out save name
ID = "xytest_" + args.instance  # str(round(time.time() / 60))[-5:]
OUT_DIR = os.path.join("..", "outputs")
OUT_PATH = os.path.join(OUT_DIR, ID)

SCALE_PARAM = args.scale
SCENARIO = 1
INST = args.instance  # "belgium"

for SCALE_PARAM in [1, 2, 5]:
    #     for INST in ["belgium", "ch", "de"]:
    #         print()
    #         print("-------------- new scnario", SCALE_PARAM, INST, "-------")

    SAVE_PICKLE = 1

    # define IO paths
    PATH_FILES = f"../data/instance_{INST}.nosync"
    IOPATH = os.path.join(
        PATH_FILES, f"{INST}_data_{SCENARIO}_{SCALE_PARAM}.dat"
    )

    # LOAD CONFIGURATION
    cfg = load_config(
        os.path.join(PATH_FILES, f"{INST}_config.json"),
        scale_factor=SCALE_PARAM
    )

    # READ DATA

    # read in files
    data = DataReader(PATH_FILES, SCENARIO, SCALE_PARAM, cfg)
    instance, edge_cost, instance_corr, config = data.get_data()
    # get graph processing specific cfg
    cfg = config.graph
    start_inds = cfg.start_inds
    dest_inds = cfg.dest_inds
    # save
    if SAVE_PICKLE:
        data_out = (instance, edge_cost, instance_corr, config)
        with open(IOPATH, "wb") as outfile:
            pickle.dump(data_out, outfile)
        print("successfully saved data")

    visualize_corr = 1 - instance_corr
    visualize_corr[visualize_corr == 1] = np.inf
    plt.figure(figsize=(8, 5))
    plt.imshow(np.sum(instance, axis=0) + visualize_corr)
    plt.colorbar()
    plt.savefig(f"surface_s100_i100_{INST}_{SCALE_PARAM}.png")
    plt.imshow(np.mean(edge_cost, axis=0))
    plt.savefig(f"edgecost_s100_i100_{INST}_{SCALE_PARAM}.png")