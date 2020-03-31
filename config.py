import numpy as np


class Config():

    def __init__(self, SCALE_PARAM):
        # define paths:
        self.HARD_CONS_PATH = "hard_constraints"
        self.CORR_PATH = "corridor/Corridor_BE.tif"
        self.COST_PATH = "COSTSURFACE.tif"
        self.START_PATH = "start_point/Start"
        self.DEST_PATH = "dest_point/Destination"
        self.WEIGHT_CSV = "layer_weights.csv"
        self.CSV_TIMES = "outputs/time_tests.csv"

        # HYPERPARAMETER:
        RASTER = 10
        PYLON_DIST_MIN = 150
        PYLON_DIST_MAX = 250
        self.MAX_ANGLE = 0.5 * np.pi
        self.SCENARIO = 1

        self.VERBOSE = 1
        self.GTNX = 1

        # compute pylon distances:
        self.PYLON_DIST_MIN = PYLON_DIST_MIN / RASTER
        self.PYLON_DIST_MAX = PYLON_DIST_MAX / RASTER
        if SCALE_PARAM > 1:
            self.PYLON_DIST_MIN /= SCALE_PARAM
            self.PYLON_DIST_MAX /= SCALE_PARAM
        print(
            "defined pylon distances in raster:", self.PYLON_DIST_MIN,
            self.PYLON_DIST_MAX
        )
