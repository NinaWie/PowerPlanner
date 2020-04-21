import numpy as np


class Config():

    def __init__(self, SCALE_PARAM):
        # define paths:
        self.HARD_CONS_PATH = "hard_constraints"
        self.CORR_PATH = "corridor/Corridor_BE.tif"
        self.COST_PATH = "COSTSURFACE.tif"
        # self.START_PATH = "start_point/Start"
        # self.DEST_PATH = "dest_point/Destination"
        self.START_PATH = "start_point/large_instance_start"
        self.DEST_PATH = "dest_point/large_instance_destination"
        self.WEIGHT_CSV = "layer_weights.csv"
        self.CSV_TIMES = "outputs/time_tests.csv"

        # HYPERPARAMETER:
        RASTER = 10
        PYLON_DIST_MIN = 150
        PYLON_DIST_MAX = 250
        self.MAX_ANGLE = 0.5 * np.pi
        self.MAX_ANGLE_LG = 0.25 * np.pi
        self.SCENARIO = 3

        self.VERBOSE = 1
        self.GTNX = 1

        self.ANGLE_WEIGHT = 0.1

        self.KSP = 10

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
