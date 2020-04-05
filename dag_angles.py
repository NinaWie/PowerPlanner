from power_planner.utils import get_half_donut, angle, normalize
from power_planner.constraints import ConstraintUtils
from power_planner.data_reader import DataReader
from power_planner.plotting import plot_path
from config import Config
import numpy as np
import argparse
import os
import time
import pickle
# import matplotlib.pyplot as plt


class PowerBF():

    def __init__(self, instance, instance_corr):
        self.instance = instance
        self.instance_corr = instance_corr
        self.x_len, self.y_len = instance_corr.shape
        self.fill_val = np.inf
        self.angle_norm_factor = 3 * np.pi / 4
        self.time_logs = {}

    def set_shift(self, lower, upper, vec, max_angle):
        """
        Initialize shift variable by getting the donut values
        :param lower, upper: min and max distance of pylons
        :param vec: vector of diretion of edges
        :param max_angle: Maximum angle of edges to vec
        """
        self.shifts = get_half_donut(lower, upper, vec, angle_max=max_angle)

    def dist_init(self, start_inds):
        tic = time.time()

        self.dists = np.zeros((len(self.shifts), self.x_len, self.y_len))
        self.dists += self.fill_val
        i, j = start_inds
        self.dists[:, i, j] = self.instance[i, j]
        self.dists_argmin = np.zeros(self.dists.shape)

        self.time_logs["init_dists"] = round(time.time() - tic, 3)

    def _precompute_angles(self):
        tic = time.time()
        angles_all = np.zeros((len(self.shifts), len(self.shifts)))
        for i in range(len(self.shifts)):
            angles_all[i] = [angle(s, self.shifts[i]) for s in self.shifts]
        angles_all = angles_all / self.angle_norm_factor
        self.time_logs["compute_angles"] = round(time.time() - tic, 3)
        return angles_all

    def compute_dists(self, n_iters, weights=[0.5, 0.5]):
        tic = time.time()

        angle_weight, resistance_weight = tuple(
            np.array(weights) / np.sum(weights)
        )
        print("weights", angle_weight, resistance_weight)

        # precompute angles
        angles_all = self._precompute_angles()

        for _ in range(n_iters):
            for i in range(len(self.shifts)):
                # compute angle from the sorted neighbor list to the new edge
                # immer plus das minimum
                curr_shift = self.shifts[i]

                angles = angles_all[i]
                # shift dists by this shift
                # todo: avoid swaping dimenions each time
                cost_switched = np.moveaxis(self.dists, 0, -1)
                # shift by shift
                costs_shifted = ConstraintUtils.shift_surface(
                    cost_switched, curr_shift, fill_val=self.fill_val
                )

                # add new costs for current edge
                angle_cost = angle_weight * angles
                together = np.moveaxis(
                    costs_shifted + angle_cost, -1, 0
                ) + self.instance * resistance_weight
                # 28 x 10 x 10 + 28 angles + 10 x 10

                # get minimum path cost for each edge
                weighted_costs_shifted = np.min(together, axis=0)

                concat = np.array([self.dists[i], weighted_costs_shifted])
                # get spots that are actually updated
                changed_ones = np.argmin(concat, axis=0)
                # get argmin for each edge
                # --> remember where the value on this edge came from
                argmin_together = np.argmin(together, axis=0)
                # update predecessors
                self.dists_argmin[i, changed_ones > 0] = argmin_together[
                    changed_ones > 0]

                # update accumulated path costs
                self.dists[i] = np.min(concat, axis=0)

        self.time_logs["dists"] = round(time.time() - tic, 3)
        time_per_iter = (time.time() - tic) / n_iters
        time_per_shift = (time.time() - tic) / (n_iters * len(self.shifts))
        self.time_logs["dists_per_iter"] = round(time_per_iter, 3)
        self.time_logs["dists_per_shift"] = round(time_per_shift, 3)

    def _compute_angles_manually(self, path):
        tic = time.time()
        ang_out = [0]
        for p in range(len(path) - 2):
            vec1 = path[p + 1] - path[p]
            vec2 = path[p + 2] - path[p + 1]
            ang_out.append(
                round(angle(vec1, vec2) / self.angle_norm_factor, 2)
            )
        ang_out.append(0)

        self.time_logs["angle_costs"] = round(time.time() - tic, 3)
        return ang_out

    def get_path_from_dists(self, start_inds, dest_inds):
        tic = time.time()
        curr_point = dest_inds
        path = [dest_inds]
        path_costs = [self.instance[dest_inds[0], dest_inds[1]]]
        # first minimum: angles don't matter, just min of in-edges
        min_shift = np.argmin(self.dists[:, dest_inds[0], dest_inds[1]])
        # track back until start inds
        while np.any(curr_point - start_inds):
            new_point = curr_point - self.shifts[int(min_shift)]
            # get new shift from argmins
            min_shift = self.dists_argmin[int(min_shift), curr_point[0],
                                          curr_point[1]]
            # append costs and path
            path_costs.append(self.instance[new_point[0], new_point[1]])
            path.append(new_point)
            curr_point = new_point

        path = np.flip(np.asarray(path), axis=0)
        path_costs = np.flip(np.asarray(path_costs), axis=0)

        # compute angle costs on path:
        ang_costs = self._compute_angles_manually(path)
        path_costs = np.array(list(zip(path_costs, ang_costs)))

        self.time_logs["get_path"] = round(time.time() - tic, 3)
        return path, path_costs

    def compute_path(self, n_iters, start_inds, dest_inds, weights=[0.5, 0.5]):
        self.compute_dists(n_iters, weights=weights)
        # plt.imshow(np.min(dists, axis=0))
        # plt.colorbar()
        # plt.show()
        path, path_costs = self.get_path_from_dists(start_inds, dest_inds)

        return path, path_costs


if __name__ == "__main__":

    # ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-cluster', action='store_true')
    parser.add_argument('scale', help="downsample", type=int, default=1)
    args = parser.parse_args()

    if args.cluster:
        PATH_FILES = os.path.join("..", "data")
    else:
        PATH_FILES = "/Users/ninawiedemann/Downloads/tifs_new"

    # DEFINE CONFIGURATION
    ID = "test_BF_onlyCosts"  # str(round(time.time() / 60))[-5:]
    OUT_PATH = "outputs/path_" + ID
    N_ITERS = 50
    WEIGHTS = [0.5, 0.5]

    # DATA IO
    LOAD = 1
    if args.cluster:
        LOAD = 1
    SAVE_PICKLE = 0
    IOPATH = os.path.join(PATH_FILES, "data_dump_" + str(args.scale) + ".dat")

    # OTHER CONFIG
    cfg = Config(args.scale)

    # READ DATA
    if LOAD:
        # load from pickle
        with open(IOPATH, "rb") as infile:
            data = pickle.load(infile)
            (instance, instance_corr, start_inds, dest_inds) = data.data
    else:
        # read in files
        data = DataReader(
            PATH_FILES, cfg.CORR_PATH, cfg.WEIGHT_CSV, cfg.SCENARIO, args.scale
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

    # compute cost surface:
    print("class_weights", data.class_weights)
    weighted_inst = np.sum(
        np.moveaxis(instance, 0, -1) * data.class_weights, axis=2
    )
    print(weighted_inst.shape, instance_corr.shape)

    tic_all = time.time()

    # initialize
    bf = PowerBF(weighted_inst, instance_corr)

    # init dists
    bf.set_shift(cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX, vec, cfg.MAX_ANGLE)
    bf.dist_init(start_inds)

    # main computation
    bf.compute_dists(N_ITERS, weights=WEIGHTS)
    path, path_costs = bf.get_path_from_dists(start_inds, dest_inds)

    bf.time_logs["bf_runtime"] = round(time.time() - tic_all, 3)

    plot_path(
        normalize(weighted_inst), path, buffer=1, out_path=OUT_PATH + ".png"
    )
    print(bf.time_logs)

    DataReader.save_json(OUT_PATH, path, path_costs, bf.time_logs, args.scale)
