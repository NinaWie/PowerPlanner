import os
from PIL import Image
import rasterio
import shapefile
import numpy as np
import json
import pandas as pd
import functools
import time
# import matplotlib.pyplot as plt

from power_planner.utils import normalize


# Decorator
def reduce_instance(func):
    """
    Scale down an instance by a factor
    :param img: instance, 2 dim array
    :returns: array of sizes img.size/factor
    """

    @functools.wraps(func)
    def call_and_reduce(self, *args, **kwargs):

        def reduce(img, scale_factor):
            x_len_new = img.shape[0] // self.scale_factor
            y_len_new = img.shape[1] // self.scale_factor
            new_img = np.zeros((x_len_new, y_len_new))
            for i in range(x_len_new):
                for j in range(y_len_new):
                    patch = img[i * self.scale_factor:(i + 1) *
                                self.scale_factor, j *
                                self.scale_factor:(j + 1) * self.scale_factor]
                    new_img[i, j] = np.mean(patch)
            return np.swapaxes(new_img, 1, 0)

        img = func(self, *args, **kwargs)
        if self.scale_factor > 1:
            if len(img.shape) == 2:
                return reduce(img, self.scale_factor)  # TODO
            elif len(img.shape) == 3:
                out = [reduce(img_i, self.scale_factor) for img_i in img]
                return np.array(out)
            else:
                raise ValueError("Passed array is not of the right shape")
        else:
            return np.swapaxes(img, 1, 0)  # TODO: why

    return call_and_reduce


def binarize(func):

    def call_and_binarize(*args, **kwargs):
        return np.ceil(func(*args, **kwargs))

    return call_and_binarize


def padding(func):

    @functools.wraps(func)
    def call_and_pad(self, *args, **kwargs):
        img = func(self, *args, **kwargs)
        if self.padding > 0:
            a = self.padding
            return np.pad(img, ((a, a), (a, a)))
        else:
            return img

    return call_and_pad


class DataReader():

    def __init__(self, base_path, instance_path, weight_csv, scale_factor):
        self.path = base_path
        self.scale_factor = scale_factor
        weights = pd.read_csv(os.path.join(base_path, weight_csv))
        self.weights = weights.dropna()
        self.padding = 0

        with rasterio.open(os.path.join(base_path, instance_path)) as dataset:
            # binary mask
            self.corridor = dataset.dataset_mask() / 255.
            # size of array
            self.raster_size = (dataset.width, dataset.height)
            # geometric bounds and transformation
            self.geo_bounds = dataset.bounds
            self.transform_matrix = dataset.transform
            #  array itself
            self.instance = dataset.read()[0]  # TODO: redundant

    def set_padding(self, padding):
        """
        Compute how much must be padded to make roll operation work
        """
        x_inds, y_inds = np.where(self.corridor)
        x_len, y_len = self.corridor.shape
        min_dist = min(
            [x_inds[0], x_len - x_inds[-1], y_inds[0], y_len - y_inds[-1]]
        )
        self.padding = int(max([padding - min_dist, 0]))
        print(min_dist, self.padding)

    def read_tif(self, path):
        with rasterio.open(path, 'r') as ds:
            arr = ds.read()
        return arr[0]

    def read_in_tifs(self, path=None):
        if path is None:
            path = self.path
        files = os.listdir(path)
        tif_list = []
        file_list = []
        for f in files:
            if f[-3:] == "tif":
                with rasterio.open(os.path.join(path, f), 'r') as ds:
                    img = ds.read()[0]
                # img = Image.open(os.path.join(path, f))
                # img = self._resize_raster(img)
                tif_list.append(np.array(img))
                file_list.append(f[:-4])
        tif_arr = np.array(tif_list)
        tif_arr = tif_arr / 255.
        print("shape of tif array:", tif_arr.shape)
        return tif_arr, file_list

    @padding
    @reduce_instance
    def get_values_corridor(self):
        self.instance = np.array(self.instance)
        minimum = np.min(self.instance)
        self.instance[self.instance == 9999] = minimum - 1
        return self.instance

    @padding
    @binarize
    @reduce_instance
    def get_mask_corridor(self):
        return self.corridor

    @padding
    @reduce_instance
    def get_cost_surface(self, cost_path):
        # get cost surface
        with rasterio.open(os.path.join(self.path, cost_path), 'r') as ds:
            cost_img = ds.read()[0]
        print("read in cost array", cost_img.shape)
        # cost_img = Image.fromarray(arr[0])
        # cost_img = self._resize_raster(cost_img)
        return np.array(cost_img)

    @padding
    @binarize
    @reduce_instance
    def get_hard_constraints(self):
        hard_cons_rows = self.weights[self.weights["weight"] == "Forbidden"]
        # read in corresponding tifs
        hard_constraints = []
        for fname in hard_cons_rows["Layer Name"]:
            constraint = self.read_tif(
                os.path.join(self.path, "tif_layers", fname + ".tif")
            )
            constraint = self._resize_raster(constraint)
            hard_constraints.append(
                constraint.astype(int) > 0.5 * np.max(constraint)
            )
        print("hard constraints shape", np.asarray(hard_constraints).shape)
        # intersection of all of the hard constraints
        hard_constraints = np.all(
            np.asarray(hard_constraints).astype(int), axis=0
        )
        return hard_constraints

    @padding
    @reduce_instance
    def get_weighted_costs(self):
        cost_sum_arr = np.zeros(self.raster_size)
        cost_sum_arr = np.swapaxes(cost_sum_arr, 1, 0)
        layers = self.weights[self.weights["weight"] != "Forbidden"]
        for fname, weight in zip(layers["Layer Name"], layers["weight"]):
            file_path = os.path.join(self.path, "tif_layers", fname + ".tif")
            if os.path.exists(file_path):
                costs = self.read_tif(file_path)
            costs = np.absolute(normalize(costs) - 1)
            cost_sum_arr = cost_sum_arr + costs * int(weight)
        return cost_sum_arr

    @padding
    @reduce_instance
    def get_costs_per_class(self, oneclass=False):
        layers = self.weights[self.weights["weight"] != "Forbidden"]
        if oneclass:
            self.layer_classes = ["resistance"]
        else:
            self.layer_classes = np.unique(layers["class"]).tolist()
        cost_sum_arr = np.zeros(
            (
                len(self.layer_classes), self.raster_size[1],
                self.raster_size[0]
            )
        )
        for i, classname in enumerate(self.layer_classes):
            if oneclass:
                class_r = layers
            else:
                class_r = layers[layers["class"] == classname]
            for fname, weight in zip(class_r["Layer Name"], class_r["weight"]):
                file_path = os.path.join(
                    self.path, "tif_layers", fname + ".tif"
                )
                if os.path.exists(file_path):
                    costs = self.read_tif(file_path)
                # binarize single tif layer so it can be weighted
                # -1  because in tifs the costly areas are black
                costs = np.absolute(normalize(costs) - 1)
                cost_sum_arr[i] = cost_sum_arr[i] + costs * int(weight)
            # normalize cost surface with all tifs together
            norm_costs = normalize(cost_sum_arr[i])
            norm_costs[norm_costs == 0] = 0.0001  # cost cannot be zero!
            cost_sum_arr[i] = norm_costs
        return cost_sum_arr

    def get_shape_point(self, start_end_path):
        sf = shapefile.Reader(os.path.join(self.path, start_end_path))
        startendpoint = sf.shapes()[0].points
        transformed = ~self.transform_matrix * startendpoint[0]
        resized = np.asarray(transformed) / self.scale_factor
        return resized.astype(int) + self.padding

    def _resize_raster(self, raster):
        """
        input: Pillow Image!
        """
        if list(reversed(raster.shape)) != list(self.raster_size):
            raster = Image.fromarray(raster)
            print("resize: from", raster.size, "to", self.raster_size)
            raster = raster.resize(self.raster_size, resample=Image.BILINEAR)
            raster = np.array(raster)  # swapaxes(raster, 1, 0)
        return raster

    @staticmethod
    def construct_corridor(
        instance,
        hard_constraints,
        start_inds,
        dest_inds,
        emergency_dist=None,
        percent_padding=0.25
    ):
        # corr = np.zeros((1313, 1511))
        # corr[40:1260, 200:1000] = 1
        # self.instance = corr
        # self.corridor = corr

        # build rectengular corridor
        start_dest_inds = np.array([start_inds, dest_inds])
        inter_line = start_dest_inds[0] - start_dest_inds[1]
        longer = np.argmin(np.abs(inter_line))

        padding = [0, 0]
        padding[longer] = abs(int(percent_padding * inter_line[longer]))

        start_x, start_y = np.min(start_dest_inds,
                                  axis=0) - np.asarray(padding)
        end_x, end_y = np.max(start_dest_inds, axis=0) + np.asarray(padding)

        corr = np.zeros(hard_constraints.shape)
        corr[start_x:end_x, start_y:end_y] = 1

        hard_constraints = np.asarray(corr * hard_constraints)

        # add emergency points in regular grid
        if emergency_dist is not None:
            # w_inds = np.arange(start_x, end_x, emergency_dist).astype(int)
            # h_inds = np.arange(start_y, end_y, emergency_dist).astype(int)
            max_cost = np.max(instance)
            # for row in w_inds:
            #     # give maximal cost to emergency points
            #     for col in h_inds:
            #         if hard_constraints[row, col] == 0:
            #             instance[:, row, col] = max_cost  # TODO
            #         hard_constraints[row, col] = 1
            d = int(emergency_dist // 2)
            tic = time.time()
            for i in range(start_x, end_x):
                for j in range(start_y, end_y):
                    if not np.any(hard_constraints[i - d:i + d, j - d:j + d]):
                        hard_constraints[i, j] = 1
                        instance[:, i, j] = max_cost
            print("time for emergency points", round(time.time() - tic, 2))

        # set start and dest to possible points
        if hard_constraints[start_inds[0], start_inds[1]] == 0:
            print("set start as possible")
            hard_constraints[start_inds[0], start_inds[1]] = 1
            instance[:, start_inds[0], start_inds[1]] = np.mean(instance)
        if hard_constraints[dest_inds[0], dest_inds[1]] == 0:
            print("set dest as possible")
            hard_constraints[dest_inds[0], dest_inds[1]] = 1
            instance[:, dest_inds[0], dest_inds[1]] = np.mean(instance)

        return instance, hard_constraints

    def get_data(self, start_path, dest_path, emergency_dist=None):
        """
        Get all data at once: intersection of hard constraints and return
        weighted sum of all layers as cost
        :returns: the cost surface, and the corridor, a binary showing possible
        tower locations of the same shape
        """
        # # corridor defined manually
        # padding: not necessary right now
        # data.set_padding(PYLON_DIST_MAX * SCALE_PARAM)

        # corridor = self.get_mask_corridor()
        hard_constraints = self.get_hard_constraints()
        # instance_corr = corridor * hard_constraints

        instance = self.get_costs_per_class()
        # in normalize(np.sum(costs_classes, axis=0))
        # instance = normalize(self.get_weighted_costs())

        # Get start and end point
        start_inds = self.get_shape_point(start_path)
        dest_inds = self.get_shape_point(dest_path)
        print("shape of inst and corr", instance.shape, hard_constraints.shape)
        # assert instance.shape == hard_constraints.shape
        print("start cells:", start_inds, "dest cells:", dest_inds)

        instance, hard_constraints = self.construct_corridor(
            instance,
            hard_constraints,
            start_inds,
            dest_inds,
            emergency_dist=emergency_dist,
            percent_padding=0.25
        )

        return instance, hard_constraints, start_inds, dest_inds

    def save_coordinates(self, power_path, out_path, scale_factor=1):
        """
        Save the coordinates in a csv file:
        @param power_path: List of path indices [[x1, y1], [x2,y2] ...]
        @patam out_path: path and filename (without .json) where to write to
        @param scale_factor: if the instance was scaled down,
        the coordinates have to be scaled up again
        """
        power_path = (np.asarray(power_path) * scale_factor).tolist()
        coordinates = [self.transform_matrix * p for p in power_path]

        df = pd.DataFrame(np.asarray(coordinates), columns=["X", "Y"])
        df.to_csv(out_path + "_coords.csv", index=False)

    @staticmethod
    def save_json(out_path, power_path, path_costs, time_logs, scale_factor=1):
        """
        Save the path as a json file:
        @param power_path: List of path indices [[x1, y1], [x2,y2] ...]
        @patam out_path: path and filename (without .json) where to write to
        @param scale_factor: if the instance was scaled down,
        the coordinates have to be scaled up again
        """
        power_path = (np.asarray(power_path) * scale_factor).tolist()

        out_dict = {
            "path_cells": power_path,
            "edgecosts": path_costs,
            "time_logs": time_logs
        }

        # save as json
        with open(out_path + "_infos.json", "w") as outfile:
            json.dump(out_dict, outfile)
