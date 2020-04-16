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

from power_planner.utils.utils import normalize, rescale


# Decorator
def reduce_instance(func):
    """
    Scale down an instance by a factor
    :param img: instance, 2 dim array
    :returns: array of sizes img.size/factor
    """

    @functools.wraps(func)
    def call_and_reduce(self, *args, **kwargs):

        img = func(self, *args, **kwargs)

        if self.scale_factor > 1 and len(img.shape) == 3:
            img = np.array(
                [rescale(img_i, self.scale_factor) for img_i in img]
            )
        elif self.scale_factor > 1 and len(img.shape) == 2:
            img = rescale(img, self.scale_factor)

        return np.swapaxes(img, len(img.shape) - 1, len(img.shape) - 2)

    return call_and_reduce


def strip(func):

    def call_and_strip(*args, **kwargs):
        instance, instance_corr, start_inds, dest_inds = func(*args, **kwargs)

        # find x and y width to strip
        x_coords, y_coords = np.where(instance_corr)
        _, x_len, y_len = instance.shape
        print(
            "dists from img bounds", [
                np.min(x_coords), x_len - np.max(x_coords),
                np.min(y_coords), y_len - np.max(y_coords)
            ]
        )
        # padding size
        padding = min(
            [
                np.min(x_coords), x_len - np.max(x_coords),
                np.min(y_coords), y_len - np.max(y_coords)
            ]
        )
        # define borders
        up = np.min(x_coords) - padding
        down = np.max(x_coords) + padding
        left = np.min(y_coords) - padding
        right = np.max(y_coords) + padding
        # strip
        instance = instance[:, up:down, left:right]
        instance_corr = instance_corr[up:down, left:right]
        start_inds -= np.array([up, left])
        dest_inds -= np.array([up, left])

        return instance, instance_corr, start_inds, dest_inds

    return call_and_strip


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

    def __init__(
        self, base_path, instance_path, weight_csv, scenario, scale_factor
    ):
        self.path = base_path
        self.scale_factor = scale_factor
        weights = pd.read_csv(os.path.join(base_path, weight_csv))
        self.class_csv = weights.dropna()
        self.padding = 0
        self.scenario = scenario
        # get classes and corresponding weights from csv
        self.compute_class_weights()

        with rasterio.open(os.path.join(base_path, instance_path)) as dataset:
            # binary mask
            self.corridor = dataset.dataset_mask() / 255.
            # size of array
            self.raster_size = (dataset.width, dataset.height)
            # geometric bounds and transformation
            self.geo_bounds = dataset.bounds
            self.transform_matrix = dataset.transform

    def compute_class_weights(self):
        """
        From the pandas dataframe with weights, get the classes and
        corresponding weights
        """
        self.layer_classes = []
        self.class_weights = []
        for c in np.unique(self.class_csv["class"]):  # "+str(self.scenario)
            class_rows = self.class_csv[self.class_csv["class"] == c]
            class_weight = np.unique(
                class_rows["category_weight_" + str(self.scenario)]
            )
            assert len(class_weight) == 1, "multiple weights for single class"
            self.layer_classes.append(c)
            self.class_weights.append(class_weight[0])

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

    @padding
    @binarize
    @reduce_instance
    def get_mask_corridor(self):
        return self.corridor

    @padding
    @binarize
    @reduce_instance
    def get_hard_constraints(self):
        """
        Intersection of all "Forbidden"-layers
        """
        hard_cons_rows = self.class_csv[self.class_csv[
            "weight_" + str(self.scenario)] == "Forbidden"]
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
    def get_costs_per_class(self, oneclass=False):
        layers = self.class_csv[
            self.class_csv["weight_" + str(self.scenario)] != "Forbidden"]
        if oneclass:
            self.layer_classes = ["resistance"]
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
            for fname, weight in zip(
                class_r["Layer Name"], class_r["weight_" + str(self.scenario)]
            ):
                file_path = os.path.join(
                    self.path, "tif_layers", fname + ".tif"
                )
                if os.path.exists(file_path):
                    costs = self.read_tif(file_path)
                    # binarize single tif layer so it can be weighted
                    # -1  because in tifs the costly areas are black
                    costs = np.absolute(normalize(costs) - 1)
                    cost_sum_arr[i] = cost_sum_arr[i] + costs * int(weight)
                # else:
                #     print("file not found:", fname)
            # normalize cost surface with all tifs together
            norm_costs = normalize(cost_sum_arr[i])
            # norm_costs[norm_costs == 0] = 0.0001  # cost cannot be zero!
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
        # build rectengular corridor
        start_dest_inds = np.array([start_inds, dest_inds])
        inter_line = start_dest_inds[0] - start_dest_inds[1]
        longer = np.argmin(np.abs(inter_line))
        # define padding size
        padding = [0, 0]
        padding[longer] = abs(int(percent_padding * inter_line[longer]))
        # get four bounds of corridor
        start_x, start_y = np.min(start_dest_inds,
                                  axis=0) - np.asarray(padding)
        end_x, end_y = np.max(start_dest_inds, axis=0) + np.asarray(padding)

        # add rectangle corridor to hard constraints
        corr = np.zeros(hard_constraints.shape)
        corr[start_x:end_x, start_y:end_y] = 1
        hard_constraints = np.asarray(corr * hard_constraints)

        # add emergency points in regular grid
        if emergency_dist is not None:
            max_cost = np.max(instance)
            d = int(emergency_dist // 2)
            tic = time.time()
            # iterate over rectangle
            for i in range(start_x, end_x):
                for j in range(start_y, end_y):
                    # if no point in distance x
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

    @strip
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

        # construct corridor for final data
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
        Save the coordinates in a csv file --> KEEP for GIS programs
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
            "edgecosts": path_costs.tolist(),
            "time_logs": time_logs
        }

        # save as json
        with open(out_path + "_infos.json", "w") as outfile:
            json.dump(out_dict, outfile)

    @staticmethod
    def save_pipeline_infos(
        out_path, output_paths, time_infos, pipeline, scale_factor=1
    ):
        """
        new function to save the information for a whole pipeline
        """
        assert len(pipeline) == len(output_paths), "must be same len"

        out_dict = {"scale": scale_factor, "pipeline": pipeline, "data": []}

        for i, (path, path_costs) in enumerate(output_paths):
            power_path = (np.asarray(path) * scale_factor).tolist()

            data_dict = {
                "path_cells": power_path,
                "edgecosts": path_costs,
                "time_logs": time_infos[i]
            }
            out_dict["data"].append(data_dict)

        # save as json
        with open(out_path + "_infos.json", "w") as outfile:
            json.dump(out_dict, outfile)
