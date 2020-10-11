import os
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
        instance, edge_inst, instance_corr, config = func(*args, **kwargs)
        start_inds = config.graph.start_inds
        dest_inds = config.graph.dest_inds

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
        edge_inst = edge_inst[:, up:down, left:right]
        instance_corr = instance_corr[up:down, left:right]
        config.graph.start_inds = start_inds - np.array([up, left])
        config.graph.dest_inds = dest_inds - np.array([up, left])

        return instance, edge_inst, instance_corr, config

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

    def __init__(self, base_path, scenario, scale_factor, config):
        self.path = base_path
        self.scale_factor = scale_factor
        # config file: here, only the data configurations are relevant
        # the other ones are still passed to be returned
        self.general_config = config
        self.config = config.data
        # read csv file with resistances
        weights = pd.read_csv(os.path.join(base_path, self.config.weight_csv))
        self.class_csv = weights.dropna()
        if len(weights) < 1:
            raise ValueError("layer weights csv file empty")
        self.padding = 0
        self.scenario = scenario
        # get classes and corresponding weights from csv
        self.compute_class_weights()

        # load project region as the main dataset
        with rasterio.open(
            os.path.join(base_path, self.config.corr_path)
        ) as dataset:
            # binary mask
            layer = dataset.read()[0]
            self.corridor = (layer == np.max(layer)).astype(int)
            # size of array
            self.raster_size = self.corridor.shape
            # (dataset.width, dataset.height)
            # geometric bounds and transformation
            self.geo_bounds = dataset.bounds
            self.transform_matrix = dataset.transform
        self.general_config.graph.transform_matrix = dataset.transform
        self.general_config.graph.scale = scale_factor

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
            assert self.config.one_class or len(
                class_weight
            ) == 1, "multiple weights for single class"
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
    def get_hard_constraints(self, is_edge=False):
        """
        Intersection of all "Forbidden"-layers
        """
        if is_edge:
            hard_cons_rows = self.class_csv[self.class_csv[
                "weight_" + str(self.scenario) + "_edge"] == "Forbidden"]
        else:
            hard_cons_rows = self.class_csv[self.class_csv[
                "weight_" + str(self.scenario)] == "Forbidden"]

        # read in corresponding tifs
        hard_constraints = []
        for fname in hard_cons_rows["Layer Name"]:
            file_path = os.path.join(self.path, "tif_layers", fname + ".tif")
            if os.path.exists(file_path):
                constraint = self.read_tif(file_path)
                constraint = self._resize_raster(constraint)
                # print(fname, constraint.shape)
                if "Scenario" in fname:
                    hard_constraints.append((constraint == 1).astype(int))
                else:
                    hard_constraints.append((constraint != 1).astype(int))
                # constraint.astype(int) > 0.5 * np.max(constraint)
            else:
                print("forbidden layer does not exist", fname)
        print("hard constraints shape", np.asarray(hard_constraints).shape)
        # no forbidden areas - return all ones
        if len(hard_constraints) == 0:
            print("default: no forbidden areas, all 1")
            return np.ones(self.raster_size)
        # intersection of all of the hard constraints
        hard_constraints = np.all(
            np.asarray(hard_constraints).astype(int), axis=0
        )
        return hard_constraints

    @padding
    @reduce_instance
    def get_costs_per_class(self, oneclass=False, is_edge=False):
        # edge instance
        if is_edge:
            weight_column = "weight_" + str(self.scenario) + "_edge"
        # normal (pylon) instance
        else:
            weight_column = "weight_" + str(self.scenario)
        layers = self.class_csv[self.class_csv[weight_column] != "Forbidden"]
        if oneclass:
            self.layer_classes = ["resistance"]
            self.class_weights = [1]
        cost_sum_arr = np.zeros(
            tuple([len(self.layer_classes)]) + self.raster_size
        )
        for i, classname in enumerate(self.layer_classes):
            if oneclass:
                class_r = layers
            else:
                class_r = layers[layers["class"] == classname]
            # Get corresponding weights and class weights
            r_weights = class_r[weight_column].values
            c_weights = class_r["category_weight_" + str(self.scenario)].values
            for fname, weight, cat_w in zip(
                class_r["Layer Name"], r_weights, c_weights
            ):
                file_path = os.path.join(
                    self.path, "tif_layers", fname + ".tif"
                )
                if os.path.exists(file_path):
                    costs_raw = self.read_tif(file_path)
                    # binarize single tif layer so it can be weighted
                    # -1  because in tifs the costly areas are black
                    # costs = np.absolute(normalize(costs) - 1)
                    costs = (costs_raw == 1).astype(int)
                    if not np.any(costs):
                        print("all zero layer", fname, np.unique(costs_raw))
                    # two options: sum up per class costs,normalize and weight
                    # classes, or directly multiply by class weight
                    if oneclass:
                        cost_sum_arr[
                            i] = cost_sum_arr[i] + costs * int(weight) * cat_w
                    else:
                        cost_sum_arr[i] = cost_sum_arr[i] + costs * int(weight)
                else:
                    print("file not found:", fname)
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
        return resized.astype(int) + self.padding, np.asarray(transformed
                                                              ).astype(int)

    def _resize_raster(self, raster):
        """
        input: Pillow Image!
        """
        # if list(reversed(raster.shape)) != list(self.raster_size):
        #     raster = Image.fromarray(raster)
        #     print("resize: from", raster.size, "to", self.raster_size)
        #     raster = raster.resize(self.raster_size, resample=Image.BILINEAR)
        #     raster = np.array(raster)  # swapaxes(raster, 1, 0)
        if list(raster.shape) != list(self.raster_size):
            return np.swapaxes(raster, 1, 0)
        return raster

    @staticmethod
    def construct_corridor(
        instance,
        hard_constraints,
        start_inds,
        dest_inds,
        emergency_dist=-1,
        percent_padding=-1
    ):
        """
        construct a rectengular corridor that is relevant for path computation
        Arguments:
            instance: 3D array of resistance values
            hard_constraints: 2D array of forbidden areas
            start_inds, dest_inds: tuples/lists with two entries (x&y coord)
            emergency_dist: distance in which emergency points should be
                placed in forbidden regions
            percent_padding: padding around start-dest-line distance
        """
        # build rectengular corridor
        start_dest_inds = np.array([start_inds, dest_inds])
        inter_line = start_dest_inds[0] - start_dest_inds[1]
        longer = np.argmin(np.abs(inter_line))
        # define padding size - if -1, then no padding
        if percent_padding != -1:
            padding = [0, 0]
            padding[longer] = abs(int(percent_padding * inter_line[longer]))
            # get four bounds of corridor
            start_x, start_y = np.min(start_dest_inds,
                                      axis=0) - np.asarray(padding)
            end_x, end_y = np.max(start_dest_inds,
                                  axis=0) + np.asarray(padding)
        else:
            # if None, only cut off a small frame around everything
            start_x, start_y = np.min(start_dest_inds, axis=0)
            end_x, end_y = np.max(start_dest_inds, axis=0)
            x_len, y_len = hard_constraints.shape
            min_startend = np.min(
                [start_x, start_y, x_len - end_x, y_len - end_y]
            )
            # TODO!!
            if instance.shape[0] == 3078 or instance.shape[1] == 3078:
                print("reset here")
                min_startend = int(min_startend / 3.5)
                print(min_startend)
            start_x, start_y = (min_startend, min_startend)
            end_x, end_y = (x_len - min_startend, y_len - min_startend)

        # add rectangle corridor to hard constraints
        corr = np.zeros(hard_constraints.shape)
        corr[start_x:end_x + 1, start_y:end_y + 1] = 1
        hard_constraints = np.asarray(corr * hard_constraints)

        # add emergency points in regular grid
        if emergency_dist > 0:
            max_cost = np.max(instance)
            d = int(emergency_dist // 2)
            tic = time.time()
            # iterate over rectangle
            for i in range(start_x, end_x + 1):
                for j in range(start_y, end_y + 1):
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
    def get_data(self):
        """
        Get all data at once: intersection of hard constraints and return
        weighted sum of all layers as cost
        :returns: the cost surface, and the corridor, a binary showing possible
        tower locations of the same shape
        """
        # # corridor defined manually
        # padding: not necessary right now
        # data.set_padding(PYLON_DIST_MAX * SCALE_PARAM)

        # Construct hard constraints (infinity cost regions)
        project_region = self.get_mask_corridor()
        hard_cons = self.get_hard_constraints()
        hard_constraints = project_region * hard_cons

        print("read forbidden areas:", project_region.shape, hard_cons.shape)

        # Construct instance and edge instance
        instance = self.get_costs_per_class(oneclass=self.config.one_class)
        print("read instance:", instance.shape)
        if "weight_" + str(self.scenario) + "_edge" in self.class_csv.columns:
            print("EDGE COL EXISTS --> constructing edge instance")
            edge_inst = self.get_costs_per_class(
                oneclass=self.config.one_class, is_edge=True
            )
            edge_corr = self.get_hard_constraints(
                is_edge=True
            ) * project_region
            edge_corr_inf = (edge_corr == 0).astype(float)
            edge_corr_inf[edge_corr_inf > 0] = np.inf
            edge_inst = edge_inst + edge_corr_inf
        elif self.config.cable_forbidden:
            print("Ueberspannen forbidden!")
            # don't take separate weights for edge instance,
            # but forbid ueberspannen of hard constraint regions
            inf_corr = (hard_constraints == 0).astype(float)
            # set the forbidden regions to infinity cost
            inf_corr[inf_corr > 0] = np.inf
            edge_inst = instance.copy() + inf_corr
        else:
            print("ueberspannen okay and edge inst is normal instance")
            # allow ueberspannen of hard constraint regions
            edge_inst = instance.copy()
        print("shape of inst and corr", instance.shape, hard_constraints.shape)

        # Get start and end point and save in config
        start_inds, self.orig_start = self.get_shape_point(
            self.config.start_path
        )
        dest_inds, self.orig_dest = self.get_shape_point(self.config.dest_path)
        self.general_config.graph.orig_start = self.orig_start
        self.general_config.graph.orig_dest = self.orig_dest
        self.general_config.graph.dest_inds = dest_inds
        self.general_config.graph.start_inds = start_inds
        print("start cells:", start_inds, "dest cells:", dest_inds)
        print(
            "orig start cells:", self.orig_start, "orig dest cells:",
            self.orig_dest
        )
        # add classes and weights to config:
        self.general_config.graph.layer_classes = self.layer_classes
        self.general_config.graph.class_weights = self.class_weights

        # construct corridor for final data
        instance, hard_constraints = self.construct_corridor(
            instance,
            hard_constraints,
            start_inds,
            dest_inds,
            emergency_dist=self.config.emergency_dist /
            (self.config.raster * self.scale_factor),
            percent_padding=self.config.perc_pad
        )
        # percent_padding: 5 for large instance, 0.25 small
        return instance, edge_inst, hard_constraints, self.general_config

    @staticmethod
    def get_raw_data(layer_path, csv_path, scenario=1):
        layer_list = pd.read_csv(csv_path).dropna()
        layer_arr = []
        layer_weights, layer_names, layer_classes = [], [], []
        forb_arr = []
        for i, row in layer_list.iterrows():
            file_path = os.path.join(layer_path, row["Layer Name"] + ".tif")
            if os.path.exists(file_path):
                with rasterio.open(file_path, 'r') as ds:
                    arr = ds.read()[0]
                # binarize single tif layer so it can be weighted
                # -1  because in tifs the costly areas are bla
                # add to hard constraints or general instance
                if row["weight_" + str(scenario)] == "Forbidden":
                    constraint = (arr.astype(int) != 1).astype(int)
                    forb_arr.append(constraint)
                    print(constraint.shape)
                else:
                    costs = (arr == 1).astype(int)
                    # costs = np.absolute(normalize(arr) - 1)
                    layer_arr.append(costs)
                    layer_weights.append(
                        int(row["weight_" + str(scenario)]) *
                        int(row["category_weight_" + str(scenario)])
                    )
                    layer_classes.append(row["class"])
                    layer_names.append(
                        row["Layer Name"]
                    )  # ["Corresponding Name"])
            else:
                print("file not found:", row["Layer Name"])
        df = pd.DataFrame()
        df["weights"] = layer_weights
        df["arr_inds"] = [i for i in range(len(layer_arr))]
        df["class"] = layer_classes
        df["layer"] = layer_names
        layer_arr = np.asarray(layer_arr)
        if len(forb_arr) == 0:
            forb_arr = np.ones(layer_arr.shape)
        return np.swapaxes(layer_arr, 2,
                           1), np.swapaxes(np.array(forb_arr), 2, 1), df

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

    @staticmethod
    def save_paths_json(paths, scale_factor, save_path):
        out_path_list = []
        for _, power_path in enumerate(paths):
            out_path_list.append(
                (np.asarray(power_path) * scale_factor).tolist()
            )
        # save as json
        with open(save_path + "_coords.json", "w") as outfile:
            json.dump(out_path_list, outfile)
