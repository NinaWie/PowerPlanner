import os
# from PIL import Image
import rasterio
import shapefile
import numpy as np
import json
import pandas as pd
import functools

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
        img = func(self, *args, **kwargs)
        if self.scale_factor > 1:
            x_len_new = img.shape[0] // self.scale_factor
            y_len_new = img.shape[1] // self.scale_factor
            new_img = np.zeros((x_len_new, y_len_new))
            for i in range(x_len_new):
                for j in range(y_len_new):
                    patch = img[i * self.scale_factor:(i + 1) *
                                self.scale_factor, j *
                                self.scale_factor:(j + 1) * self.scale_factor]
                    new_img[i, j] = np.mean(patch)
            return np.swapaxes(new_img, 1, 0)  # TODO
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
        hard_constraints = np.asarray(
            [
                self.read_tif(
                    os.path.join(self.path, "tif_layers", fname + ".tif")
                ) for fname in hard_cons_rows["Layer Name"]
            ]
        )
        # set to zero
        hard_constraints -= np.min(hard_constraints)
        # intersection of all of the hard constraints
        hard_constraints = np.all(hard_constraints.astype(int), axis=0)
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
        return normalize(cost_sum_arr)

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
        if raster.size != self.raster_size:
            print("RESIZING: from", raster.size, "to", self.raster_size)
            raster = raster.resize(self.raster_size, resample=Image.BILINEAR)
        return raster

    def get_data(self):
        """
        Get all data at once: intersection of hard constraints and return
        weighted sum of all layers as cost
        :returns: the cost surface, and the corridor, a binary showing possible
        tower locations of the same shape
        """
        # # corridor workaround
        # corr = np.zeros((1313, 1511))
        # corr[40:1260, 200:1000] = 1
        # data.instance = corr
        # data.corridor = corr
        # padding: not necessary right now
        # data.set_padding(PYLON_DIST_MAX * SCALE_PARAM)
        corridor = self.get_mask_corridor()
        hard_constraints = self.get_hard_constraints()
        instance_corr = corridor * hard_constraints
        instance = self.get_weighted_costs()
        return instance, instance_corr

    def save_json(self, power_path, out_path, scale_factor=1, time_logs={}):
        """
        Save the path as a json file:
        @param power_path: List of path indices [[x1, y1], [x2,y2] ...]
        @patam out_path: path and filename (without .json) where to write to
        @param scale_factor: if the instance was scaled down, 
        the coordinates have to be scaled up again
        """
        power_path = (np.asarray(power_path) * scale_factor).tolist()
        coordinates = [self.transform_matrix * p for p in power_path]

        out_dict = {
            "path_cells": power_path,
            "path_coordinates": coordinates,
            "time_logs": time_logs
        }

        df = pd.DataFrame(np.asarray(coordinates), columns=["X", "Y"])
        df.to_csv(out_path + "_coords.csv", index=False)

        # save as json
        with open(out_path + "_infos.json", "w") as outfile:
            json.dump(out_dict, outfile)
