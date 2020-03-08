import os
from PIL import Image
import rasterio
import numpy as np
import json


class DataReader():

    def __init__(self, base_path, instance_path):
        self.path = base_path

        with rasterio.open(os.path.join(base_path, instance_path)) as dataset:
            # binary mask
            self.corridor = dataset.dataset_mask()
            # size of array
            self.raster_size = (dataset.width, dataset.height)
            # geometric bounds and transformation
            self.geo_bounds = dataset.bounds
            self.transform_matrix = dataset.transform
            #  array itself
            self.instance = dataset.read()[0]

    def read_in_tifs(self, path=None):
        if path is None:
            path = self.path
        files = os.listdir(path)
        tif_list = []
        file_list = []
        for f in files:
            if f[-3:] == "tif":
                img = Image.open(os.path.join(path, f))
                img = self._resize_raster(img)
                tif_list.append(np.array(img))
                file_list.append(f[:-4])
        tif_arr = np.array(tif_list)
        tif_arr = tif_arr / 255.
        print("shape of tif array:", tif_arr.shape)
        return tif_arr, file_list

    def get_corridor(self):
        self.instance = np.array(self.instance)
        minimum = np.min(self.instance)
        self.instance[self.instance == 9999] = minimum - 1  # TODO: scaling?
        return self.instance

    def get_cost_surface(self, cost_path):
        # get cost surface
        with rasterio.open(os.path.join(self.path, cost_path), 'r') as ds:
            arr = ds.read()
        print("read in cost array", arr.shape)
        cost_img = Image.fromarray(arr[0])
        cost_img = self._resize_raster(cost_img)
        return np.array(cost_img)

    def get_hard_constraints(self, hard_cons_file):
        # get hard constraints
        hard_cons_path = os.path.join(self.path, hard_cons_file)

        hard_cons_arr, _ = self.read_in_tifs(path=hard_cons_path)
        # concatenate with corridor
        hard_cons_arr = np.concatenate(
            (hard_cons_arr, np.expand_dims(self.corridor, axis=0)), axis=0
        )
        # logical and between all hard constraints
        hard_constraints = np.all(hard_cons_arr.astype(int), axis=0)
        return hard_constraints

    def _resize_raster(self, raster):
        if raster.size != self.raster_size:
            print("RESIZING: from", raster.size, "to", self.raster_size)
            raster = raster.resize(self.raster_size, resample=Image.BILINEAR)
        return raster

    def save_json(self, power_path, out_path, scale_factor):
        """
        Save the path as a json file:
        @param power_path: List of path indices [[x1, y1], [x2,y2] ...]
        @patam out_path: path and filename (without .json) where to write to
        @param scale_factor: if the instance was scaled down, 
        the coordinates have to be scaled up again
        """
        power_path = (np.asarray(power_path) * scale_factor).tolist()

        # save pixel value path
        with open(out_path + ".json", "w") as outfile:
            json.dump(power_path, outfile)

        coordinates = [self.transform_matrix * p for p in power_path]
        # save coordinates
        with open(out_path + "_coordinates.json", "w") as outfile:
            json.dump(coordinates, outfile)
