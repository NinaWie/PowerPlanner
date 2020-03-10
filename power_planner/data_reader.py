import os
# from PIL import Image
import rasterio
import shapefile
import numpy as np
import json


class DataReader():

    def __init__(self, base_path, instance_path, scale_factor):
        self.path = base_path
        self.scale_factor = scale_factor

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

    def get_corridor(self):
        self.instance = np.array(self.instance)
        minimum = np.min(self.instance)
        self.instance[self.instance == 9999] = minimum - 1  # TODO: scaling?
        return self._reduce_instance(self.instance)

    def get_cost_surface(self, cost_path):
        # get cost surface
        with rasterio.open(os.path.join(self.path, cost_path), 'r') as ds:
            cost_img = ds.read()[0]
        print("read in cost array", cost_img.shape)
        # cost_img = Image.fromarray(arr[0])
        # cost_img = self._resize_raster(cost_img)
        return self._reduce_instance(np.array(cost_img))

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
        return self._reduce_instance(hard_constraints)

    def get_shape_point(self, start_end_path):
        sf = shapefile.Reader(os.path.join(self.path, start_end_path))
        startendpoint = sf.shapes()[0].points
        transformed = ~self.transform_matrix * startendpoint[0]
        transformed = list(reversed(transformed))  # TODO: why
        resized = np.asarray(transformed) / self.scale_factor
        return resized.astype(int)

    def _resize_raster(self, raster):
        """
        input: Pillow Image!
        """
        if raster.size != self.raster_size:
            print("RESIZING: from", raster.size, "to", self.raster_size)
            raster = raster.resize(self.raster_size, resample=Image.BILINEAR)
        return raster

    def _reduce_instance(self, img):
        """
        Scale down an instance by a factor
        :param img: instance, 2 dim array
        :returns: array of sizes img.size/factor
        """
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
            return new_img
        else:
            return img

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

        # save as json
        with open(out_path + "_infos.json", "w") as outfile:
            json.dump(out_dict, outfile)
