import os
from PIL import Image
import rasterio
import numpy as np


class DataReader():

    def __init__(self, base_path):
        self.path = base_path

    def read_in_tifs(self, path=None):
        if path is None:
            path = self.path
        files = os.listdir(path)
        tif_list = []
        file_list = []
        for f in files:
            if f[-3:] == "tif":
                img = Image.open(os.path.join(path, f))
                tif_list.append(np.array(img))
                file_list.append(f[:-4])
        tif_arr = np.array(tif_list)
        tif_arr = tif_arr / 255.
        print("shape of tif array:", tif_arr.shape)
        return tif_arr, file_list

    def get_corridor(self, corr_path, out_shape=(3022, 2627)):
        with rasterio.open(os.path.join(self.path, corr_path), 'r') as ds:
            arr = ds.read()
        corr_img = Image.fromarray(arr[0])
        corr_resized = corr_img.resize(out_shape, resample=Image.BILINEAR)
        corridor = (np.array(corr_resized) < 9900).astype(int)
        # plt.imshow(corridor)
        # plt.colorbar()
        # plt.show()
        return corridor

    def get_cost_surface(self, cost_path, out_shape=(3022, 2627)):
        # get cost surface
        with rasterio.open(os.path.join(self.path, cost_path), 'r') as ds:
            arr = ds.read()
        cost_img = Image.fromarray(arr[0])
        print("read in cost array", arr.shape)
        costs = cost_img.resize(out_shape, resample=Image.BILINEAR)
        # normalize(costs)
        return np.array(costs)

    def get_hard_constraints(self, corrfile, hard_cons_file):
        # get hard constraints
        corridor_path = os.path.join(self.path, corrfile)
        hard_cons_path = os.path.join(self.path, hard_cons_file)

        corridor = self.get_corridor(corridor_path)
        hard_cons_arr, _ = self.read_in_tifs(path=hard_cons_path)
        # concatenate with corridor
        hard_cons_arr = np.concatenate(
            (hard_cons_arr, np.expand_dims(corridor, axis=0)), axis=0
        )
        # logical and between all hard constraints
        hard_constraints = np.all(hard_cons_arr.astype(int), axis=0)
        return hard_constraints
