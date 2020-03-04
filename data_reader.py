import os
from PIL import Image
import rasterio
import numpy as np


def read_in_tifs(path):
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


def get_corridor(path, fn="Corridor_BE.tif"):
    with rasterio.open(os.path.join(path, fn), 'r') as ds:
        arr = ds.read()
    corr_img = Image.fromarray(arr[0])
    corr_resized = corr_img.resize((3022, 2627), resample=Image.BILINEAR)
    corridor = (np.array(corr_resized) < 9900).astype(int)
    # plt.imshow(corridor)
    # plt.colorbar()
    # plt.show()
    return corridor


def get_hard_constraints(corridor_path, hard_cons_path):
    # get hard constraints
    corridor = get_corridor(corridor_path)
    hard_cons_arr, _ = read_in_tifs(hard_cons_path)
    # concatenate with corridor
    hard_cons_arr = np.concatenate(
        (hard_cons_arr, np.expand_dims(corridor, axis=0)), axis=0
    )
    print(hard_cons_arr.shape)
    # logical and between all hard constraints
    hard_constraints = np.all(hard_cons_arr.astype(int), axis=0)
    return hard_constraints
