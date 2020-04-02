from power_planner.utils import get_distance_surface, rescale, normalize
import numpy as np
import matplotlib.pyplot as plt
import rasterio


class CorridorUtils():

    def __init__(self):
        pass

    @staticmethod
    def get_middle_line(start_inds, dest_inds, instance_corr, num_points=2):
        vec = (dest_inds - start_inds) / 2
        middle_point = start_inds + vec
        ortho_vec = [-vec[1], vec[0]]
        ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)

        inds_x, inds_y = np.where(instance_corr)
        xs, xe = (inds_x[0], inds_x[-1])
        ys, ye = (inds_y[0], inds_y[-1])
        x, y = tuple(middle_point)
        v1, v2 = tuple(ortho_vec)

        dists_each = min(
            np.absolute(
                [(x - xs) / v1, (xe - x) / v1, (y - ys) / v2, (ye - y) / v2]
            )
        ) / (num_points + 1)

        points = [middle_point.astype(int)]  # start_inds, dest_inds,
        for i in range(num_points):
            points.append(
                (middle_point + ortho_vec * dists_each * (i + 1)).astype(int)
            )
            points.append(
                (middle_point - ortho_vec * dists_each * (i + 1)).astype(int)
            )
        return points

    @staticmethod
    def generate_corridors_middle_line(
        instance_corr, start_inds, dest_inds, num_corrs=5, n_dilate=100
    ):
        num_middle_points = num_corrs // 2
        points = CorridorUtils.get_middle_line(
            start_inds, dest_inds, instance_corr, num_points=num_middle_points
        )
        all_corridors = []
        for p in points:
            path = [[start_inds.tolist(), p.tolist(), dest_inds.tolist()]]
            all_corridors.append(
                get_distance_surface(
                    instance_corr.shape, path, n_dilate=n_dilate
                )
            )
        return all_corridors

    @staticmethod
    def visualize_middle_line(
        all_points, instance_corr, buffer=2, out_path=None
    ):
        example = instance_corr.copy()
        for p in all_points:
            (i, j) = tuple(p)
            example[i - buffer:i + buffer, j - buffer:j + buffer] = 2
        plt.figure(figsize=(20, 10))
        plt.imshow(example)
        if out_path is None:
            plt.show()
        else:
            plt.savefig(out_path + "_corr_lines.png")

    @staticmethod
    def visualize_corrs(corrs, out_path=None):
        plt.figure(figsize=(20, 10))
        for i, corr in enumerate(corrs):
            plt.subplot(1, len(corrs), (i + 1))
            plt.imshow(corr)
        if out_path is not None:
            plt.savefig(out_path + "corridor_quantiles.png")
        else:
            plt.show()

    @staticmethod
    def generate_corridors_from_file(corr_path, nr_corrs=4):
        with rasterio.open(corr_path, 'r') as ds:
            cost_img = ds.read()[0]
        print("read in corridor", cost_img.shape)
        actual_vals = cost_img[cost_img != 9999]

        corrs = []
        cut_val_prev = 0
        log_vals = np.logspace(np.log(0.1), np.log(1), 4, base=2)
        for i in range(4):
            cut_val = np.quantile(actual_vals, log_vals[i])  # (i+1)*0.24)
            copied = cost_img.copy()
            copied[copied < cut_val_prev] = 9999
            copied[copied > cut_val] = 9999
            corrs.append((copied != 9999).astype(int))
            cut_val_prev = cut_val
        return corrs

    @staticmethod
    def get_reduced_patches(
        instance, start_inds, dest_inds, factor, balance=[1, 1], quantile=0.1
    ):
        summed = np.sum(instance, axis=0)
        red = rescale(summed, factor)
        x_len, y_len = red.shape
        path_start_end = [
            [
                (start_inds / factor).astype(int).tolist(),
                (dest_inds / factor).astype(int).tolist()
            ]
        ]
        dist_corr = 1 - normalize(
            get_distance_surface(
                red.shape, path_start_end, n_dilate=min([x_len, y_len])
            )
        )
        surface_comb = balance[0] * dist_corr + balance[1] * red

        quantile_surface = np.quantile(surface_comb, quantile)
        patches = surface_comb < quantile_surface
        plt.imshow(patches.astype(int))
        plt.show()

        inds_x, inds_y = np.where(patches)
        return np.array([inds_x, inds_y])  # *factor

    @staticmethod
    def sample_path(
        instance,
        start_inds,
        dest_inds,
        factor,
        balance=[1, 3],
        quantile=0.1,
        n_sample=4,
        n_onpath=5,
        n_dilate=100
    ):
        out_inds = CorridorUtils.get_reduced_patches(
            instance,
            start_inds,
            dest_inds,
            factor,
            balance=[1, 3],
            quantile=0.1
        )
        # compute distances from start point
        minus_start = [
            np.linalg.norm(out_inds[:, i] - start_inds / factor)
            for i in range(out_inds.shape[1])
        ]
        sorted_patches = np.argsort(minus_start)
        all_corridors = list()
        for _ in range(n_sample):
            drawn_points = np.random.choice(
                np.arange(out_inds.shape[1]), n_onpath, replace=False
            )
            drawn_path = out_inds[:,
                                  sorted_patches[np.
                                                 sort(drawn_points)]] * factor
            path = [
                [start_inds.tolist()] +
                np.swapaxes(drawn_path, 1, 0).tolist() + [dest_inds.tolist()]
            ]
            print(path)
            all_corridors.append(
                get_distance_surface(
                    instance.shape[1:], path, n_dilate=n_dilate
                )
            )
        return all_corridors
