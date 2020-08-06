import numpy as np
import pandas as pd
from power_planner.utils.utils_costs import CostUtils

# ---------------------------------------------------------------------
# Compute raw (unnormalized) costs and output csv


def raw_path_costs(cost_instance, path, edge_instance, heights=None):
    """
    Compute raw angles, edge costs, pylon heights and normal costs
    (without weighting)
    Arguments:
        List or array of path coordinates
    """
    path_costs = np.array([cost_instance[:, p[0], p[1]] for p in path])
    # raw angle costs
    ang_costs = CostUtils.compute_raw_angles(path)
    # raw edge costs
    if edge_instance is None:
        edge_instance = cost_instance
    edge_costs = CostUtils.compute_edge_costs(path, edge_instance)
    # pylon heights
    if heights is not None:
        heights = np.expand_dims(heights, 1)
    else:
        heights = np.zeros((len(edge_costs), 1))
    # concatenate
    all_costs = np.concatenate(
        (
            np.expand_dims(ang_costs, 1), path_costs,
            np.expand_dims(edge_costs, 1), heights
        ), 1
    )
    # names = self.cost_classes + ["edge_costs", "heigths"]
    # assert all_costs.shape[1] == len(names)
    return all_costs


def save_path_cost_csv(
    save_path,
    paths,
    cost_instance,
    edge_instance=None,
    heights=None,
    big_inst=None,
    class_weights=[1],
    **kwargs
):
    """
    save coordinates in original instance (tifs) without padding etc
    """
    # shifts caused by scaling
    if edge_instance is None:
        normed_class_weights = np.array(class_weights) / np.sum(class_weights)
        edge_instance = np.sum(
            np.moveaxis(cost_instance, 0, -1) * normed_class_weights, axis=2
        )
    scale = kwargs["scale"]
    start_shift_x, start_shift_y = (
        kwargs["orig_start"][0] % scale, kwargs["orig_start"][1] % scale
    )
    dest_shift_x, dest_shift_y = (
        kwargs["orig_dest"][0] % scale, kwargs["orig_dest"][1] % scale
    )
    # out_path_list = []
    for i, path in enumerate(paths):
        # compute raw costs and column names
        names = ["angle costs"
                 ] + list(class_weights) + ["edge costs", "heights"]
        raw_cost = raw_path_costs(
            cost_instance, path, edge_instance=edge_instance, heights=heights
        )
        # round raw costs of this particular path
        raw_costs = np.around(raw_cost, 2)
        # compute_shift shift
        path = np.asarray(path)
        shift_to_orig = (kwargs["orig_start"] / scale).astype(int) - path[0]
        print("shift", shift_to_orig)
        print(
            "correct start ", kwargs["orig_start"], start_shift_x,
            start_shift_y
        )
        print("correct dest ", kwargs["orig_dest"], dest_shift_x, dest_shift_y)
        shifted_path = path + shift_to_orig
        power_path = shifted_path * scale

        # correct the start and end
        power_path[0] = power_path[0] + [start_shift_x, start_shift_y]
        power_path[-1] = power_path[-1] + [dest_shift_x, dest_shift_y]

        if big_inst is not None:
            new_path = []
            for k, (i, j) in enumerate(power_path):
                # skip start and dest
                if k == 0 or k == len(power_path) - 1:
                    new_path.append([i, j])
                    continue
                check_patch = big_inst[i:i + scale, j:j + scale]
                min_x, min_y = np.where(check_patch == np.min(check_patch))
                new_path.append([i + min_x[0], j + min_y[0]])
                print("prev:", i, j, "new", [i + min_x[0], j + min_y[0]])
            power_path = np.asarray(new_path)

        # scaled_path = np.asarray(path) * kwargs["scale"]
        # shift_to_orig = kwargs["orig_start"] - scaled_path[0]
        # power_path = scaled_path + shift_to_orig
        # out_path_list.append(shifted_path.tolist())

        coordinates = [kwargs["transform_matrix"] * p for p in power_path]

        all_coords = np.concatenate(
            (coordinates, power_path, raw_costs), axis=1
        )
        df = pd.DataFrame(
            all_coords, columns=["X", "Y", "X_raw", "Y_raw"] + names
        )
        if len(paths) == 1:
            df.to_csv(save_path + ".csv", index=False)
        else:
            df.to_csv(save_path + "_" + str(i) + ".csv", index=False)
