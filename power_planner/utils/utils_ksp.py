import numpy as np
from numba import jit
# from numba.typed import List


@jit(nopython=True)
def compute_eucl(path1, path2, mode="mean"):
    min_dists_out = np.zeros(len(path1))
    for p1 in range(len(path1)):
        min_dists = np.zeros(len(path2))
        for p2 in range(len(path2)):
            min_dists[p2] = np.linalg.norm(path1[p1] - path2[p2])
        min_dists_out[p1] = np.min(min_dists)
    if mode == "mean":
        return np.mean(min_dists_out)
    elif mode == "max":
        return np.max(min_dists_out)


class KspUtils():

    @staticmethod
    def get_sp_from_preds(pred_map, curr_vertex, start_vertex):
        """
        Compute path from start_vertex to curr_vertex from the predecessor map
        Arguments:
            pred_map: map / dictionary with predecessor for each vertex
            curr_vertex: integer denoting any vertex
            start_vertex: integer denoting start vertex
        returns:
            list of vertices (integers)
        """
        path = [int(curr_vertex)]
        counter = 0
        while curr_vertex != start_vertex:
            curr_vertex = pred_map[curr_vertex]
            path.append(curr_vertex)
            if counter > 1000:
                print(path)
                raise RuntimeWarning("while loop for sp not terminating")
            counter += 1
        return path

    @staticmethod
    def path_distance(p1, p2, mode="jaccard"):
        """
        Compute the distance between two paths
        NOTE: all modes in this method are valid metrics
        Arguments:
            p1,p1: two paths (lists of coordinates!)
            mode: jaccard: jaccard distance (1-IoU)
                eucl_mean: from all min eucl distances, take mean
                eucl_max: from all min eucl distances, take max
        """
        if mode == "jaccard":
            s1 = set([tuple(p) for p in p1])
            s2 = set([tuple(p) for p in p2])
            # s1,s2 = (set(list(p1)),set(list(p2)))
            return 1 - len(s1.intersection(s2)) / len(s1.union(s2))
        elif mode.startswith("euc"):
            p1 = np.array(p1).astype("float")
            p2 = np.array(p2).astype("float")
            eucl_mode = mode.split("_")[1]
            return max(
                [
                    compute_eucl(p1, p2, mode=eucl_mode),
                    compute_eucl(p2, p1, mode=eucl_mode)
                ]
            )
        else:
            raise NotImplementedError(
                "mode " + mode + " wrong, not implemented yet"
            )

    @staticmethod
    def similarity(s1, s2, mode="IoU"):
        """
        Implements similarity metrics from Liu et al paper
        Arguments:
            s1,s2: SETS of path points
        """
        path_inter = len(s1.intersection(s2))
        if mode == "IoU":
            return path_inter / len(s1.union(s2))
        elif mode == "sim2paper":
            return path_inter / (2 * len(s1)) + path_inter / (2 * len(s2))
        elif mode == "sim3paper":
            return np.sqrt(path_inter**2 / (len(s1) * len(s2)))
        elif mode == "max_norm_sim":
            return path_inter / (max([len(s1), len(s2)]))
        elif mode == "min_norm_sim":
            return path_inter / (min([len(s1), len(s2)]))
        else:
            raise NotImplementedError("mode wrong, not implemented yet")

    @staticmethod
    def short_eval(ksp):
        """
        compute sum of all costs
        """
        return np.sum([k[2] for k in ksp])

    @staticmethod
    def pairwise_dists(collected_coords, mode="jaccard"):
        nr_paths = len(collected_coords)
        dists = np.zeros((nr_paths, nr_paths))
        for i in range(nr_paths):
            for j in range(i, nr_paths):
                dists[i, j] = KspUtils.path_distance(
                    collected_coords[i], collected_coords[j], mode=mode
                )
                dists[j, i] = dists[i, j]
        return dists

    @staticmethod
    def _flat_ind_to_inds(flat_ind, arr_shape):
        """
        Transforms an index of a flattened 3D array to its original coords
        """
        _, len2, len3 = arr_shape
        x1 = flat_ind // (len2 * len3)
        x2 = (flat_ind % (len2 * len3)) // len3
        x3 = (flat_ind % (len2 * len3)) % len3
        return (x1, x2, x3)

    @staticmethod
    def get_sp_dest_shift(
        dists,
        preds,
        pos2node,
        start_inds,
        dest_inds,
        shifts,
        min_shift,
        dest_edge=False
    ):
        """
        dest_edge: If it's the edge at destination, we cannot take the current
        """
        if not dest_edge:
            dest_ind = pos2node[tuple(dest_inds)]
            min_shift = preds[dest_ind, int(min_shift)]
        curr_point = np.asarray(dest_inds)
        my_path = [dest_inds]
        while np.any(curr_point - start_inds):
            new_point = curr_point - shifts[int(min_shift)]
            pred_ind = pos2node[tuple(new_point)]
            min_shift = preds[pred_ind, int(min_shift)]
            my_path.append(new_point)
            curr_point = new_point
        return my_path

    @staticmethod
    def get_sp_start_shift(
        dists, preds, pos2node, start_inds, dest_inds, shifts, min_shift
    ):
        dest_ind_stack = pos2node[tuple(dest_inds)]
        if not np.any(dists[dest_ind_stack, :] < np.inf):
            raise RuntimeWarning("empty path")
        curr_point = np.asarray(dest_inds)
        my_path = [dest_inds]
        while np.any(curr_point - start_inds):
            new_point = curr_point - shifts[int(min_shift)]
            pred_ind = pos2node[tuple(curr_point)]
            min_shift = preds[pred_ind, int(min_shift)]
            my_path.append(new_point)
            curr_point = new_point
        return my_path

    @staticmethod
    def evaluate_sim(ksp, metric):
        """
        evaluate ksp diversity according to several metric
        """
        ksp_paths = [k[0] for k in ksp]
        # out_diversity = []  # np.zeros((3,2))
        # for k, metric in enumerate(["eucl_mean", "eucl_max", "jaccard"]):
        divs = []
        for i in range(len(ksp_paths)):
            for j in range(i + 1, len(ksp_paths)):
                divs.append(
                    KspUtils.path_distance(
                        ksp_paths[i], ksp_paths[j], mode=metric
                    )
                )
            # out_diversity.append(np.mean(divs))
            # out_diversity[k,1] = np.sum(divs)
        # return out_diversity
        return np.mean(divs)

    @staticmethod
    def evaluate_cost(ksp):
        """
        Evaluate ksp with respect to the overall and maximal costs
        """
        # ksp_path_costs = [k[1] for k in ksp]
        # for p_cost in ksp_path_costs:
        #     p = np.asarray(p_cost)
        #     c_m = np.mean(np.sum(p,axis=1))
        ksp_all_costs = [k[2] for k in ksp]
        return [np.sum(ksp_all_costs), np.max(ksp_all_costs)]

    # @staticmethod
    # def compare_ksp():
    #     """
    #     METHOD UNUSABLE HERE; JUST BACKUP FROM NOTEBOOK
    #     """
    #     res_dict = defaultdict(list)
    #     metrics = ["eucl_mean", "eucl_max", "jaccard"]

    #     func_eval = [
    #         eucl_max, eucl_max, eucl_max, eucl_max, currently_implemented,
    #         currently_implemented, currently_implemented,
    #         currently_implemented, most_diverse_jaccard, most_diverse_jaccard,
    #         most_diverse_eucl_max, most_diverse_eucl_max, laplace, laplace,
    #         laplace, laplace
    #     ]
    #     names = [
    #         "vertex eucl max", "vertex eucl max", "vertex eucl max",
    #         "vertex eucl max", "greedy max set", "greedy max set",
    #         "greedy max set", "greedy max set", "diverse jaccard",
    #         "diverse jaccard", "diverse eucl max", "diverse eucl max",
    #         "corridor", "corridor", "corridor", "corridor"
    #     ]
    #     thresh_eval = [
    #         6, 8, 12, 14, 0.4, 0.6, 0.8, 1.0, 1.01, 1.05, 1.01, 1.05, 7, 10,
    #         15, 20
    #     ]
    #     assert len(func_eval) == len(names)
    #     assert len(names) == len(thresh_eval)

    #     all_ksps = []
    #     for name, func, param in zip(names, func_eval, thresh_eval):
    #         ksp, tic = func(graph, start_inds, dest_inds, 5, param)
    #         all_ksps.append(ksp)
    #         res_dict["name"].append(name)
    #         res_dict["threshold"].append(param)
    #         res_dict["times"].append(tic)
    #         for m in metrics:
    #             res_dict[m + "_distance"].append(evaluate_sim(ksp, m))
    #         cost_sum, cost_max = evaluate_cost(ksp)
    #         res_dict["cost_sum"].append(cost_sum)
    #         res_dict["cost_max"].append(cost_max)
