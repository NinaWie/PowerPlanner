import numpy as np
from numba import jit


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
        Compute path from start_vertex to curr_vertex form the predecessor map 
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
