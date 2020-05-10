import numpy as np
from numba import jit
from numba.typed import List


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


@jit(nopython=True)
def add_out_edges(stack, shifts, angles_all, dists, instance):
    """
    Compute cumulative distances with each point of dists containing OUT edges
    """
    preds = np.zeros(dists.shape)
    preds = preds - 1
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[-i - 1][0]
        v_y = stack[-i - 1][1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if 0 <= neigh_x < dists.shape[1] and 0 <= neigh_y < dists.shape[2]:
                cost_per_angle = np.zeros(len(shifts))
                for s2 in range(len(shifts)):
                    in_neigh_x = v_x - shifts[s2][0]
                    in_neigh_y = v_y - shifts[s2][1]
                    cost_per_angle[
                        s2] = dists[s2, in_neigh_x, in_neigh_y
                                    ] + angles_all[s, s2] + instance[v_x, v_y]
                dists[s, v_x, v_y] = np.min(cost_per_angle)
                preds[s, v_x, v_y] = np.argmin(cost_per_angle)
    return dists, preds


@jit(nopython=True)
def get_sp_start_shift(
    dists, dists_argmin, start_inds, dest_inds, shifts, min_shift
):
    if not np.any(dists[:, dest_inds[0], dest_inds[1]] < np.inf):
        raise RuntimeWarning("empty path")
    curr_point = np.asarray(dest_inds)
    my_path = List()
    tmp_list_inner = List()
    tmp_list_inner.append(dest_inds[0])
    tmp_list_inner.append(dest_inds[1])
    my_path.append(tmp_list_inner)
    # min_shift = np.argmin(dists[:, dest_inds[0], dest_inds[1]])
    while np.any(curr_point - start_inds):
        new_point = curr_point - shifts[int(min_shift)]
        min_shift = dists_argmin[int(min_shift), curr_point[0], curr_point[1]]
        my_path.append(List(new_point))
        curr_point = new_point
    return my_path


@jit(nopython=True)
def get_sp_dest_shift(
    dists, dists_argmin, start_inds, dest_inds, shifts, min_shift
):
    if not np.any(dists[:, dest_inds[0], dest_inds[1]] < np.inf):
        raise RuntimeWarning("empty path")
    min_shift = dists_argmin[int(min_shift), dest_inds[0], dest_inds[1]]
    curr_point = np.asarray(dest_inds)
    my_path = List()
    tmp_list_inner = List()
    tmp_list_inner.append(dest_inds[0])
    tmp_list_inner.append(dest_inds[1])
    my_path.append(tmp_list_inner)
    # min_shift = np.argmin(dists[:, dest_inds[0], dest_inds[1]])
    while np.any(curr_point - start_inds):
        new_point = curr_point - shifts[int(min_shift)]
        min_shift = dists_argmin[int(min_shift), new_point[0], new_point[1]]
        my_path.append(List(new_point))
        curr_point = new_point
    return my_path


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
