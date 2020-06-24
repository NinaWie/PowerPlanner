import numpy as np
import rasterio
import time
from numba import jit
from power_planner.graphs.implicit_lg import topological_sort_jit, ImplicitLG
from power_planner.utils.utils import rescale

CAT_H = 6300
CAT_W = 3.589


@jit(nopython=True)
def catenary_y(x):
    return (CAT_W * x**2) / (2 * CAT_H)


@jit(nopython=True)
def check_edge_heights(
    stack, shifts, height_resistance, shift_lines, height_arr, MIN_H, MAX_H,
    RESOLUTION
):
    """
    Check all edges and output an array indicating which ones are
    0 - okay at minimum pylon height, 2 - forbidden, 1 - to be computed
    NOTE: function not used here! only for test purposes
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[-i - 1][0]
        v_y = stack[-i - 1][1]

        # so far height on in edges
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]

            # get minimum heights of v_x,v_y dependent on incoming edge
            bres_line = shift_lines[s] + np.array([v_x, v_y])

            # required heights
            S = int(
                np.sqrt((v_x - neigh_x)**2 + (v_y - neigh_y)**2)
            ) * RESOLUTION

            # left and right point
            yA = height_resistance[v_x, v_y] + MIN_H
            yB = height_resistance[neigh_x, neigh_y] + MIN_H

            # compute lowest point of sag
            x0 = S / 2 - ((yB - yA) * CAT_H / (CAT_W * S))
            # compute height above x0 at left point
            A_height = (CAT_W * x0**2) / (2 * CAT_H)
            # print(height_bline)

            # iterate over points on bres_line
            stepsize = S / (len(bres_line) + 1)
            heights_above = np.zeros(len(bres_line))
            for k, (i, j) in enumerate(bres_line):
                x = x0 - stepsize * (k + 1)
                cat = (CAT_W * x**2) / (2 * CAT_H)
                heights_above[k
                              ] = yA - A_height - height_resistance[i, j] + cat

            # analyse heights_above:
            if np.all(heights_above >= 11):
                # whole cable is okay
                fine_60 = 0
            elif np.any(heights_above < -MAX_H - MIN_H):
                # would not work with 80 - 80
                fine_60 = 2
            else:
                # somewhere inbetween
                fine_60 = 1
            height_arr[s, neigh_x, neigh_y] = fine_60

    return height_arr


@jit(nopython=True)
def min_heights(height_bline, dists_bline, S, MIN_H, MAX_H, nr_steps=5):
    """
    convolve with cable over height profile
    """
    # define height profile of start and dest
    yB = height_bline[-1]
    yA = height_bline[0]
    # left tower 80, right tower 60 and other way round
    x0_right = S / 2 - ((yB + MIN_H - yA - MAX_H) * CAT_H / (CAT_W * S))
    x0_left = S / 2 - ((yB + MAX_H - yA - MIN_H) * CAT_H / (CAT_W * S))
    # stepsize: all possible positions for x0 / nr_steps
    stepsize = (x0_right - x0_left) // (nr_steps - 1)
    height_combinations = np.zeros((nr_steps, 2))
    for s in range(nr_steps):
        start = int(s * stepsize - x0_right)
        # base height
        base_height = catenary_y(start)
        # compute heights of the catenary
        heights_catenary = np.zeros(len(dists_bline))
        heights_above = np.zeros(len(dists_bline))
        for k in range(len(dists_bline)):
            cat = catenary_y(dists_bline[k] + start) - base_height
            heights_catenary[k] = cat
            heights_above[k] = height_bline[k] - cat
        min_uplift = 11 + np.max(heights_above)
        # the zero point x0 must be at (absolute above sea) height min_uplift.
        # thus, pylon A must be at min_uplift + heights_above[0],
        # and then subtract to get the added part
        min_height_A = min_uplift + heights_catenary[0] - yA
        min_height_B = min_uplift + heights_catenary[-1] - yB
        if min_height_A > MAX_H or min_height_B > MAX_H:
            min_height_A = np.inf
            min_height_B = np.inf
        if min_height_A < MIN_H:
            min_height_A = MIN_H
        if min_height_B < MIN_H:
            min_height_B = MIN_H
        # print(start, min_height_A, min_height_B)
        # # CHECK:
        # actual_h = min_height_B + yB - min_height_A - yA
        # actual_xr = lowest_point(S, actual_h)
        # print("ACTUAL")
        # print(actual_h, actual_xr)
        # cat_heights =
        # [catenary_y(d-actual_xr) for i, d in enumerate(dists_bline)]
        # # check:
        # print([cat_h + min_height_A - cat_heights[0] + height_bline[0] -
        # height_bline[i] for i,cat_h in enumerate(cat_heights)])
        height_combinations[s, 0] = min_height_A
        height_combinations[s, 1] = min_height_B
    return height_combinations


@jit(nopython=True)
def select_best_height(heights_in, height_combinations, MIN_H, MAX_H):
    """
    For each incoming edge, compute the optimal edge heights
    """
    best_height_costs = np.zeros(len(heights_in))
    hin_new = np.zeros(len(heights_in))
    hout_new = np.zeros(len(heights_in))
    # TODO: np.all(heights_in==60)
    for h in range(len(heights_in)):
        h_in = heights_in[h]
        if h_in < MIN_H:
            # problem: for unseen edges,
            # the in height is zero (from height_arr).
            # circumvent here by processing them as if they were MIN_H ones,
            # and later are not feasible anyways (dists[THIS, v_x,v_y]=inf)
            h_in = MIN_H
        # print(h_in)
        # get the subtracted in height cost
        h_in_sub = np.array(
            [max([0, h - h_in]) for h in height_combinations[:, 0]]
        )
        # overall costs
        summed_hinhout = h_in_sub + height_combinations[:, 1]
        # print("summed_hinhout", summed_hinhout)
        best_height_costs[h] = np.min(summed_hinhout)
        best_hinhout = height_combinations[np.argmin(summed_hinhout)]
        if h_in > best_hinhout[0]:
            hin_new[h] = h_in
        else:
            hin_new[h] = best_hinhout[0]
        hout_new[h] = best_hinhout[1]
        # print(hin_hout[-1])
        # height_in = h_in - height_combinations[:,0]

    # normalize and subtract the 60 because we don't pay for the base height
    best_height_costs = best_height_costs / MIN_H - 1
    # TODO: normalization --> currently percentage above MIN_H

    # deal with inf ones
    if np.all(best_height_costs == np.inf):
        # only 80-80 works but hasn't been found
        best_height_costs = np.zeros(len(best_height_costs)) + 2.0
        hin_new = np.zeros(len(hin_new)) + MAX_H
        hout_new = np.zeros(len(hout_new)) + MAX_H
    return best_height_costs, hin_new, hout_new


@jit(nopython=True)
def check_heigths(bres_line, S, height_resistance, yA, yB, h_diff):
    # get minimum heights of v_x,v_y dependent on incoming edge

    # compute lowest point of sag
    x0 = S / 2 - ((yB - yA) * CAT_H / (CAT_W * S))
    # compute height above x0 at left point
    A_height = (CAT_W * x0**2) / (2 * CAT_H)
    # print(height_bline)

    # iterate over points on bres_line
    stepsize = S / (len(bres_line) + 1)
    heights_above = np.zeros(len(bres_line))
    for k, (i, j) in enumerate(bres_line):
        x = x0 - stepsize * (k + 1)
        cat = (CAT_W * x**2) / (2 * CAT_H)
        heights_above[k] = yA - A_height - height_resistance[i, j] + cat

    # analyse heights_above:
    if np.all(heights_above >= 11):
        return 0
    elif np.any(heights_above < 11 - h_diff):
        # would not work with 80 - 80
        return 2
    else:
        # somewhere inbetween
        return 1


@jit(nopython=True)
def height_optimal_sp(
    stack, shifts, angles_all, dists, preds, instance, edge_inst, shift_lines,
    edge_weight, height_arr, height_resistance, height_weight, RESOLUTION,
    MIN_H, MAX_H
):
    """
    Compute the shortest path costs with considering pylon heights
    """
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[-i - 1][0]
        v_y = stack[-i - 1][1]

        # so far height on in edges
        heights_in = height_arr[:, v_x, v_y]

        # iterate over neighbors
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]

            if (
                0 <= neigh_x < dists.shape[1] and 0 <= neigh_y < dists.shape[2]
                and instance[neigh_x, neigh_y] < np.inf
            ):
                # compute bresenham line and distance
                bres_line = shift_lines[s] + np.array([v_x, v_y])
                # span of cable
                S = int(
                    np.sqrt((v_x - neigh_x)**2 + (v_y - neigh_y)**2)
                ) * RESOLUTION
                # left and right point
                yA = height_resistance[v_x, v_y] + MIN_H
                yB = height_resistance[neigh_x, neigh_y] + MIN_H
                # check heights
                height_OK = check_heigths(
                    bres_line, S, height_resistance, yA, yB, MAX_H - MIN_H
                )

                # TODO: wrong resistances or edge not possible
                if (
                    height_OK == 2
                    or height_resistance[neigh_x, neigh_y] == 200
                    or height_resistance[v_x, v_y] == 200
                ):
                    continue

                # compute edge costs
                edge_cost_list = np.zeros(len(bres_line))
                for k in range(len(bres_line)):
                    edge_cost_list[k] = edge_inst[bres_line[k, 0],
                                                  bres_line[k, 1]]
                if np.any(edge_cost_list == np.inf):
                    continue
                edge_cost = np.mean(edge_cost_list)
                # edge_cost = comp_edge_cost(bres_line, edge_weight, edge_inst)

                # compute height costs if applicable
                if height_OK == 1:
                    # get bline heights: pixel-wise height from the resistances
                    height_bline = np.zeros((len(bres_line) + 2))
                    height_bline[0] = height_resistance[v_x, v_y]
                    height_bline[-1] = height_resistance[neigh_x, neigh_y]
                    dists_bline = np.zeros((len(bres_line) + 2))
                    base_fraction = S / (len(bres_line) + 1)
                    for d in range(len(bres_line) + 2):
                        dists_bline[d] = d * base_fraction
                        if d != len(bres_line) + 1 and d != 0:
                            (ind_x, ind_y) = bres_line[d - 1]
                            height_bline[d] = height_resistance[ind_x, ind_y]

                    height_combinations = min_heights(
                        height_bline, dists_bline, S, MIN_H, MAX_H, nr_steps=5
                    )

                    best_height_costs, hin_new, hout_new = select_best_height(
                        heights_in, height_combinations, MIN_H, MAX_H
                    )
                    # print(best_height_costs * height_weight)
                    # put all together:
                    cost_per_angle = (
                        best_height_costs * height_weight
                    ) + dists[:, v_x, v_y] + angles_all[s] + instance[
                        neigh_x, neigh_y] + (edge_cost * edge_weight)

                    # get best in edge and update the predecessors
                    best_in_edge = np.argmin(cost_per_angle)
                    # refunction height_arr: s is now the outgoing shift
                    height_arr[s, v_x, v_y] = hin_new[best_in_edge]
                    height_arr[s, neigh_x, neigh_y] = hout_new[best_in_edge]
                # OTHER CASE: MIN_H - MIN_H is fine, height cost is zero
                else:
                    height_arr[s, neigh_x, neigh_y] = MIN_H
                    # weighted costs --> height cost is zero
                    cost_per_angle = dists[:, v_x, v_y] + angles_all[
                        s] + instance[neigh_x, neigh_y
                                      ] + edge_cost * edge_weight
                    best_in_edge = np.argmin(cost_per_angle)
                    # update the in height - here it would be 60
                    height_arr[s, v_x, v_y] = heights_in[best_in_edge]

                # set distances and predecessors
                dists[s, neigh_x, neigh_y] = np.min(cost_per_angle)
                preds[s, neigh_x, neigh_y] = best_in_edge

    return dists, preds, height_arr


class HeightGraph(ImplicitLG):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        super(HeightGraph, self).__init__(
            cost_instance,
            hard_constraints,
            directed=directed,
            graphtool=graphtool,
            verbose=verbose
        )

    def init_heights(self, height_resistance_path, min_h, max_h, scale_factor):
        # Load height resistance tif
        # TODO: change because corridors etc
        with rasterio.open(height_resistance_path, "r") as ds:
            height_rest = ds.read()[0]
            # for swiss instance artifacts:
            height_rest[height_rest < 0] = 200
        height_resistance = np.swapaxes(
            rescale(height_rest, scale_factor), 1, 0
        )
        self.height_resistance = height_resistance
        self.min_h = min_h
        self.max_h = max_h
        self.resolution = 10 * scale_factor

    def add_edges(self, height_weight=0, edge_weight=0):
        self.edge_weight = edge_weight
        self.height_weight = height_weight
        tic = time.time()
        # SORT
        tmp_list = self._helper_list()
        visit_points = (self.instance < np.inf).astype(int)
        stack = topological_sort_jit(
            self.start_inds[0], self.start_inds[1], np.asarray(self.shifts),
            visit_points, tmp_list
        )
        if self.verbose:
            print("time topo sort:", round(time.time() - tic, 3))
            # print("stack length", len(stack))
        tic = time.time()

        # RUN - add edges
        height_arr = np.ones(self.dists.shape)
        height_arr[:, self.start_inds[0], self.start_inds[1]] = self.min_h
        self.dists, self.preds, self.height_arr = height_optimal_sp(
            stack, np.asarray(self.shifts), self.angle_cost_array, self.dists,
            self.preds, self.instance, self.edge_inst, self.shift_lines,
            edge_weight, height_arr, self.height_resistance, height_weight,
            self.resolution, self.min_h, self.max_h
        )

        self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time edges:", round(time.time() - tic, 3))

    def get_shortest_path(self, start_inds, dest_inds, ret_only_path=False):
        if not np.any(self.dists[:, dest_inds[0], dest_inds[1]] < np.inf):
            raise RuntimeWarning("empty path")
        tic = time.time()
        curr_point = dest_inds
        path = [dest_inds]
        heights = [self.min_h]
        # first minimum: angles don't matter, just min of in-edges
        min_shift = np.argmin(self.dists[:, dest_inds[0], dest_inds[1]])
        # track back until start inds
        while np.any(curr_point - start_inds):
            new_point = curr_point - self.shifts[int(min_shift)]
            # get pylon height
            heights.append(
                self.height_arr[int(min_shift), new_point[0], new_point[1]]
            )
            # get new shift from argmins
            min_shift = self.preds[int(min_shift), curr_point[0], curr_point[1]
                                   ]
            path.append(new_point)
            curr_point = new_point

        self.heights = np.flip(np.asarray(heights), axis=0)
        path = np.flip(np.asarray(path), axis=0)
        if ret_only_path:
            return path
        self.sp = path
        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        # print(self.heights, np.sum(self.heights))
        return self.transform_path(path)
