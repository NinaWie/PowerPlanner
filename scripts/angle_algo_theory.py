# test data
def greedy(test_c, acd):
    best_ones = []
    for i in range(len(test_c)):
        helper = [val + abs(j - i) * acd for j, val in enumerate(test_c)]
        pred = np.argmin(helper)
        best_ones.append(pred)
    return np.asarray(best_ones)


test_range = 200
for acd in [10, 15, 20, 30]:
    iterations = []
    for _ in range(50):
        test_c = (np.random.rand(test_range) * 100).astype(int)
        c_tuples = [(c, i, 0) for i, c in enumerate(test_c)]
        sorted_c = sorted(c_tuples, key=lambda x: x[0])

        # auxiliary lists
        e_update = [i for i in range(len(test_c))]
        to_update = np.ones(len(test_c))

        # compute greedy method for validation
        preds_gt = greedy(test_c, acd)

        # queue
        q = []
        update_counter = 1
        predecessor = np.zeros(len(test_c))
        while np.any(to_update):
            # print(q)
            # print(sorted_c)
            # print(to_update)
            # print("------------")

            # get next lowest value
            if len(q) > 0 and q[0][0] < sorted_c[0][0]:
                cost, ind, div = q[0]
                del q[0]
            else:
                cost, ind, div = sorted_c[0]
                del sorted_c[0]

            # update step - only update if not updated yet!
            if to_update[ind + div]:
                # update
                updated_ind = ind + div
                predecessor[ind + div] = ind
                to_update[ind + div] = 0

                # CORRECTNESS: compare to ground truthpredecessor
                div_gt_pred = abs(preds_gt[updated_ind] - updated_ind)
                # if not equal predecessor and it does matter (not exactly equal costs)
                if ind != preds_gt[
                    updated_ind] and cost != abs(div_gt_pred) * acd + test_c[
                        preds_gt[updated_ind]]:
                    print(test_c)
                    print(
                        preds_gt[ind + div], ind, div,
                        test_c[updated_ind - 4:updated_ind + 4], cost
                    )

                # get in which direction to go further
                add = np.sign(div)
                if add == 0:
                    # update -1: if in bounds and not updated yet
                    if ind - 1 >= 0 and to_update[ind - 1]:
                        q.append((cost + acd, ind, -1))
                    add = 1
                if ind + div + add >= 0 and ind + div + add < len(
                    to_update
                ) and to_update[ind + div + add]:
                    q.append((cost + acd, ind, div + add))
                # to_update[ind-div] = 0

            update_counter += 1
        iterations.append(update_counter)

        # print(predecessor)
        # print(preds_gt)
        # assert np.all(predecessor==preds_gt)
    print(np.mean(iterations), np.max(iterations) / test_range)

# BACKUP: working algorithm

dists = graph.dists.copy()
preds = graph.preds.copy()
shifts = graph.shifts
stack = graph.stack_array.copy()
pos2node = graph.pos2node.copy()
angles_all = graph.angle_cost_array
instance = graph.instance
edge_cost = np.zeros(graph.dists.shape)

inst_x_len, inst_y_len = instance.shape
n_neighbors = len(shifts)
for i in range(len(dists)):
    v_x = stack[i, 0]
    v_y = stack[i, 1]

    # sort the in edge distances and initialize
    initial_S = np.argsort(dists[i])
    marked_plus = np.zeros(n_neighbors)
    marked_minus = np.zeros(n_neighbors)

    # initialize dists and do first pass
    neighbor_vals = np.zeros(n_neighbors) + np.inf
    neighbor_inds = np.zeros(n_neighbors).astype(int) - 1
    ground_truth = np.zeros(n_neighbors) + np.inf
    ground_truth_pred = np.zeros(n_neighbors)

    for s in range(n_neighbors):
        neigh_x = int(v_x + shifts[s][0])
        neigh_y = int(v_y + shifts[s][1])
        if (
            0 <= neigh_x < inst_x_len and 0 <= neigh_y < inst_y_len
            and pos2node[neigh_x, neigh_y] >= 0
            and instance[neigh_x, neigh_y] < np.inf
        ):
            # PROBLE
            neighbor_vals[s] = instance[neigh_x, neigh_y]
            neigh_stack_ind = pos2node[neigh_x, neigh_y]
            neighbor_inds[s] = neigh_stack_ind
            # initialize distances to the straight line value
            dists[neigh_stack_ind, s] = dists[
                i, s
            ]  # + instance[neigh_x, neigh_y]+ edge_cost[neigh_stack_ind, s]
            preds[neigh_stack_ind, s] = s

            cost_per_angle = dists[i] + angles_all[s] + instance[
                neigh_x, neigh_y] + edge_cost[neigh_stack_ind, s]
            ground_truth[s] = np.min(cost_per_angle)
            ground_truth_pred[s] = np.argmin(cost_per_angle)

    # set current tuple: in edge and shift (out edge index unncessary because same as in edge)
    current_in_edge = initial_S[0]
    current_shift = 0
    tuple_counter = 0

    # debug
    update_shift = np.zeros(n_neighbors)

    while tuple_counter < len(initial_S) - 1:
        # best out edge is exactly the same shift!
        current_out_edge = (current_in_edge + current_shift) % n_neighbors
        # print(current_out_edge, current_shift)
        # compute possible update value:
        update_val = dists[i, current_in_edge] + angles_all[current_out_edge,
                                                            current_in_edge]

        if current_shift == 0:
            marked = marked_plus[current_out_edge
                                 ] and marked_minus[current_out_edge]
        elif current_shift > 0:
            marked = marked_plus[current_out_edge]
        else:
            marked = marked_minus[current_out_edge]
        # update only if better
        neigh_stack_ind = neighbor_inds[current_out_edge]
        # print(marked, neigh_stack_ind, update_val)

        # actual update: only if the neighbor exists
        # PROBLEM: what if angle cost becomes inf
        if marked == 0 and neigh_stack_ind >= 0 and update_val <= dists[
            neigh_stack_ind, current_out_edge]:
            dists[neigh_stack_ind, current_out_edge] = update_val
            preds[neigh_stack_ind, current_out_edge] = current_in_edge
            update_shift[current_out_edge] = current_shift
            # if not np.isclose(dists[neigh_stack_ind, current_out_edge], ground_truth[current_out_edge]):
            #     print(i, current_out_edge, current_shift, "wrong")
            #     print("new", dists[neigh_stack_ind, current_out_edge], "gt", ground_truth[current_out_edge], "val", update_val, "in dist", dists[i, current_in_edge])
            # else:
            #     pass # print(i, current_in_edge, current_shift, "ok")
            # Consider next edge in this direction
            if current_shift < 0:
                current_shift -= 1
            if current_shift <= 0:
                marked_minus[current_out_edge] = 1
            if current_shift >= 0:
                current_shift += 1
                marked_plus[current_out_edge] = 1

        # inf neighbor --> jump over it if its incoming edge is worse
        elif marked == 0 and neigh_stack_ind < 0 and update_val <= dists[
            i, current_out_edge]:
            if current_shift < 0:
                current_shift -= 1
            if current_shift <= 0:
                marked_minus[current_out_edge] = 1
            if current_shift >= 0:
                current_shift += 1
                marked_plus[current_out_edge] = 1

        # already marked or update not successful:
        # Consider first edge in other direction or next overall tuple
        else:
            if current_shift > 0:
                current_shift = -1
            else:
                # get next tuple from stack
                tuple_counter += 1
                current_in_edge = initial_S[tuple_counter]
                current_shift = 0

    # previously: Always updated only with distance and angle --> now add the instance cost and edge cost
    # PROBLEM IF NO DAG
    for s in range(n_neighbors):
        neigh_stack_ind = neighbor_inds[s]
        if neigh_stack_ind >= 0:
            pred_shift = int(preds[neigh_stack_ind, s])
            dists[neigh_stack_ind, s
                  ] = dists[neigh_stack_ind, s] + neighbor_vals[s] + edge_cost[
                      neigh_stack_ind, pred_shift]

    for s in range(n_neighbors):
        stack_ind = neighbor_inds[s]
        if stack_ind >= 0 and not np.isclose(
            dists[stack_ind, s], ground_truth[s]
        ):
            print("PROBLEM")
            # print(dists[stack_ind, s], ground_truth[s])
            # neigh_x = int(v_x + shifts[s][0])
            # neigh_y = int(v_y + shifts[s][1])
            print(s)
            print("updated with", update_shift[s])
            print(dists[stack_ind, s], ground_truth[s])
            print(
                "new pred", preds[stack_ind, s], "gt_pred",
                ground_truth_pred[s]
            )
            # print(dists[i, int(ground_truth_pred[s])], dists[i, int(preds[stack_ind, s])])
            print(neighbor_inds[14])
    print("-------------------------")
    # print(dists[i])
    # print(instance[neigh_x, neigh_y])
    # print("-------------------------", i)