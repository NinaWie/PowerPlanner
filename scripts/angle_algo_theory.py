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
