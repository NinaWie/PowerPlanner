import numpy as np
import time
from numba import jit
from power_planner.graphs.implicit_lg import ImplicitLG


class ImplicitPareto(ImplicitLG):

    def get_pareto(
        self, vary, source, dest, out_path=None, compare=[0, 1], plot=1
    ):
        """
            Arguments:
                vary: how many weights to explore
                        e.g 3 --> each cost class can have weight 0, 0.5 or 1
                source, dest: as always the source and destination vertex
                out_path: where to save the pareto figure(s)
                compare: indices of cost classes to compare
            Returns:
                paths: All found paths
                pareto: The costs for each combination of weights
            """
        tic = time.time()
        # initialize lists
        pareto = list()
        paths = list()
        cost_sum = list()

        # get vary weights between 0 and 1
        var_weights = np.around(np.linspace(0, 1, vary), 2)

        # construct weights array
        if len(compare) == 2:
            weights = [[v, 1 - v] for v in var_weights]
        elif len(compare) == 3:
            weights = list()
            for w0 in var_weights:
                for w1 in var_weights[var_weights <= 1 - w0]:
                    weights.append([w0, w1, 1 - w0 - w1])
        else:
            raise ValueError("argument compare can only have length 2 or 3")

        # compute paths for each combination of weights
        for j in range(len(weights)):
            # option 2: np.zeros(len(cost_arrs)) + non_compare_weight
            self.set_edge_costs(
                self.cost_classes,
                weights,
                angle_weight=0.1  # cfg.ANGLE_WEIGHT
            )
            self.add_edges(edge_weight=e_w)  # TODO
            path, path_costs, cost_sum = self.get_shortest_path(source, dest)
            # get shortest path
            path, path_costs, _ = self.get_shortest_path(source, dest)
            # don't take cost_sum bc this is sum of original weighting
            pareto.append(np.sum(path_costs, axis=0)[compare])
            paths.append(path)
            # take overall sum of costs (unweighted) that this w leads to
            cost_sum.append(np.sum(path_costs))

        # print best weighting
        best_weight = np.argmin(cost_sum)
        w = self.cost_weights.copy()
        w[compare] = np.array(weights[best_weight]) * w_avail
        print("Best weights:", w, "with (unweighted) costs:", np.min(cost_sum))

        self.time_logs["pareto"] = round(time.time() - tic, 3)

        pareto = np.array(pareto)
        classes = [self.cost_classes[comp] for comp in compare]
        # Plotting
        if plot:
            if len(compare) == 2:
                plot_pareto_scatter_2d(
                    pareto,
                    weights,
                    classes,
                    cost_sum=cost_sum,
                    out_path=out_path
                )
            elif len(compare) == 3:
                # plot_pareto_3d(pareto, weights, classes)
                plot_pareto_scatter_3d(
                    pareto,
                    weights,
                    classes,
                    cost_sum=cost_sum,
                    out_path=out_path
                )
        return paths, weights, cost_sum
