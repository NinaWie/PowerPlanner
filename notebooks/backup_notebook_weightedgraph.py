class WeightedGraph(GeneralGraph):

    def __init__(
        self,
        cost_instance,
        hard_constraints,
        directed=True,
        graphtool=1,
        verbose=1
    ):
        # assert cost_instance.shape == hard_constraints.shape
        # , "Cost size must be equal to corridor definition size!"

        # time logs
        self.time_logs = {}
        tic = time.time()

        # indicator whether to use networkx or graph tool
        self.graphtool = graphtool

        # cost surface
        self.cost_instance = cost_instance
        self.hard_constraints = hard_constraints
        self.x_len, self.y_len = hard_constraints.shape

        # initialize graph:
        GeneralGraph.__init__(
            self, directed=directed, graphtool=graphtool, verbose=verbose
        )

        # print statements
        self.verbose = verbose

        self.time_logs["init_graph"] = round(time.time() - tic, 3)

        # _, self.cost_rest = self.set_pos2node(self.cost_rest, 4) # self.pos2node,

        # original pos2node: all filled except for hard constraints
        self.pos2node_orig = np.arange(1, self.x_len * self.y_len + 1).reshape(
            (self.x_len, self.y_len)
        )
        self.pos2node_orig *= (self.hard_constraints > 0).astype(int)
        # self.pos2node_orig[self.pos2node_orig==0] = -1
        self.pos2node = self.pos2node_orig - 1  # CHANGED

        print("initialized weighted graph pos2node")

    def _get_seeds(self, greater_zero, factor):
        lab = 0
        x_len, y_len = greater_zero.shape
        seeds = np.zeros(greater_zero.shape)
        omitted = 0
        for i in np.arange(0, x_len, factor):
            for j in np.arange(0, y_len, factor):
                if greater_zero[i, j]:
                    seeds[i, j] = lab
                    lab += 1
                else:
                    omitted += 1
        print("omitted:", omitted)
        return seeds

    def _watershed_transform(
        self, cost_rest, factor, compact=0.01, mode="center"
    ):
        """
        :param mode: all = all combinations in one cluster possible
                --> leading to larger distances
                center = only the center of each cluster can be connected
        """
        tic = time.time()
        img = np.mean(
            cost_rest, axis=0
        )  # take mean image for clustering TODO: weighted sum?

        greater_zero = (img > 0).astype(int)

        edges = filters.sobel(img)

        seeds = self._get_seeds(greater_zero, factor)
        if self.verbose:
            print("number seeds: ", np.sum(seeds > 0))

        w1 = watershed(edges, seeds, compactness=compact)
        # w1 is full watershed --> labels spread over corridor borders
        # but: label 0 also included --> +1 before corridor
        w1_g_zero = (w1 + 1) * greater_zero
        # labels: 0 is forbidden, 1 etc is watershed labels
        labels = np.unique(w1_g_zero)

        new_cost_rest = np.zeros(cost_rest.shape)
        # iterate over labels (except for 0 - forbidden)
        for i, lab in enumerate(labels[1:]):
            inds = w1_g_zero == lab

            # , lab in enumerate(labels):
            x_inds, y_inds = np.where(w1_g_zero == lab)
            for j in range(len(cost_rest)):
                new_cost_rest[j, int(np.mean(x_inds)),
                              int(np.mean(y_inds))] = np.mean(
                                  cost_rest[j, x_inds, y_inds]
                              )
        return new_cost_rest

    def downsample(self, img, factor):
        x_len_new = img.shape[1] // factor
        y_len_new = img.shape[2] // factor
        new_img = np.zeros(img.shape)
        for i in range(x_len_new):
            for j in range(y_len_new):
                patch = img[:, i * factor:(i + 1) * factor,
                            j * factor:(j + 1) * factor]
                if np.any(patch):  # >0.01):
                    for k in range(len(new_img)):
                        part = patch[k]
                        if np.any(part):  # >0.01):
                            new_img[k, i * factor, j * factor] = np.min(
                                part[part > 0]
                            )  # >0.01])
        return new_img

    def set_cost_rest(self, factor, corridor, start_inds, dest_inds):
        self.cost_rest = self.cost_instance * (self.hard_constraints >
                                               0).astype(int) * corridor

        # downsample
        if factor > 1:
            self.cost_rest = self.downsample(self.cost_rest, factor)
        # else:
        #     self.cost_rest = self.cost_instance
        # repeat because edge artifacts
        self.cost_rest = self.cost_rest * (self.hard_constraints >
                                           0).astype(int) * corridor

        # add start and end TODO ugly
        graph.cost_rest[:, dest_inds[0],
                        dest_inds[1]] = instance[:, dest_inds[0], dest_inds[1]]
        graph.cost_rest[:, start_inds[0],
                        start_inds[1]] = instance[:, start_inds[0],
                                                  start_inds[1]]

        # define pos2node accordingly:
        inverted_corridor = np.absolute(1 - corridor).astype(bool)
        self.pos2node = self.pos2node_orig.copy()
        self.pos2node[inverted_corridor
                      ] = -1  # set all which are not in the corridor to -1

    def set_shift(self, lower, upper, vec, max_angle):
        GeneralGraph.set_shift(self, lower, upper, vec, max_angle)
        self.shift_vals = get_donut_vals(self.shifts, vec)

    def add_nodes(self):
        tic = time.time()
        # add nodes to graph
        n_nodes = len(np.unique(self.pos2node))
        GeneralGraph.add_nodes(self, n_nodes)
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)

    def _update_time_logs(
        self, times_add_edges, times_edge_list, tic_function
    ):
        self.time_logs["add_edges"] = round(np.mean(times_add_edges), 3)
        self.time_logs["add_edges_times"] = times_add_edges

        self.time_logs["edge_list"] = round(np.mean(times_edge_list), 3)
        self.time_logs["edge_list_times"] = times_edge_list

        self.time_logs["add_all_edges"] = round(time.time() - tic_function, 3)

    def _compute_edge_costs(self, shift_index):
        # switch axes for shift
        cost_rest_switched = np.moveaxis(self.cost_rest, 0, -1)
        # shift by shift
        costs_shifted = ConstraintUtils.shift_surface(
            cost_rest_switched, self.shifts[shift_index]
        )
        # switch axes back
        costs_shifted = np.moveaxis(costs_shifted, -1, 0)

        weights = (costs_shifted + self.cost_rest) / 2
        # new version: edge weights
        # weights = convolve_faster(self.cost_rest, kernels[i], posneg[i])
        # weights = weights1 + 2 * weights2
        # print(
        #     "max node weights", np.max(weights1), "max edge weights:",
        #     np.max(weights2), "min node weights", np.min(weights1),
        #     "min edge weights:", np.min(weights2)
        # )

        mean_costs_shifted = np.mean(costs_shifted, axis=0) > 0

        inds_shifted = self.pos2node[mean_costs_shifted]
        # delete the ones where inds_shifted is zero

        # take weights of the shifted ones
        weights_arr = np.array(
            [w[mean_costs_shifted] for i, w in enumerate(weights)]
        )

        return inds_shifted, weights_arr

    def add_edges(self, corridor):
        corridor = corridor * (self.hard_constraints >
                               0).astype(int)  # ADDED --> shows probabilities
        print("max min corridor", np.max(corridor), np.min(corridor))

        tic_function = time.time()

        n_edges = 0
        # kernels, posneg = ConstraintUtils.get_kernel(self.shifts, self.shift_vals)
        # edge_array = []

        times_edge_list = []
        times_add_edges = []

        if self.verbose:
            print("n_neighbors:", len(self.shifts))

        for i in range(len(self.shifts)):

            prob_arr = np.random.rand(*corridor.shape)
            prob_arr = (corridor > prob_arr).astype(int)

            self.cost_rest = self.cost_instance * prob_arr

            inds_orig = self.pos2node[prob_arr > 0]  # changed!

            if i < 3:
                plt.imshow(np.mean(self.cost_rest, axis=0))
                plt.show()
            tic_edges = time.time()

            # compute shift and weights
            inds_shifted, weights_arr = self._compute_edge_costs(i)
            assert len(inds_shifted) == len(
                inds_orig
            ), "orig:{},shifted:{}".format(len(inds_orig), len(inds_shifted))

            # concatenete indices and weights, select feasible ones
            inds_arr = np.asarray([inds_orig, inds_shifted])
            inds_weights = np.concatenate((inds_arr, weights_arr), axis=0)
            pos_inds = inds_shifted >= 0
            out = np.swapaxes(inds_weights, 1, 0)[pos_inds]

            # remove edges with high costs:
            # first two columns of out are indices
            # weights_arr = np.mean(out[:, 2:], axis=1)
            # weights_mean = np.quantile(weights_arr, 0.9)
            # inds_higher = np.where(weights_arr < weights_mean)
            # out = out[inds_higher[0]]

            # Error if -1 entries because graph-tool crashes with -1 nodes
            if np.any(out[:2].flatten() == -1):
                print(np.where(out[:2] == -1))
                raise RuntimeError

            n_edges += len(out)
            times_edge_list.append(round(time.time() - tic_edges, 3))

            # add edges to graph
            tic_graph = time.time()
            if self.graphtool:
                self.graph.add_edge_list(out, eprops=self.cost_props)
            else:
                nx_edge_list = [(e[0], e[1], {"weight": e[2]}) for e in out]
                self.graph.add_edges_from(nx_edge_list)
            times_add_edges.append(round(time.time() - tic_graph, 3))

            # alternative: collect edges here and add alltogether
            # edge_array.append(out)

        # # alternative: add edges all in one go
        # tic_concat = time.time()
        # edge_lists_concat = np.concatenate(edge_array, axis=0)
        # self.time_logs["concatenate"] = round(time.time() - tic_concat, 3)
        # print("time for concatenate:", self.time_logs["concatenate"])
        # tic_graph = time.time()
        # self.graph.add_edge_list(edge_lists_concat, eprops=[self.weight])
        # self.time_logs["add_edges"] = round(
        #     (time.time() - tic_graph) / len(shifts), 3
        # )

        self._update_time_logs(times_add_edges, times_edge_list, tic_function)
        if self.verbose:
            print("DONE adding", n_edges, "edges:", time.time() - tic_function)

    def add_edges_old(self):
        tic_function = time.time()
        inds_orig = self.pos2node[np.mean(self.cost_rest, axis=0) > 0]

        n_edges = 0
        # kernels, posneg = get_kernel(self.shifts, self.shift_vals)
        # edge_array = []

        times_edge_list = []
        times_add_edges = []

        for i in range(len(self.shifts)):

            tic_edges = time.time()

            # compute shift and weights
            inds_shifted, weights_arr = self._compute_edge_costs(i)
            assert len(inds_shifted) == len(
                inds_orig
            ), "orig:{},shifted:{}".format(len(inds_orig), len(inds_shifted))

            # concatenete indices and weights, select feasible ones
            inds_arr = np.asarray([inds_orig, inds_shifted])
            inds_weights = np.concatenate((inds_arr, weights_arr), axis=0)
            pos_inds = inds_shifted >= 0
            out = np.swapaxes(inds_weights, 1, 0)[pos_inds]

            # remove edges with high costs:
            # first two columns of out are indices
            # weights_arr = np.mean(out[:, 2:], axis=1)
            # weights_mean = np.quantile(weights_arr, 0.9)
            # inds_higher = np.where(weights_arr < weights_mean)
            # out = out[inds_higher[0]]
            if np.any(out[:2].flatten() == -1):
                print(np.where(out[:2] == -1))
                raise RuntimeError

            n_edges += len(out)
            times_edge_list.append(round(time.time() - tic_edges, 3))

            # add edges to graph
            tic_graph = time.time()
            if self.graphtool:
                self.graph.add_edge_list(out, eprops=self.cost_props)
            else:
                nx_edge_list = [(e[0], e[1], {"weight": e[2]}) for e in out]
                self.graph.add_edges_from(nx_edge_list)
            times_add_edges.append(round(time.time() - tic_graph, 3))

            # alternative: collect edges here and add alltogether
            # edge_array.append(out)

        # # alternative: add edges all in one go
        # tic_concat = time.time()
        # edge_lists_concat = np.concatenate(edge_array, axis=0)
        # self.time_logs["concatenate"] = round(time.time() - tic_concat, 3)
        # print("time for concatenate:", self.time_logs["concatenate"])
        # tic_graph = time.time()
        # self.graph.add_edge_list(edge_lists_concat, eprops=[self.weight])
        # self.time_logs["add_edges"] = round(
        #     (time.time() - tic_graph) / len(shifts), 3
        # )

        self._update_time_logs(times_add_edges, times_edge_list, tic_function)
        if self.verbose:
            print("DONE adding", n_edges, "edges:", time.time() - tic_function)

    def remove_vertices(self, corridor):
        ## Possibility 1: remove all edges of vertices in the (smaller) corridor
        # corr_vertices = self.pos2node * corridor
        # new_vertices = corr_vertices[corr_vertices>0]
        # for v in new_vertices:
        #     self.graph.clear_vertex(self.graph.vertex(v))
        ## Possibility 2: remove all edges --> only considering corridor then
        # self.graph.clear_edges()
        ## Possibility 3: remove all out_edges of corridor vertices (inefficient probably)
        corr_vertices = self.pos2node * corridor
        new_vertices = corr_vertices[corr_vertices > 0]
        remove_property = self.graph.new_edge_property("float")
        remove_property.a = np.zeros(self.weight.get_array().shape)
        for v in new_vertices:
            for e in self.graph.vertex(v).out_edges():
                remove_property[e] = 1
        remove_labeled_edges(self.graph, remove_property)

    def add_start_and_dest(self, start_inds, dest_inds):
        start_node_ind = self.pos2node[start_inds[0], start_inds[1]]
        dest_node_ind = self.pos2node[dest_inds[0], dest_inds[1]]
        if self.graphtool:
            return self.graph.vertex(start_node_ind
                                     ), self.graph.vertex(dest_node_ind)
        else:
            return start_node_ind, dest_node_ind

    def get_shortest_path(self, source, target):
        """
        Compute shortest path from source vertex to target vertex
        """
        tic = (time.time())
        # #if source and target are given as indices:
        vertices_path = GeneralGraph.get_shortest_path(self, source, target)

        path = []
        for v in vertices_path:
            if self.graphtool:
                ind = self.graph.vertex_index[v]
            else:
                ind = int(v)
            path.append((ind // self.y_len, ind % self.y_len))

        # if self.verbose:
        #     print("time for shortest path", time.time() - tic)

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)

        # compute edge costs
        out_costs = []
        for (i, j) in path:
            out_costs.append(self.cost_instance[:, i, j].tolist())
        # TODO: append graph.weight[edge]?

        return path, out_costs

    def get_shortest_path_old(self, source, target):
        tic = time.time()

        # compute shortest path
        vertices_path, _ = shortest_path(
            self.graph,
            source,
            target,
            weights=self.weight,
            negative_weights=True
        )
        # else:
        #     vertices_path = nx.dijkstra_path(self.graph, source, target)

        # transform path
        path = []
        out_costs = []
        for v in vertices_path:
            v_ind = self.graph.vertex_index[v]
            x_inds, y_inds = np.where(self.pos2node == v_ind)
            # find minimum value field out of possible

            min_ind_x = int(np.mean(x_inds))
            min_ind_y = int(np.mean(y_inds))

            path.append((min_ind_x, min_ind_y))
            out_costs.append(self.cost_rest[:, min_ind_x, min_ind_y].tolist())

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)

        return path, out_costs