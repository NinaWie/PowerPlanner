from power_planner.utils import normalize

import matplotlib.pyplot as plt
import numpy as np
# import pwlf


def plot_path(instance, path, out_path=None, buffer=2):
    """
    Colour points on path red on the instance image
    :param instance: cost surface (numpy array)
    :param path: list of indices to colour red
    :param out_path: file path where to save the figure (if None, then show)
    :return coloured image (same shape as instance)
    """
    # expand to greyscale
    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one
    # colour nodes in path in red
    for (x, y) in path:
        expanded[x - buffer:x + buffer + 1,
                 y - buffer:y + buffer + 1] = [0.9, 0.2, 0.2]  # colour red

    plt.figure(figsize=(25, 15))
    plt.imshow(expanded, origin="lower")
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


def plot_path_costs(
    instance, path, edgecosts, class_names, out_path=None, buffer=1
):

    edgecosts = np.asarray(edgecosts)
    print("out costs shape:", edgecosts.shape)
    # env_costs = np.mean(edgecosts, axis=1)  # edgecosts[:, 1]  #
    n_crit = len(instance)
    # print("number crit", n_crit)

    # fill for angle costs
    # if len(instance) < n_crit:
    #     print("fill angle")
    #     repeat_list = [1 for _ in range(len(instance))]
    #     repeat_list[0] = 2
    #     instance = np.repeat(instance, repeat_list, axis=0)
    # print("instance shape", instance.shape)
    if n_crit < edgecosts.shape[1]:
        edgecosts = edgecosts[:, 1:]  # exclude angle costs

    plt.figure(figsize=(25, 15))
    for j in range(n_crit):
        curr_costs = instance[j]
        expanded = np.expand_dims(curr_costs, axis=2)
        expanded = np.tile(
            expanded, (1, 1, 3)
        )  # overwrite instance by tiled one
        # put values into visible range
        normed_env_costs = normalize(edgecosts[:, j])
        # colour nodes in path
        for i, (x, y) in enumerate(path):
            # colour red for high cost
            expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                     1] = [0.9, 1 - normed_env_costs[i], 0.2]
        # wo_zero = expanded[:, np.any(curr_costs > 0, axis=0)]
        # wo_zero = wo_zero[np.any(curr_costs > 0, axis=1), :]
        wo_zero = expanded

        # display
        plt.subplot(1, n_crit, j + 1)
        plt.imshow(np.swapaxes(wo_zero, 1, 0), origin="upper")
        plt.title(class_names[j])
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


def plot_pipeline_paths(plot_surfaces, output_paths, buffer=1, out_path=None):
    """
    subplots of different steps in the pipeline
    """
    plt.figure(figsize=(20, 10))
    for i, (p, p_cost) in enumerate(output_paths):
        plt.subplot(1, len(output_paths), i + 1)

        # expand to greyscale
        expanded = np.expand_dims(plot_surfaces[i], axis=2)
        expanded = np.tile(expanded, (1, 1, 3))
        # colour nodes in path in red
        for (x, y) in p:
            expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                     1] = [0.9, 0.2, 0.2]  # colour red
        plt.imshow(np.swapaxes(expanded, 1, 0), origin="upper")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


def plot_pareto(pareto0, pareto1, paths, vary, classes, out_path=None):
    # plot pareto
    color = plt.cm.rainbow(np.linspace(0, 1, len(pareto0)))
    # scatter pareto curve
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(pareto0, pareto1, c=color)
    plt.xlabel(classes[0], fontsize=15)
    plt.ylabel(classes[1], fontsize=15)
    plt.title("Pareto frontier for " + classes[0] + " vs " + classes[1])

    # plot paths
    plt.subplot(1, 2, 2)
    for i, p in enumerate(paths):
        p_arr = np.array(p)
        plt.plot(p_arr[:, 1], p_arr[:, 0], label=str(i), c=color[i])
        # print("path length:", len(p))
    plt.legend(title="Weight of " + classes[0] + " costs")
    plt.title(
        "Paths for varied weights for " + classes[0] + " vs " + classes[1]
    )

    if out_path is not None:
        plt.savefig(out_path + "_pareto.png")
    else:
        plt.show()


# def plot_pareto_paths(paths, classes, out_path=None):
#     color = iter(plt.cm.rainbow(np.linspace(0, 1, len(paths))))
#     plt.figure(figsize=(20, 10))
#     for i, p in enumerate(paths):
#         p_arr = np.array(p)
#         c = next(color)
#         plt.plot(p_arr[:, 1], p_arr[:, 0], label=str(i), c=c)
#         # print("path length:", len(p))
#     plt.legend(title="Weight of " + classes[0] + " costs")
#     if out_path is not None:
#         plt.savefig(out_path + "_pareto_paths.png")
#     else:
#         plt.show()


def plot_graph(g):
    """
    Plot networkx graph with edge attributes
    :param g: graph object
    """
    labels = nx.get_edge_attributes(g, 'weight')  # returns dictionary
    pos = nx.get_node_attributes(g, 'pos')
    plt.figure(figsize=(20, 10))
    nx.draw(g, pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    # plt.savefig("first_graph.png")
    plt.show()


def _transform_cols(arr):
    """
    transform integer values into ranfom colours
    """
    uni = np.unique(arr)
    transformed = np.zeros(int(np.max(uni) + 1))
    for i, u in enumerate(uni):
        transformed[int(u)] = i
    cols = np.random.rand(len(uni), 3)
    x, y = arr.shape
    new = np.zeros((x, y, 3))
    for i in range(x):
        for j in range(y):
            new[i, j] = cols[int(transformed[int(arr[i, j])])]
    return new


def piecewise_linear_fit(path, segments=5, out_path=None):
    x = path[:, 0]
    y = path[:, 1]
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(segments)
    print("breaks", breaks)
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = my_pwlf.predict(x_hat)

    plt.figure(figsize=(20, 10))
    plt.plot(x, y, 'o')
    plt.plot(x_hat, y_hat, '-')
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
