from power_planner.utils import normalize

import matplotlib.pyplot as plt
import numpy as np


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
    n_crit = edgecosts.shape[1]
    print("number crit", n_crit)

    # fill for angle costs
    if len(instance) < n_crit:
        print("fill angle")
        repeat_list = [1 for _ in range(len(instance))]
        repeat_list[0] = 2
        instance = np.repeat(instance, repeat_list, axis=0)
    print("instance shape", instance.shape)

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
        wo_zero = expanded[:, np.any(curr_costs > 0, axis=0)]
        wo_zero = wo_zero[np.any(curr_costs > 0, axis=1), :]

        # display
        plt.subplot(1, n_crit, j + 1)
        plt.imshow(np.swapaxes(wo_zero, 1, 0), origin="upper")
        plt.title(class_names[j])
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


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
