from power_planner.utils.utils import normalize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
# import pwlf


def plot_path(instance, path, out_path=None, buffer=2):
    """
    Simplest version: Colour points on path red on the instance image
    Arguments:
        instance: cost surface (numpy array)  
        path: list of indices to colour red
        out_path: file path where to save the figure (if None, then show)
    Return:
        Saving or showing the path image
    """
    # normalize instance for better display
    non_inf = instance[instance < np.inf]
    instance = (instance -
                np.min(non_inf)) / (np.max(non_inf) - np.min(non_inf))
    # expand from greyscale to colour channels
    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one
    # colour nodes in path in red
    for (x, y) in path:
        expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                 1] = [0.9, 0.2, 0.2]  # colour red
    # plot and save
    plt.figure(figsize=(25, 15))
    plt.imshow(np.swapaxes(expanded, 1, 0))
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


def plot_path_costs(
    instance, path, edgecosts, class_names, out_path=None, buffer=1
):
    """
    Cost visualization: Plot one image for each cost class:
    Arguments:
        instance: n_classes x imgwidth x imgheight sized array
        path: list or array of (x,y) coordinates
        edgecosts: path length x n_classes array or list containing costs
        class_names: n_classes list of names for plot titles
        out_path: where to save
        buffer: num pixels to color (for large images one pixel too small)
    """
    edgecosts = np.asarray(edgecosts)
    print("out costs shape:", edgecosts.shape)
    n_crit = len(instance)
    # exclude angle costs,
    if n_crit < edgecosts.shape[1]:
        edgecosts = edgecosts[:, 1:]

    # iterate over cost classes to make subplots
    plt.figure(figsize=(25, 15))
    for j in range(n_crit):
        curr_costs = instance[j]
        # from grey scale to colour channel image
        expanded = np.expand_dims(curr_costs, axis=2)
        expanded = np.tile(expanded, (1, 1, 3))
        # put values into visible range
        normed_env_costs = normalize(edgecosts[:, j])
        # colour nodes in path
        for i, (x, y) in enumerate(path):
            # colour red for high cost
            expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                     1] = [0.9, 1 - normed_env_costs[i], 0.2]
        # comment in next lines for stripping zero rows and columns
        # wo_zero = expanded[:, np.any(curr_costs > 0, axis=0)]
        # wo_zero = wo_zero[np.any(curr_costs > 0, axis=1), :]

        # display
        plt.subplot(1, n_crit, j + 1)
        plt.imshow(np.swapaxes(expanded, 1, 0), origin="upper")
        plt.title(class_names[j])
    plt.tight_layout()
    # save image
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


def plot_pipeline_paths(plot_surfaces, output_paths, buffer=1, out_path=None):
    """
    Visualize pipeline progress
    Arguments:
        plot_surfaces: List of 2D arrays (cost image) to plot on
        output_paths: list of same length as plot_surfaces containing paths
        buffer: how many pixels to colour
    """
    plt.figure(figsize=(20, 10))
    for i, (p, _) in enumerate(output_paths):
        plt.subplot(1, len(output_paths), i + 1)

        # expand to rgb
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


def plot_prior_paths(plot_surfaces, output_paths, buffer=2, out_path=None):
    """
    Huge plot with one pipeline plot for each prior corridor
    Arguments:
        plot_surfaces: array of size nr_corrs x nr_pipeline
        output_paths: list of paths, one for each corridor
    Returns:
        Num_corr x num_pipeline_steps subplots with paths
    """
    plt.figure(figsize=(20, 20))
    # iterate over corridor images
    for corr in range(len(plot_surfaces)):
        # iterate over pipeline steps
        for i, p in enumerate(output_paths[corr]):
            plt.subplot(
                len(plot_surfaces), len(output_paths[corr]),
                (len(output_paths[corr]) * corr) + i + 1
            )

            # expand to greyscale
            expanded = np.expand_dims(plot_surfaces[corr][i], axis=2)
            expanded = np.tile(expanded, (1, 1, 3))
            # colour nodes in path in red
            for (x, y) in p:
                expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                         1] = [0.9, 0.2, 0.2]  # colour red
            plt.imshow(expanded, origin="upper")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


def plot_k_sp(ksp, inst, out_path=None):
    """
    Plot k shortest paths on the instance
    Arguments:
        ksp: list of infos for the k shortest path: for each path, the first
            entry is the path itself, the second the costs array, the third
            the cost sum
        inst: instance to plot on
    """
    # get relevant information
    costs = [k[2] for k in ksp]
    paths = [k[0] for k in ksp]

    # plot main image (cost surface)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.swapaxes(inst, 1, 0))
    # iterate over k shortest paths
    for i, path in enumerate(paths):
        path = np.asarray(path)
        plt.plot(
            path[:, 0], path[:, 1], label=str(round(costs[i], 2)), linewidth=3
        )
    # plot and save
    leg = plt.legend(fontsize=15)
    leg.set_title('Costs', prop={'size': 15})
    plt.axis("off")
    if out_path is not None:
        plt.savefig(out_path + "_ksp.png")
    else:
        plt.show()


def plot_pareto_paths(pareto_out, inst, out_path=None):
    """
    Plot k shortest paths on the instance
    Arguments:
        pareto_out: tuple of pareto outputs: (paths, weights, cost sums)
        inst: instance to plot on
    """
    # get relevant information
    paths, weight_colours, cost_sum = pareto_out

    # plot main image (cost surface)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.swapaxes(inst, 1, 0))
    # iterate over k shortest paths
    for i, path in enumerate(paths):
        path = np.asarray(path)
        col = weight_colours[i]
        if len(col) == 2:
            col = [col[0], 0, col[1]]
        plt.plot(
            path[:, 0],
            path[:, 1],
            label=str(round(cost_sum[i], 2)),
            c=col,
            linewidth=3
        )
    # plot and save
    leg = plt.legend(fontsize=15)
    leg.set_title('Weighted costs', prop={'size': 15})
    plt.axis("off")
    if out_path is not None:
        plt.savefig(out_path + "_pareto_paths.png")
    else:
        plt.show()


def plot_pareto_3d(pareto, weights, classes, out_path=None):
    """
    3D plot of pareto points from 3 cost classes
    Arguments:
        pareto: np array of shape num_paths x 3, with costs for each path
        weights: array or list of shape num_paths x 3, giving the weights
                that yielded the pareto costs
        classes: list of 3 strings, the compared graphs
    """
    fig = plt.figure(figsize=(10, 5))
    ax = Axes3D(fig)

    for i in range(len(pareto)):
        x, y, z = tuple(pareto[i])
        col_weights = (np.asarray([weights[i]]) + 0.4) / 1.4
        # print(x,y,z, weights[i])
        ax.scatter(x, y, z, marker='o', s=100, c=col_weights, label=weights[i])

    ax.set_xlabel(classes[0], fontsize=15)
    ax.set_ylabel(classes[1], fontsize=15)
    ax.set_zlabel(classes[2], fontsize=15)
    # manually define legend
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color=[1, 0, 0],
            markersize=10,
            lw=0.1,
            label='only ' + classes[0]
        ),
        Line2D(
            [0], [0],
            marker='o',
            color=[0, 1, 0],
            lw=0.1,
            label='only ' + classes[1],
            markersize=10
        ),
        Line2D(
            [0], [0],
            marker='o',
            color=[0, 0, 1],
            lw=0.1,
            label='only ' + classes[2],
            markersize=10
        )
    ]

    # Create the figure
    ax.legend(handles=legend_elements, loc='upper center', fontsize=17)

    if out_path is not None:
        plt.savefig(out_path + "_pareto.png")
    else:
        plt.show()


def plot_pareto_scatter_3d(
    pareto, weights, classes, cost_sum=None, out_path=None
):
    """
    3D plot of pareto points from 3 cost classes
    Arguments:
        pareto: np array of shape num_paths x 3, with costs for each path
        weights: array or list of shape num_paths x 3, giving the weights that
                yielded the pareto costs
        classes: list of 3 strings, the compared graphs
    """
    if cost_sum is not None:
        norm_cost_sums = np.array(cost_sum)
        # min max normalization
        norm_cost_sums = (norm_cost_sums - np.min(norm_cost_sums)) / (
            np.max(norm_cost_sums) - np.min(norm_cost_sums)
        )
    # print(norm_cost_sums)
    size = 100

    fig = plt.figure(figsize=(20, 5))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        ind1 = i
        ind2 = (i + 1) % 3
        for j in range(len(pareto)):
            label = np.argmax(weights[j])
            if cost_sum is not None:
                size = norm_cost_sums[j] * 1000 + 20
            col_weights = (np.asarray([weights[j]]) + 0.4) / 1.4
            ax.scatter(
                pareto[j, ind1],
                pareto[j, ind2],
                c=col_weights,
                s=size,
                label=label
            )
        plt.xlabel(classes[ind1], fontsize=17)
        plt.ylabel(classes[ind2], fontsize=17)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color=[1, 0, 0],
            markersize=10,
            lw=0.1,
            label='high ' + classes[0] + ' weight',
        ),
        Line2D(
            [0], [0],
            marker='o',
            color=[0, 1, 0],
            lw=0.1,
            label='high ' + classes[1] + ' weight',
            markersize=10
        ),
        Line2D(
            [0], [0],
            marker='o',
            color=[0, 0, 1],
            lw=0.1,
            label='high ' + classes[2] + ' weight',
            markersize=10
        )
    ]

    # Create the figure
    ax.legend(handles=legend_elements, loc='upper center', fontsize=17)
    if out_path is not None:
        plt.savefig(out_path + "_pareto.png")
    else:
        plt.show()


def plot_pareto_scatter_2d(
    pareto, weights, classes, cost_sum=None, out_path=None
):
    """
    Scatter to compare two cost classes
    Arguments:
        pareto: np array of shape num_paths x 2, with costs for each path
        weights: array or list of shape num_paths x 2, giving the weights
                that yielded the pareto costs
        classes: list of 2 strings, the compared graphs
    """
    if cost_sum is not None:
        norm_cost_sums = np.array(cost_sum)
        # normalize
        norm_cost_sums = (norm_cost_sums - np.min(norm_cost_sums)) / (
            np.max(norm_cost_sums) - np.min(norm_cost_sums)
        )
        size = norm_cost_sums * 1000 + 20
    else:
        size = 100
    color = np.array([[w[0], 0, w[1]] for w in weights])
    # scatter pareto curve
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(pareto[:, 0], pareto[:, 1], c=color, s=size)
    plt.xlabel(classes[0], fontsize=15)
    plt.ylabel(classes[1], fontsize=15)
    # manually create legend
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color=[1, 0, 0],
            markersize=10,
            lw=0.1,
            label='only ' + classes[0]
        ),
        Line2D(
            [0], [0],
            marker='o',
            color=[0, 0, 1],
            lw=0.1,
            label='only ' + classes[1],
            markersize=10
        )
    ]
    # Create the figure
    plt.legend(handles=legend_elements, loc='upper center', fontsize=17)
    plt.title(
        "Pareto frontier for " + classes[0] + " vs " + classes[1], fontsize=17
    )
    if out_path is not None:
        plt.savefig(out_path + "_pareto.png")
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
    transform integer values into random colours
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
    """
    Function from pwlf to straighten a path afterwards
    Performs piecewise linear fit
    Arguments:
        path: actual path
        segments: how many linear segments the final path should have
    Attention! Runtime scales a lot with the number of segments
    """
    x = path[:, 0]
    y = path[:, 1]
    # fit
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(segments)
    print("breaks", breaks)
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = my_pwlf.predict(x_hat)
    # plot
    plt.figure(figsize=(20, 10))
    plt.plot(x, y, 'o')
    plt.plot(x_hat, y_hat, '-')
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
