import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pwlf


def pos2node(pos, length):
    return pos[0] * length + pos[1]


def node2pos(node, length):
    j = node % length  # rest
    i = node // length
    return i, j


def normalize(instance):
    """
    0-1 normalization of values of instance
    """
    return (instance -
            np.min(instance)) / (np.max(instance) - np.min(instance))


def get_donut(radius_low, radius_high):
    """
    Compute all indices of points in donut around (0,0)
    :param radius_low: minimum radius
    :param radius_high: maximum radius
    :returns: tuples of indices of points with radius between radius_low
    and radius_high around (0, 0)
    """
    img_size = int(radius_high + 10)
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[-img_size:img_size, -img_size:img_size]
    # circle equation
    circle = (xx)**2 + (yy)**2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    donut = np.logical_and(
        circle <= (radius_high**2), circle >= (radius_low**2)
    )
    pos_x, pos_y = np.where(donut > 0)
    return pos_x - img_size, pos_y - img_size


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


def angle(vec1, vec2):
    # path = np.asarray(path)
    # for p, (i, j) in enumerate(path[:-2]):
    #     v1 = path[p + 1] - path[p]
    #     v2 = path[p + 1] - path[p + 2]
    v1_norm = np.linalg.norm(vec1)
    v2_norm = np.linalg.norm(vec2)
    v1 = np.asarray(vec1) / v1_norm
    v2 = np.asarray(vec2) / v2_norm
    angle = np.arccos(np.dot(v1, v2))
    return angle


def get_donut_vals(donut_tuples, vec):
    return [angle(tup, vec) + 0.1 for tup in donut_tuples]


def get_half_donut(radius_low, radius_high, vec, angle_max=0.5 * np.pi):
    """
    Returns only the points with x >= 0 of the donut points (see above)
    :param radius_low: minimum radius
    :param radius_high: maximum radius
    :returns: tuples of indices of points with radius between radius_low
    and radius_high around (0, 0)
    """
    pos_x, pos_y = get_donut(radius_low, radius_high)
    new_tuples = []
    for i, j in zip(pos_x, pos_y):
        # if i > 0 or i == 0 and j > 0:
        # if i * vec[0] + j * vec[1] >= 0:
        ang = angle([i, j], vec)
        if ang <= angle_max:
            new_tuples.append((i, j))
    return new_tuples


def get_lg_donut(radius_low, radius_high, vec, min_angle=3 * np.pi / 4):
    """
    Compute all possible combinations of edges in restricted angle
    :param radius_low: minimum radius
    :param radius_high: maximum radius
    :param vec: direction vector
    :returns: list with entries [[edge1, edge2, cost of angle between them]]
    where costs are normalized values between 0 and 1
    """
    donut = get_donut(radius_low, radius_high)
    tuple_zip = list(zip(donut[0], donut[1]))
    linegraph_tuples = []
    norm_factor = np.pi - min_angle  # here: 1/4 pi
    for (i, j) in tuple_zip:
        # if in incoming half
        if i * vec[0] + j * vec[1] <= 0:
            for (k, l) in tuple_zip:
                ang = angle([k, l], [i, j]) - min_angle
                # min angle and general outgoing edges half
                if ang >= 0 and k * vec[0] + l * vec[1] >= 0:
                    angle_norm = round(1 - (ang / norm_factor), 2)
                    linegraph_tuples.append([[i, j], [k, l], angle_norm])
    return linegraph_tuples


def shift_surface_old(costs, shift):
    """
    Shifts a numpy array and pads with zeros
    :param costs: 2-dim numpy array
    :param shift: tuple of shift in x and y direction
    (negative value for left / up shift)
    :returns shifted array of same size
    """
    if shift[0] < 0:
        tup1 = (0, -shift[0])
    else:
        tup1 = (shift[0], 0)
    if shift[1] < 0:
        tup2 = (0, -shift[1])
    else:
        tup2 = (shift[1], 0)

    costs_shifted = np.pad(costs, (tup1, tup2), mode='constant')

    if shift[0] > 0 and shift[1] > 0:
        costs_shifted = costs_shifted[:-shift[0], :-shift[1]]
    elif shift[0] > 0 and shift[1] <= 0:
        costs_shifted = costs_shifted[:-shift[0], -shift[1]:]
    elif shift[0] <= 0 and shift[1] > 0:
        costs_shifted = costs_shifted[-shift[0]:, :-shift[1]]
    elif shift[0] <= 0 and shift[1] <= 0:
        costs_shifted = costs_shifted[-shift[0]:, -shift[1]:]

    return costs_shifted


def shift_surface(costs, shift):
    """
    Shifts a numpy array and pads with zeros
    :param costs: 2-dim numpy array
    :param shift: tuple of shift in x and y direction
    BUT: ONLY WORKS FOR (+,+) or (+,-) shift tuples
    :returns shifted array of same size
    """
    rolled_costs = np.roll(costs, shift, axis=(0, 1))
    if shift[0] >= 0:
        rolled_costs[:shift[0], :] = 0
    else:
        rolled_costs[shift[0]:, :] = 0
    if shift[1] >= 0:
        rolled_costs[:, :shift[1]] = 0
    else:
        rolled_costs[:, shift[1]:] = 0
    return rolled_costs


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
        plt.savefig(out_path)
    else:
        plt.show()


def plot_path_costs(instance, path, edgecosts, out_path=None, buffer=1):
    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one

    edgecosts = np.asarray(edgecosts)
    env_costs = edgecosts[:, 1]  # np.sum(edgecosts, axis=1) #
    normed_env_costs = (env_costs - np.min(env_costs)
                        ) / (np.max(env_costs) - np.min(env_costs))
    # colour nodes in path in red
    for i, (x, y) in enumerate(path):
        # print(edgecosts[i])
        expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                 1] = [0.9, 1 - normed_env_costs[i], 0.2]  # colour red

    plt.figure(figsize=(25, 15))
    plt.imshow(np.swapaxes(expanded, 1, 0), origin="upper")
    if out_path is not None:
        plt.savefig(out_path)
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


def bresenham_line(x0, y0, x1, y1):
    """
    Finds the cell indices on a straight line between two raster cells
    """
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []
    for x in range(x0, x1 + 1):
        if steep:
            line.append([y, x])
        else:
            line.append([x, y])

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line
