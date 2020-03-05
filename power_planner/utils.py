import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def pos2node(pos, length):
    return pos[0] * length + pos[1]


def node2pos(node, length):
    j = node % length  # rest
    i = node // length
    return i, j


def normalize(instance):
    return (instance -
            np.min(instance)) / (np.max(instance) - np.min(instance))


def get_donut(radius_low, radius_high):
    img_size = int(radius_high + 10)
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[-img_size:img_size, -img_size:img_size]
    # circle equation
    circle = (xx)**2 + (yy)**2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    donut = np.logical_and(circle < (radius_high**2), circle > (radius_low**2))
    pos_x, pos_y = np.where(donut > 0)
    return pos_x - img_size, pos_y - img_size


def get_half_donut(radius_low, radius_high):
    pos_x, pos_y = get_donut(radius_low, radius_high)
    new_tuples = []
    for i, j in zip(pos_x, pos_y):
        if i > 0 or i == 0 and j > 0:
            new_tuples.append((i, j))
    return new_tuples


def reduce_instance(img, square):
    x_len, y_len = img.shape
    new_img = np.zeros((x_len // square, y_len // square))
    for i in range(x_len // square):
        for j in range(y_len // square):
            patch = img[i * square:(i + 1) * square,
                        j * square:(j + 1) * square]
            new_img[i, j] = np.mean(patch)
    return new_img


def get_shift_transformed(shifts):
    shift_tuples = []
    for shift in shifts:
        if shift[0] < 0:
            tup1 = (0, -shift[0])
        else:
            tup1 = (shift[0], 0)
        if shift[1] < 0:
            tup2 = (0, -shift[1])
        else:
            tup2 = (shift[1], 0)
        shift_tuples.append((tup1, tup2))
    return shift_tuples


def plot_path(instance, path, out_path=None):
    # expand to greyscale
    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one
    # colour nodes in path in red
    for (x, y) in path:
        expanded[x - 2:x + 2, y - 2:y + 2] = [0.9, 0.2, 0.2]  # colour red

    plt.figure(figsize=(25, 15))
    plt.imshow(expanded)
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


def plot_graph(g):
    labels = nx.get_edge_attributes(g, 'weight')  # returns dictionary
    pos = nx.get_node_attributes(g, 'pos')
    plt.figure(figsize=(20, 10))
    nx.draw(g, pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    # plt.savefig("first_graph.png")
    plt.show()
