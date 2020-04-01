import numpy as np
from csv import writer

from scipy.ndimage.morphology import binary_dilation
from scipy.spatial.distance import cdist
# from numba import jit, njit


def append_to_csv(file_name, list_of_elem):
    """
    Append a row to a csv file
    :param file_name: filename to open csv file
    :param list_of_elem: list corresponding to new row
    """
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def time_test_csv(
    ID, CSV_TIMES, SCALE_PARAM, GTNX, GRAPH_TYPE, graph, path_costs, cost_sum,
    dist, time_pipeline, notes
):
    """
    Prepare current data for time logs csv file
    :param CSV_TIMES: output file
    :params: all parameters to save in the csv
    """
    if GTNX:
        n_nodes = len(list(graph.graph.vertices()))
    else:
        n_nodes = len(graph.graph.nodes())
    # compute average costs:
    costs = [round(s, 3) for s in np.sum(path_costs, axis=0)]
    print("costs", costs)
    # get scale factor and number of nonzero pixels:
    factor = graph.factor
    n_pixels = np.sum(np.mean(graph.cost_rest, axis=0) > 0)
    # --> csv columns:
    # scale,graphtool,graphtype,n_nodes,n_edges,add_nodes_time,add_edge_time,
    # shortest_path_time, notes
    param_list = [
        ID, SCALE_PARAM, GTNX, GRAPH_TYPE, factor, dist, n_pixels, n_nodes,
        len(list(graph.graph.edges())), graph.time_logs["add_nodes"],
        graph.time_logs["add_all_edges"], graph.time_logs["shortest_path"],
        costs, cost_sum, time_pipeline, notes
    ]
    append_to_csv(CSV_TIMES, param_list)


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


#@njit
def angle(vec1, vec2):
    """
    Compute angle between two vectors
    :params vec1, vec2: two 1-dim vectors of same size, can be lists or array
    :returns angle
    """
    # path = np.asarray(path)
    # for p, (i, j) in enumerate(path[:-2]):
    #     v1 = path[p + 1] - path[p]
    #     v2 = path[p + 1] - path[p + 2]
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    v1_norm = np.linalg.norm(vec1)
    v2_norm = np.linalg.norm(vec2)
    v1 = vec1 / v1_norm
    v2 = vec2 / v2_norm
    angle = np.arccos(np.dot(v1, v2))
    return angle


def get_donut_vals(donut_tuples, vec):
    """
    compute the angle between edges defined by donut tuples
    :param donut_tuples: list of pairs of tuples, each pair defines an edge 
    going from [(x1,y1), (x2,y2)]
    :param vec: vector to compute angle with
    :returns: list of same length as donut_tuples with all angles
    """
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


def get_path_lines(cost_shape, paths):
    """
    Given a list of paths, compute the continous lines in an array of cost_shape
    :param cost_shape: desired 2-dim output shape of array
    :param paths: list of paths of possibly different lengths, each path is 
    a list of tuples
    :returns: 2-dim binary of shape cost_shape where paths are set to 1
    """
    path_dilation = np.zeros(cost_shape)
    for path in paths:
        # iterate over path nodes
        for i in range(len(path) - 1):
            line = bresenham_line(*path[i], *path[i + 1])
            # print(line)
            for (j, k) in line:
                path_dilation[j, k] = 1
    return path_dilation


def dilation_dist(path_dilation, n_dilate=None):
    """
    Compute surface of distances with dilation
    :param path_dilation: binary array with zeros everywhere except for paths
    :param dilate: How often to do dilation --> defines radious of corridor
    :returns: 2dim array of same shape as path_dilation, with values 
    0 = infinite distance from path
    n_dilation = path location
    """
    saved_arrs = [path_dilation]
    if n_dilate is None:
        # compute number of iterations: maximum distance of pixel to line
        x_coords, y_coords = np.where(path_dilation)
        x_len, y_len = path_dilation.shape
        # print([np.min(x_coords), x_len- np.max(x_coords), np.min(y_coords), y_len- np.max(y_coords)])
        n_dilate = max(
            [
                np.min(x_coords), x_len - np.max(x_coords),
                np.min(y_coords), y_len - np.max(y_coords)
            ]
        )

    # dilate
    for _ in range(n_dilate):
        path_dilation = binary_dilation(path_dilation)
        saved_arrs.append(path_dilation)
    saved_arrs = np.sum(np.array(saved_arrs), axis=0)
    return saved_arrs


def cdist_dist(path_dilation):
    """
    Use scipy cdist function to compute distances from path (SLOW!)
    :param path_dilation: binary array with zeros everywhere except for paths
    :returns: 2dim array of same shape as path_dilation, with values 
    0 = path
    x = pixel has distance x from the path
    """
    saved_arrs = np.zeros(path_dilation.shape)
    x_len, y_len = path_dilation.shape
    xa = np.array([[i, j] for i in range(x_len) for j in range(y_len)])
    xb = np.swapaxes(np.vstack(np.where(path_dilation > 0)), 1, 0)
    print(xa.shape, xb.shape)
    all_dists = cdist(xa, xb)
    print(all_dists.shape)
    out = np.min(all_dists, axis=1)
    k = 0
    for i in range(x_len):
        for j in range(y_len):
            saved_arrs[i, j] = out[k]
            k += 1
    return saved_arrs


def get_distance_surface(out_shape, paths, mode="dilation", n_dilate=None):
    """
    Given a list of paths, compute the corridor
    :param mode: How to compute --> dilation or cdist
    :param out_shape: desired output shape
    :param paths: list of paths of possibly different lengths, each path is 
    a list of tuples
    :returns: 2dim array showing the distance from the path
    """
    path_dilation = get_path_lines(out_shape, paths)
    if mode == "dilation":
        dist_surface = dilation_dist(path_dilation, n_dilate=n_dilate)
    elif mode == "cdist":
        dist_surface = cdist_dist(path_dilation)
    else:
        raise NotImplementedError
    return dist_surface


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
