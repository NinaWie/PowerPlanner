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
        ID,  , GTNX, GRAPH_TYPE, factor, dist, n_pixels,
        graph.n_nodes, graph.n_edges, graph.time_logs["add_nodes"],
        graph.time_logs["add_all_edges"], graph.time_logs["shortest_path"],
        costs, cost_sum, time_pipeline, notes
    ]
    append_to_csv(CSV_TIMES, param_list)


def compute_pylon_dists(PYLON_DIST_MIN, PYLON_DIST_MAX, RASTER, SCALE_PARAM):
    PYLON_DIST_MIN = PYLON_DIST_MIN / RASTER
    PYLON_DIST_MAX = PYLON_DIST_MAX / RASTER
    if SCALE_PARAM > 1:
        PYLON_DIST_MIN /= SCALE_PARAM
        PYLON_DIST_MAX /= SCALE_PARAM
    print("defined pylon distances in raster:", PYLON_DIST_MIN, PYLON_DIST_MAX)
    return PYLON_DIST_MIN, PYLON_DIST_MAX


def normalize(instance):
    """
    0-1 normalization of values of instance
    """
    return (instance -
            np.min(instance)) / (np.max(instance) - np.min(instance))


def rescale(img, scale_factor):
    """
    Scale down image by a factor
    Arguments:
        img: numpy array of any dimension
        scale_factor: integer >= 1
    Returns:
        numpy array with 1/scale_factor size along each dimension
    """
    if scale_factor == 1:
        return img
    x_len_new = img.shape[0] // scale_factor
    y_len_new = img.shape[1] // scale_factor
    new_img = np.zeros((x_len_new, y_len_new))
    for i in range(x_len_new):
        for j in range(y_len_new):
            patch = img[i * scale_factor:(i + 1) * scale_factor, j *
                        scale_factor:(j + 1) * scale_factor]
            new_img[i, j] = np.mean(patch)
    return new_img


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
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    # normalize
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    # special cases where arcos is nan
    if np.allclose(v1, v2):
        return 0
    if np.allclose(-v1, v2):
        return np.pi
    # compute angle
    angle = np.arccos(np.dot(v1, v2))
    # want to use full 360 degrees
    if np.sin(angle) < 0:
        angle = 2 * np.pi - angle
    # can still be nan if v1 or v2 is 0
    if np.isnan(angle):
        print(vec1, vec2, v1, v2)
        return 0
        # raise ValueError("angle is nan, check whether vec1 or vec2 = 0")
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
        # compute angle
        ang = angle([i, j], vec)
        # add all valid ones
        if ang <= angle_max:
            new_tuples.append((i, j))
    return new_tuples


def discrete_angle_costs(ang, max_angle_lg):
    """
    Define angle costs for each angle
    Arguments:
        ang: float between 0 and pi, angle between edges
        max_angle_lg: maximum angle cutoff
    returns: angle costs
    Here computed as Stefano said: up to 30 degrees + 50%, up to 60 degrees
    3 times the cost, up to 90 5 times the cost --> norm: 1.5 / 5 = 0.3
    """
    # TODO: 3 times technical costs for example
    # previously:
    # return ang / max_angle_lg
    if ang <= np.pi / 6:
        return 0.3
    elif ang <= np.pi / 3:
        return 0.6
    else:
        return 1


def get_lg_donut(
    radius_low, radius_high, vec, max_angle, max_angle_lg=np.pi / 4
):
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
    for (i, j) in tuple_zip:
        # if in incoming half
        if i * vec[0] + j * vec[1] <= 0:
            for (k, l) in tuple_zip:
                ang = angle([-k, -l], [i, j])
                # if smaller max angle and general outgoing half
                if ang <= max_angle_lg and k * vec[0] + l * vec[1] >= 0:
                    angle_norm = discrete_angle_costs(ang, max_angle_lg)
                    linegraph_tuples.append([[i, j], [k, l], angle_norm])
    return linegraph_tuples


def get_path_lines(cost_shape, paths):
    """
    Given a list of paths, compute continous lines in an array of cost_shape
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
            # set all pixels on line to 1
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
        # dilate as much as the largest distance from the sides
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
    # transform array to indices array as input to cdist
    xa = np.array([[i, j] for i in range(x_len) for j in range(y_len)])
    xb = np.swapaxes(np.vstack(np.where(path_dilation > 0)), 1, 0)
    print(xa.shape, xb.shape)
    # main computation
    all_dists = cdist(xa, xb)
    print(all_dists.shape)
    out = np.min(all_dists, axis=1)
    # re-transform indices to image
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
