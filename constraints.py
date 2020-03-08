import numpy as np

from power_planner.utils import get_half_donut


def bresenham_line(x0, y0, x1, y1):
    """
    find pixels on line between two pixels
    https://stackoverflow.com/questions/50995499/generating-pixel-values-of-line-connecting-2-points

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


def get_kernel(shifts):
    """
    Get all kernels describing the path of the edges in a discrete raster
    :param shifts: possible circle points
    :returns kernel: all possible kernels (number of circle points x upper x upper)
    :returns posneg: a list indicating whether it is a path to the left (=1) or to the right(=0)
    """
    upper = np.amax(np.absolute(shifts)) + 1
    posneg = []
    kernel = np.zeros((len(shifts), upper, upper))

    for i, shift in enumerate(shifts):
        if shift[1] < 0:
            posneg.append(1)
            line = bresenham_line(0, upper - 1, shift[0], upper - 1 + shift[1])
        else:
            posneg.append(0)
            line = bresenham_line(0, 0, shift[0], shift[1])
        # add points of line to the kernel
        for (j, k) in line:
            kernel[i, j, k] += 1
    return kernel, posneg


def convolve(img, kernel, neg=0):
    k_size = len(kernel)
    if neg:
        padded = np.pad(img, ((0, k_size - 1), (k_size - 1, 0)))
    else:
        padded = np.pad(img, ((0, k_size), (0, k_size)))
    # print(padded.shape)
    convolved = np.zeros(img.shape)
    w, h = img.shape
    for i in range(0, w):
        for j in range(0, h):
            patch = padded[i:i + k_size, j:j + k_size]
            convolved[i, j] = np.sum(patch * kernel)
    return convolved


def angle(path):
    path = np.asarray(path)
    for p, (i, j) in enumerate(path[:-2]):
        v1 = path[p + 1] - path[p]
        v2 = path[p + 1] - path[p + 2]
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        angle = np.arccos(np.dot(v1, v2))
        if angle < np.pi:
            pass


# Questions:
## altitude leads to more costs for pylons? because they are higher?

# height profile constraints:
## simply exclude edges which cannot be placed --> only works when iterating over edges

## Angle constraints:
# * line graph
# * path straighening toolbox
