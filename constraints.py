import numpy as np


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


def get_kernel(shifts, shift_vals):
    """
    Get all kernels describing the path of the edges in a discrete raster
    :param shifts: possible circle points
    :returns kernel: all possible kernels 
     shape: (number of circle points x upper x upper)
    :returns posneg: a list indicating whether it is a path to the left (=1) 
    or to the right(=0)
    """
    upper = np.amax(np.absolute(shifts)) + 1
    posneg = []
    kernel = np.zeros((len(shifts), upper, upper))

    max_val = np.max(shift_vals)

    for i, shift in enumerate(shifts):
        if shift[1] < 0:
            posneg.append(1)
            line = bresenham_line(0, upper - 1, shift[0], upper - 1 + shift[1])
        else:
            posneg.append(0)
            line = bresenham_line(0, 0, shift[0], shift[1])
        # add points of line to the kernel
        normed_val = shift_vals[i] / (len(line) * max_val)
        for (j, k) in line:
            kernel[i, j, k] = normed_val
    return kernel, posneg


def convolve_faster(img, kernel, neg):
    """
    Convolve a 2d img with a kernel, storing the output in the cell
    corresponding the the left or right upper corner
    :param img: 2d numpy array
    :param kernel: kernel (must have equal size and width)
    :param neg: if neg=0, store in upper left corner, if neg=1, store in upper
    right corner
    :return convolved image of same size
    """
    k_size = len(kernel)
    # a = np.pad(img, ((0, k_size-1), (0, k_size-1)))
    if neg:
        padded = np.pad(img, ((0, k_size - 1), (k_size - 1, 0)))
    else:
        padded = np.pad(img, ((0, k_size - 1), (0, k_size - 1)))

    s = kernel.shape + tuple(np.subtract(padded.shape, kernel.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(padded, shape=s, strides=padded.strides * 2)
    return np.einsum('ij,ijkl->kl', kernel, subM)


def convolve(img, kernel, neg=0):
    """
    Convolve a 2d img with a kernel, storing the output in the cell
    corresponding the the left or right upper corner
    :param img: 2d numpy array
    :param kernel: kernel (must have equal size and width)
    :param neg: if neg=0, store in upper left corner, if neg=1, store in upper
    right corner
    :return convolved image of same size
    """
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


# Questions:
## altitude leads to more costs for pylons? because they are higher?

# height profile constraints:
## simply exclude edges which cannot be placed --> only works when iterating over edges

## Angle constraints:
# * line graph
# * path straighening toolbox
