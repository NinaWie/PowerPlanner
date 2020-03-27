import numpy as np
from power_planner.utils import bresenham_line


class ConstraintUtils():

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                line = bresenham_line(
                    0, upper - 1, shift[0], upper - 1 + shift[1]
                )
            else:
                posneg.append(0)
                line = bresenham_line(0, 0, shift[0], shift[1])
            # add points of line to the kernel
            normed_val = shift_vals[i] / (len(line) * max_val)
            for (j, k) in line:
                kernel[i, j, k] = normed_val
        return kernel, posneg

    @staticmethod
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

    @staticmethod
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
