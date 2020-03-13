import matplotlib.pyplot as plt
import numpy as np

from power_planner.utils import linegraph_donut, angle


def test_linegraph_donut():
    vec = [1, 1]
    linegraph_tuples = linegraph_donut(3, 5, vec, thresh=np.pi)
    for l in linegraph_tuples:
        assert np.dot(l[1], vec) >= 0
        assert np.dot(l[0], vec) <= 0
        assert angle(l[0], l[1]) >= np.pi
