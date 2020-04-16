import matplotlib.pyplot as plt
import numpy as np

from power_planner.utils.utils import linegraph_donut, angle


def test_linegraph_donut():
    vec = [1, 1]
    linegraph_tuples = linegraph_donut(3, 5, vec, thresh=np.pi)
    for l in linegraph_tuples:
        assert np.dot(l[1], vec) >= 0
        assert np.dot(l[0], vec) <= 0
        assert angle(l[0], l[1]) >= np.pi


# unit tests dag_angles
example2 = np.zeros(test_example.shape)
example2 += np.inf
line = bresenham_line(start_inds[0], start_inds[1], dest_inds[0], dest_inds[1])
for (i, j) in line:
    example2[i, j] = 1
# get out thing, check that nonempty and corresponding

out = np.zeros(test_example.shape)
for (i, j) in my_path:
    out[i, j] = 1
plt.imshow(out)
plt.show()

# unit test2: with 90 grad winkel --> not supposed to work with max_angle 3/4, only with 1/4
example2 = np.zeros(test_example.shape)
example2 += np.inf
example2[start_inds[0], start_inds[1]:dest_inds[1] - 3] = 1
example2[start_inds[0], dest_inds[1]] = 1
example2[dest_inds[0]:start_inds[0] - 3, dest_inds[1]] = 1
plt.imshow(example2)
plt.show()