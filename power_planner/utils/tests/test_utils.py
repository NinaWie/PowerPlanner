# import matplotlib.pyplot as plt
import numpy as np
import unittest

from power_planner.utils.utils import *


class TestUtils(unittest.TestCase):

    def test_donut(self) -> None:
        for max_angle in [np.pi / 2, np.pi / 4]:
            shifts = get_half_donut(3, 5, [1, 1], angle_max=max_angle)
            self.assertNotEqual(len(shifts), 0)
            for (i, j) in shifts:
                rad = np.sqrt(i**2 + j**2)
                self.assertGreaterEqual(rad, 3)
                self.assertLessEqual(rad, 5)
                self.assertLessEqual(angle([1, 1], [i, j]), max_angle)
            # assert that all possible fields are in the list
            for i in range(-5, 5, 1):
                for j in range(-5, 5, 1):
                    rad = np.sqrt(i**2 + j**2)
                    if rad >= 3 and rad <= 5 and angle(
                        [i, j], [1, 1]
                    ) < max_angle:
                        self.assertIn((i, j), shifts)

    def test_linegraph_donut(self) -> None:
        for max_angle in [np.pi / 2, np.pi / 4]:
            vec = [1, 1]
            linegraph_tuples = get_lg_donut(
                3, 5, vec, np.pi / 2, max_angle_lg=max_angle
            )
            for l in linegraph_tuples:
                self.assertTrue(np.dot(l[1], vec) >= 0)
                self.assertTrue(np.dot(l[0], vec) <= 0)
                transformed = np.array(l[0]) * (-1)
                self.assertTrue(angle(transformed, l[1]) <= max_angle)
                if angle(transformed, l[1]) > np.pi / 6:
                    self.assertGreaterEqual(l[2], 0.6)
                    if max_angle == np.pi / 4:
                        self.assertEqual(l[2], 0.6)

    def test_rescale(self) -> None:
        arr = np.random.rand(20, 20)
        new_arr = rescale(arr, 3)
        self.assertTrue(new_arr.shape == (6, 6))
        self.assertTrue(new_arr[2, 1] == np.mean(arr[6:9, 3:6]))

    def test_angle(self) -> None:
        actual_angles = [np.pi, 3 * np.pi / 4, np.pi / 2, 0.25 * np.pi, 0]
        for i, shift in enumerate(
            [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]
        ):
            ang = angle(shift, [1, 0])
            self.assertAlmostEqual(ang, actual_angles[i])

    def test_dilation(self) -> None:
        path_dilation = np.zeros((20, 20))
        for i in range(19):
            path_dilation[i, i] = 1
        out = dilation_dist(path_dilation, n_dilate=5)
        self.assertTrue(out[5, 5] == 6)
        self.assertTrue(np.all(out >= 0))
        self.assertEqual(out.shape, path_dilation.shape)
        self.assertEqual(np.max(out), 6)
        self.assertEqual(out[5, 6], 5)

    def test_upscale(self) -> None:
        instance_corr = (np.random.rand(40, 40) > 0.2).astype(int)
        downsampled_corr = np.ceil(rescale(instance_corr, 2))
        upsampled = upscale_corr(instance_corr, downsampled_corr, 2)
        self.assertTrue(np.all(upsampled == instance_corr))


if __name__ == "__main__":
    unittest.main()
