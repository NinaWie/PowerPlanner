import unittest
import numpy as np
from power_planner.utils.utils_ksp import KspUtils


class TestKspUtils(unittest.TestCase):

    def test_pairwise(self) -> None:
        p1 = np.array([[0, 1], [0, 2], [0, 3]])
        p2 = np.array([[0, 1], [0, 3], [0, 3]])
        p3 = np.array([[1, 1], [1, 2], [1, 3]])
        dists_jac = [0, 1 / 3, 1]
        dists_eucl = [0, 1, 1]
        dists = KspUtils.pairwise_dists([p1, p2, p3], mode="jaccard")
        for i in range(3):
            self.assertAlmostEqual(dists[0, i], dists_jac[i])
        dists = KspUtils.pairwise_dists([p1, p2, p3], mode="eucl_max")
        for i in range(3):
            self.assertAlmostEqual(dists[0, i], dists_eucl[i])


if __name__ == "__main__":
    unittest.main()
