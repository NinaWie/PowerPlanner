import unittest
import numpy as np
import matplotlib.pyplot as plt
from power_planner.algorithms import (
    optimal_pylon_spotting, optimal_route, ksp_pylons, ksp_routes
)


def plot_paths(instance, paths, buffer=0, out_path="test_path.png"):
    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one
    # colour nodes in path in red
    for path in paths:
        for (x, y) in path:
            expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                     1] = [0.9, 0.2, 0.2]  # colour red
    # plot and save
    plt.figure(figsize=(25, 15))
    plt.imshow(np.swapaxes(expanded, 1, 0))
    plt.savefig(out_path, bbox_inches='tight')


class TestAPI(unittest.TestCase):

    test_instance = np.random.rand(100, 100)
    num_nans = 100
    forb_x = (np.random.rand(num_nans) * 100).astype(int)
    forb_y = (np.random.rand(num_nans) * 100).astype(int)
    test_instance[forb_x, forb_y] = np.nan

    # create configuration
    cfg = dict()
    cfg["start_inds"] = [6, 6]
    cfg["dest_inds"] = [94, 90]
    test_instance[tuple(cfg["start_inds"])] = 0
    test_instance[tuple(cfg["dest_inds"])] = 0

    def test_optimal_route(self) -> None:
        path = optimal_route(
            self.test_instance,
            self.cfg.copy(),
        )
        self.assertTrue(len(path) > 0)
        self.assertListEqual(list(path[0]), list(self.cfg["start_inds"]))
        self.assertListEqual(list(path[-1]), list(self.cfg["dest_inds"]))

        plot_paths(
            self.test_instance, [path],
            buffer=0,
            out_path="test_optimal_route.png"
        )

    def test_optimal_pylon_spotting(self) -> None:
        path = optimal_pylon_spotting(
            self.test_instance,
            self.cfg.copy(),
        )
        self.assertTrue(len(path) > 0)
        self.assertListEqual(list(path[0]), list(self.cfg["start_inds"]))
        self.assertListEqual(list(path[-1]), list(self.cfg["dest_inds"]))

        plot_paths(
            self.test_instance, [path],
            buffer=0,
            out_path="test_optimal_pylon_spotting.png"
        )

    def test_ksp_routes(self) -> None:
        paths = ksp_routes(self.test_instance, self.cfg.copy(), 5)
        self.assertTrue(len(paths) == 5)
        plot_paths(self.test_instance, paths, out_path="test_route_ksp.png")

    def test_ksp_pylons(self) -> None:
        paths = ksp_pylons(self.test_instance, self.cfg.copy(), 5)
        self.assertTrue(len(paths) == 5)
        plot_paths(self.test_instance, paths, out_path="test_pylon_ksp.png")


if __name__ == '__main__':
    unittest.main()
