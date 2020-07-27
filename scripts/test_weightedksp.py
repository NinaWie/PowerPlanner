# import warnings
import numpy as np
from types import SimpleNamespace
import warnings
import matplotlib.pyplot as plt

from power_planner import graphs

# EXAMPLE DATA
instance = np.random.rand(1, 100, 100)
instance_corr = np.zeros((100, 100))
# corridor: 1 is feasible region, 0 is forbidden
# pad at the border, necessary for weighted_graph processing (np.roll function)
instance_corr[6:-6, 6:-6] = 1

instance_corr[:]
cfg = SimpleNamespace(
    **{
        # angle weight doesn't matter
        "ANGLE_WEIGHT": 0,
        # maximum angle -> needed to define half donut, can stay like that
        "MAX_ANGLE": 1.57079,
        "MAX_ANGLE_LG": 1.57079,
        # scale can stay 1 as well, probably not used
        "scale": 1,
        # you need to set this, the pixel-wise minimum and maximum distance
        # between pylons
        "PYLON_DIST_MAX": 5.0,
        "PYLON_DIST_MIN": 3.0,
        # if you have only one category:
        "class_weights": [1],
        "layer_classes": ["resistance"],
        # you need to set this, the start and destination points
        "dest_inds": np.array([93, 90]),
        "start_inds": np.array([7, 9])
    }
)

graph = graphs.WeightedKSP(instance, instance_corr)
# single shortest path (building the graph)
path, path_cost, cost_sum = graph.single_sp(**vars(cfg))
print("output path:", path)

graph.get_shortest_path_tree()
# to output k paths
ksp = graph.find_ksp(5, overlap=0.5)

# ksp ist a list of form:
# [[path1, path_costs1, cost_sum1], [path2, path_costs2, cost_sum2], ...]
ksp_output_paths = [k[0] for k in ksp]
print(ksp_output_paths)

plt.figure(figsize=(10, 10))
plt.imshow(np.tile(np.expand_dims(instance[0], 2), 3))
for path in ksp_output_paths:
    path = np.asarray(path)
    # switched 0 and 1 because scatter and imshow have switched axes
    plt.scatter(path[:, 1], path[:, 0], s=50)
plt.savefig("test_ksp.png")
