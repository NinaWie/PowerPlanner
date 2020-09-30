import numpy as np
from power_planner import graphs
from power_planner.ksp import KSP
from power_planner.plotting import plot_path, plot_k_sp
import time
from types import SimpleNamespace

VERBOSE = 0

# def transform_instance_decorator(method):
#     """
#     Decorator to transform np array into instance and corridor seperated
#     """

#     def instance_wrapper(instance, *args):
#         # make forbidden region array
#         project_region = np.ones(instance.shape)
#         project_region[np.isnan(instance)] = 0

#         # modify instance to have a 3-dimensional input as required
#         max_val = np.max(instance[~np.isnan(instance)])
#         instance[np.isnan(instance)] = max_val
#         instance = np.array([instance])
#         # change new_args here
#         return method(instance, *args)

#     return instance_wrapper


def transform_instance(instance, fillval=np.max):
    """
    transform instance to get instance and corridor separately
    """
    # make forbidden region array
    project_region = np.ones(instance.shape)
    project_region[np.isnan(instance)] = 0
    project_region[instance == np.inf] = 0
    # modify instance to have a 3-dimensional input as required
    max_val = fillval(instance[~np.isnan(instance)])
    instance[np.isnan(instance)] = max_val
    # add dimension to instance
    instance = np.array([instance])
    # change new_args here
    return instance, project_region


def optimal_route(instance, cfg):
    """
    Compute the (angle-) optimal path through a grid
    @params:
        instance (2D numpy array, forbidden regions are nans)
        cfg: configuration details (weights etc)
    @returns:
        a single optimal path (list of X Y coordinates)
    """
    instance, project_region = transform_instance(instance)

    # initialize graph
    graph = graphs.ImplicitLG(instance, project_region, verbose=VERBOSE)

    # set the ring to the 8-neighborhood
    cfg["pylon_dist_min"] = 0.9
    cfg["pylon_dist_max"] = 1.5

    # compute path
    tic_raster = time.time()
    path, _, _ = graph.single_sp(**cfg)
    toc_raster = time.time() - tic_raster
    print("DONE, processing time:", toc_raster)
    return path


def optimal_pylon_spotting(instance, cfg, corridor=None):
    """
    Compute the (angle-) optimal pylon spotting
    @params:
        instance (2D numpy array, forbidden regions are nans)
        cfg: configuration details (weights etc)
        corridor: relevant region --> either
            - a path, e.g. output of optimal_route, or
            - a 2D array with 0: do not consider, 1: consider
    @returns:
        a single optimal path of pylons (list of X Y coordinates)
    """
    instance, project_region = transform_instance(instance)

    if corridor is not None:
        if isinstance(corridor, np.ndarray) and corridor.shape != 2:
            raise ValueError("corridor must be a two-dimensional ndarray")
        # first option: corridor is a path
        if corridor.shape[1] == 2:
            pylon_spotting_corr = np.zeros(project_region.shape)
            for (i, j) in corridor:
                if project_region[i, j] > 0:
                    pylon_spotting_corr[i, j] = 1
            project_region = pylon_spotting_corr * project_region
        # second option: corridor is given --> intersection of corr and forb
        else:
            project_region = corridor * project_region

    # Pylon spotting
    graph = graphs.ImplicitLG(instance, project_region, verbose=VERBOSE)
    path, _, _ = graph.single_sp(**cfg)
    return path


# ----------------------------------- KSP  ---------------------------------
def run_ksp(graph, cfg, k, thresh=10, algorithm=KSP.find_ksp):
    """
    Build the shortest path trees and compute k diverse shortest paths
    """
    # construct sp trees
    _ = graph.sp_trees(**cfg)
    # compute k shortest paths
    ksp_processor = KSP(graph)
    ksp_out = algorithm(ksp_processor, k, thresh=thresh)
    ksp_paths = [k[0] for k in ksp_out]
    return ksp_paths


def ksp_routes(instance, cfg, k, thresh=10, algorithm=KSP.find_ksp):
    """
    Compute the (angle-) optimal k diverse shortest paths through a grid
    @params:
        instance (2D numpy array, forbidden regions are nans)
        cfg: configuration details (weights etc)
        k: number of paths to compute
        thresh: distance threshold, dependent on algorithm
        algorithm: algorithm for computation of KSP
    @returns:
        A list of paths (each path is again a list of X Y coordinates)
    """
    instance, project_region = transform_instance(instance)

    # initialize graph
    graph = graphs.ImplicitLG(instance, project_region, verbose=VERBOSE)

    # set the ring to the 8-neighborhood
    cfg["pylon_dist_min"] = 1
    cfg["pylon_dist_max"] = 1.5

    return run_ksp(graph, cfg, k, thresh=thresh, algorithm=algorithm)


def ksp_pylons(instance, cfg, k, thresh=10, algorithm=KSP.find_ksp):
    """
    Compute the (angle-) optimal k diverse shortest path of PYLONS
    @params:
        instance (2D numpy array, forbidden regions are nans)
        cfg: configuration details (weights etc), start and dest!
        k: number of paths to compute
        thresh: distance threshold, dependent on algorithm
        algorithm: algorithm for computation of KSP
    @returns:
        A list of paths (each path is again a list of X Y coordinates)
    """
    instance, project_region = transform_instance(instance)
    # initialize graph
    graph = graphs.ImplicitLG(instance, project_region, verbose=VERBOSE)
    return run_ksp(graph, cfg, k, thresh=thresh, algorithm=algorithm)


# if __name__ == "__main__":
#     test_instance = np.random.rand(100, 100)
#     num_nans = 100
#     forb_x = (np.random.rand(num_nans) * 100).astype(int)
#     forb_y = (np.random.rand(num_nans) * 100).astype(int)
#     test_instance[forb_x, forb_y] = np.nan

#     # create configuration
#     cfg = SimpleNamespace()
#     cfg.start_inds = np.array([6, 6])
#     cfg.dest_inds = np.array([94, 90])
#     cfg.max_angle_lg = np.pi / 2

#     path = optimal_route(test_instance, cfg)  # , 5)
#     print(path)
#     plot_path(test_instance, path, buffer=0, out_path="test_optimal_route.png")
