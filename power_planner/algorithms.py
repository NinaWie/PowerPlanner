import numpy as np
from power_planner import graphs
from power_planner.ksp import KSP
import time

VERBOSE = 0

__all__ = [
    "optimal_route", "optimal_pylon_spotting", "ksp_routes", "ksp_pylons"
]


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
    # extract path itself
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

    # run algorithm
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
    # run algorithm
    return run_ksp(graph, cfg, k, thresh=thresh, algorithm=algorithm)
