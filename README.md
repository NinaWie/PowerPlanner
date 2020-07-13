[![Build Status]]

# Optimizing power infrastructure layout

![GUI](assets/GUI.png "UI for optimal power line layout")

Given resistance costs for a raster of geo locations, the goal is to compute the optimal power line layout from a given start point to a given end point. The approach is to represent raster cells as vertices in a graph, place edges between them based on the minimal and maximal distance of the power towers, and define edge costs based on the underlying cost surface.

## Installation

The library itself has few major dependencies (see [setup.py](setup.py)). 
* `kivy` is required for the GUI
* `numpy` and `matplotlib`

If wanted, create a virtual environment and activate it:

```sh
python3 -m venv env
source env/bin/activate
```

Install the repository in editable mode:

```sh
git clone https://github.com/NinaWie/PowerPlanner
cd PowerPlanner
pip install -e .
```

### Install manually:

NOTE: If you do not want to use the UI, you can 

```sh
pip install -r requirements.txt
```

## Optimal power infrastructure planning

### UI

A small python GUI serves as a demo app for the toolbox. Start the UI by running
```sh
python ui.py
```

### Demo notebook

In the [demo](demo.ipynb) notebook, the most important functions are explained and visualizations show the input data and outputs for each processing step.

### Codebase

If you want to run one instance without UI, execute 
```sh
python main.py [-h] [-cluster] [-i INSTANCE] [-s SCALE]
```

Optional arguments:
  -h, --help:
  -cluster:
  -i INSTANCE: the id of the instance to use (here one of ch, de, belgium)
  -s SCALE: the resolution: 1 for 10m, 2 for 20m, 5 for 50m resolution

Specify other hyperparameters in the config file of the instance (example will be added soon).

## Algorithms

![Angle](assets/cost_angle_visualization.png "Visualization of LCP computation")

Different types of costs are considered and weighted: Angle costs, edge costs (cost for putting a cable above a particular position), and vertex costs (cost to place a pylon).
An algorithm was developed to deal with angle costs, which is not possible with normal shortest path algorithms (e.g. Dijkstra). The code can be found as ```sp_dag``` or ```sp_bf``` dependent whether it is a directed acyclic graph or not, and is implemented in `numba` for efficiency.
