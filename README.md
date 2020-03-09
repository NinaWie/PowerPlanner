[![Build Status]]

# Optimizing infrastructure layout for power lines

Given resistance costs for a raster of geo locations, the goal is to compute the optimal power line layout from a given start point to a given end point. The approach is to represent raster cells as vertices in a graph, place edges between them based on the minimal and maximal distance of the power towers, and define edge costs based on the underlying cost surface.

## Installation

The library itself has few major dependencies (see [setup.py](setup.py)). 
* One of `graph-tool` or `networkx` is required
* `numpy` and `matplotlib`

### Install the required packaged by defining an virtual environment:

Create a virtual environment:

```sh
python3 -m venv env
```

Activate the environment:

```sh
source env/bin/activate
```

Install in editable mode for development:

```sh
pip install  -r requirements.txt
```

### Install with pip in editable mode for development:

```console
pip install -e .
```

## Usage

Run 
```
python main.py
```
to execute the algorithm. Specify input paths and hyperparameters in the beginning of the file

The class `WeightedGraph` provides an interface to build a graph and compute edge costs based on a cost array with either networkx or graph-tool.
