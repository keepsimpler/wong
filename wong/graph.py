# AUTOGENERATED! DO NOT EDIT! File to edit: dev/02_graph.ipynb (unless otherwise specified).

__all__ = ['complete_dag']

# Cell
from .imports import *

# Cell
def complete_dag(n:int):
    "Generate a complete directed acyclic graph, which corresponds to architecture of ResNet."
    G = nx.DiGraph()
    nodes = list(range(n))
    for id in nodes:
        for succ in range(id+1, n):
            G.add_edge(id, succ)
    return G