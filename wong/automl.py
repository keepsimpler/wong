#AUTOGENERATED! DO NOT EDIT! File to edit: dev/05_automl.ipynb (unless otherwise specified).

__all__ = ['get_num_nodes']

#Cell
from .imports import *

#Cell
def get_num_nodes(num_all_nodes:int=64, num_stages:int=3, fold:int=4):
    "generate num of nodes of all stages randomly, constrint to a condition based on fold"
    success = False
    while not success:
        num_nodes = [0] + sorted([random.randint(1, num_all_nodes-1) for _ in range(num_stages)]) + [num_all_nodes]
        num_nodes = [num_nodes[i] - num_nodes[i-1]  for i in range(1, len(num_nodes))]
        larger_than_fold = [e >= 2*(fold-1) for e in num_nodes]
        if all(larger_than_fold): success = True
    return num_nodes
