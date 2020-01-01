# AUTOGENERATED! DO NOT EDIT! File to edit: dev/01_config.ipynb (unless otherwise specified).

__all__ = []

# Cell
from .imports import *

# Cell
from yacs.config import CfgNode as CN

# Cell
_C = CN()

# Cell
_C.GRAPH = CN()

# Cell
_C.GRAPH.NUM_STAGES = 4
_C.GRAPH.NUM_NODES = (3, 8, 36, 3)
_C.GRAPH.NUM_CHANNELS = (64, 128, 256, 512)