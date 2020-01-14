#AUTOGENERATED! DO NOT EDIT! File to edit: dev/01_config.ipynb (unless otherwise specified).

__all__ = ['assert_cfg', 'cfg']

#Cell
from .imports import *

#Cell
from yacs.config import CfgNode as CN

#Cell
_C = CN()
_C.URL = ''  # url for pretrained model storage


#Cell
_C.GRAPH = CN()

#Cell
_C.GRAPH.NUM_STAGES = 4
_C.GRAPH.NUM_NODES = (3, 8, 36, 3)
_C.GRAPH.NUM_CHANNELS = (64, 128, 256, 512)
_C.GRAPH.STEM = ''
_C.GRAPH.UNIT = ''
_C.GRAPH.CONN = ''
_C.GRAPH.FOLD = 1
_C.GRAPH.NI = 64
_C.GRAPH.START_ID = 0
_C.GRAPH.BASE = 64
_C.GRAPH.EXP = 2
_C.GRAPH.BOTTLE_SCALE = 4
_C.GRAPH.FIRST_DOWNSAMPLE = False
_C.GRAPH.DEEP_STEM = False

#Cell
def assert_cfg(cfg):
    "Assert config options"
    assert cfg.GRAPH.NUM_STAGES == len(cfg.GRAPH.NUM_NODES), 'num_stages should equal to length of num_nodes'

#Cell
cfg = _C
