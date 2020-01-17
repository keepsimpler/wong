import torch
import torch.nn as nn
import torch.utils.checkpoint as cp # checkpointing to elimit memory

torch.backends.cudnn.benchmark = True

import sys
from collections import OrderedDict
from functools import partial
import itertools

import numpy as np
from numpy.linalg import matrix_power # for calculation of paths in graph
import matplotlib.pylab as plt

import networkx as nx

