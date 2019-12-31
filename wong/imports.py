import torch
import torch.nn as nn
import torch.utils.checkpoint as cp # checkpointing to elimit memory

torch.backends.cudnn.benchmark = True

import sys
from collections import OrderedDict
import numpy as np
from numpy.linalg import matrix_power # for calculation of paths in graph
import networkx as nx

import matplotlib.pylab as plt