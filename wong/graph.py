# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_graph.ipynb (unless otherwise specified).

__all__ = ['resnet_dag']

# Cell
from .imports import *

# Cell
def resnet_dag(num_nodes:tuple):
    "Generate the DAG corresponding to the architecture of ResNet. Input is numbers of nodes of all stages."
    num_stages = len(num_nodes)
    # first stage add one input node, last stage add one `idmap` node and one output node, the other stages add one `idmap` node
    num_all_nodes = sum(num_nodes) + 1 + (num_stages-1) + 1
    G = nx.DiGraph()
    for i in range(num_all_nodes):
        if i < 1 + num_nodes[0]: # nodes of first stage
            if i == 0:
                G.add_node(i, stage = 0, optype = 'input') # the first node of first stage is input node
            else:
                G.add_node(i, stage = 0, optype = 'resnet_bottleneck') # the other nodes has `resnet_bottleneck` operation
            for succ in range(i+1, 1 + num_nodes[0] + 1 + 1): # connect to all other nodes before the acrossing nodes of second stage
                G.add_edge(i, succ)
        elif i < 1 + num_nodes[0] + num_nodes[1] + 1: # nodes of second stage
            if i == 1 + num_nodes[0]: # the first node of second stage has `idmap` operation
                G.add_node(i, stage = 1, optype = 'idmap')
            else:
                G.add_node(i, stage = 1, optype = 'resnet_bottleneck')
            for succ in range(i+1, 1 + num_nodes[0] + num_nodes[1] + 1 + 1 + 1): # connect to all other nodes before the acrossing nodes of second stage
                if not (i == 1 + num_nodes[0] and succ == 1 + num_nodes[0] + 1): # except the connection between two acrossing nodes
                    G.add_edge(i, succ)
        elif i < 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1: # nodes of third stage
            if i == 1 + num_nodes[0] + num_nodes[1] + 1: # the first node of third stage has `idmap` operation
                G.add_node(i, stage = 2, optype = 'idmap')
            else:
                G.add_node(i, stage = 2, optype = 'resnet_bottleneck')
            for succ in range(i+1, 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1 + 1 + 1): # connect to all other nodes before the acrossing nodes of second stage
                if not (i == 1 + num_nodes[0] + num_nodes[1] + 1 and succ == 1 + num_nodes[0] + num_nodes[1] + 1 + 1): # except the connection between two acrossing nodes
                    G.add_edge(i, succ)
        elif i < 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1 + num_nodes[3] + 2: # nodes of fourth stage
            if i == 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1: # the first node of fourth stage has `idmap` operation
                G.add_node(i, stage = 3, optype = 'idmap')
            elif i == 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1 + num_nodes[3] + 2 - 1: # the last node is output node
                G.add_node(i, stage = 3, optype = 'output')
            else:
                G.add_node(i, stage = 3, optype = 'resnet_bottleneck')
            for succ in range(i+1, 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1 + num_nodes[3] + 2): # connect to all other nodes in the same stage
                if not (i == 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1 and succ == 1 + num_nodes[0] + num_nodes[1] + 1 + num_nodes[2] + 1 + 1): # except the connection between two acrossing nodes
                    G.add_edge(i, succ)
    G.graph['n'] = num_all_nodes
    return G