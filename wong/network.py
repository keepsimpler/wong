#AUTOGENERATED! DO NOT EDIT! File to edit: dev/03_network.ipynb (unless otherwise specified).

__all__ = ['NodeOP', 'NetworkOP']

#Cell
from .imports import *
from .core import *
from .config import cfg, assert_cfg
from .graph import *

#Cell
class NodeOP(nn.Module):
    "The Operation of inner nodes in the network."
    def __init__(self, ni:int, no:int, nh:int, Unit:nn.Module, **kwargs):
        super(NodeOP, self).__init__()
        self.unit = Unit(ni, no, nh, **kwargs)

    def forward(self, *inputs):
        sum_inputs = sum(inputs)
        out = self.unit(sum_inputs)
        return out


#Cell
class NetworkOP(nn.Module):
    "The operations along a DAG network."
    def __init__(self, G:nx.DiGraph, ni:int, no:int, Unit:nn.Module, **kwargs):
        super(NetworkOP, self).__init__()
        self.G = G
        self.n = G.graph['n'] # number of nodes
        self.nodeops = nn.ModuleList()
        for id in G.nodes(): # for each node
            if id == 0:  # if is the unique input node, do nothing
                continue
            elif id == self.n:  # if is the unique output node
                # then, concat its predecessors
                n_preds = len([*G.predecessors(id)])
                self.nodeops += [IdentityMapping(n_preds * ni, no)]
            else:  # if is the inner node
                self.nodeops += [NodeOP(ni, ni, ni, Unit, **kwargs)]

    def forward(self, x):
        results = {}
        results[-1] = x  # input data is the result of the unique input node
        for id in self.G.nodes(): # for each node
            if id == -1:  # if is the input node, do nothing
                continue
            # get the results of all predecessors
            inputs = [results[pred]  for pred in self.G.predecessors(id)]
            if id == self.n: # if is the output node
                cat_inputs = torch.cat(inputs, dim=1) # concat results of all predecessors
                if self.efficient:
                    return cp.checkpoint(self.nodeops[id], cat_inputs)
                else:
                    return self.nodeops[id](cat_inputs)
            else: # if is inner nodes
                if self.efficient:
                    results[id] = cp.checkpoint(self.nodeops[id], *inputs)
                else:
                    results[id] = self.nodeops[id](*inputs)

            # 删除前驱结点result中，不再需要的result
            for pred in self.G.predecessors(id):  # 获得节点的所有前驱结点
                succs = list(self.G.successors(pred))  # 获得每个前驱结点的所有后继节点
                # 如果排名最后的后继节点是当前节点，说明该前驱结点的result不再被后续的节点需要，可以删除
                if max(succs) == id:
                    del results[pred]

