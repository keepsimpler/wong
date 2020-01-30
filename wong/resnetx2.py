#AUTOGENERATED! DO NOT EDIT! File to edit: dev/07_resnetx2.ipynb (unless otherwise specified).

__all__ = ['InitBlock', 'FoldBlock', 'ResNetX2']

#Cell
from .imports import *
from .core import *
from .config import cfg, assert_cfg


#Cell
class InitBlock(nn.Module):
    "Init block of folded ResNet"
    def __init__(self, Unit:nn.Module, ni:int, fold:int, stride:int=1, **kwargs):
        super(InitBlock, self).__init__()
        self.ni, self.fold = ni, fold
        units = []
        for i in range(fold-1):
            units += [Unit(ni, stride=stride, **kwargs)]
        self.units = nn.ModuleList(units)

    def forward(self, x):
        xs = [x]
        for i in range(self.fold-1):
            xs += [xs[i] + self.units[i](xs[i])]
        xs.reverse()
        return xs

#Cell
class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, Unit:nn.Module, ni:int, fold:int, stride:int=1, **kwargs):
        super(FoldBlock, self).__init__()
        self.ni, self.fold, self.stride = ni, fold, stride
        units = []
        for i in range(fold-1):
            if i==0:
                units += [Unit(ni, stride=stride, **kwargs)]
            else:
                units += [Unit(ni, stride=1, **kwargs)]
        self.units = nn.ModuleList(units)
        if stride==2:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, *xs):
        xs = list(xs)
        for i in range(self.fold-1):
            if self.stride==2:
                xs[i+1] = self.pool(xs[i+1])
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        if self.stride==2:
            xs[0] = self.pool(xs[0])
        xs.reverse()
        return xs

#Cell
class ResNetX2(nn.Module):
    "A folded resnet."
    def __init__(self, Stem, Unit, fold:int, ni:int, num_nodes:tuple,
                 bottle_scale:int=1, first_downsample:bool=False,
                 c_in:int=3, c_out:int=10, **kwargs):
        super(ResNetX2, self).__init__()
        num_stages = len(num_nodes)
        nh = ni * bottle_scale
        strides = [1 if i==0 and not first_downsample else 2 for i in range(num_stages)]

        self.stem = Stem(c_in, no=ni) # , deep_stem
        self.init = InitBlock(Unit, ni, fold, nh=nh)

        units = []
        idmappings = []
        cur = 1
        for i, (nu, stride) in enumerate(zip(num_nodes, strides)):
            for j in range(nu):
                if j == 0: # the first node(layer) of each stage
                    units += [FoldBlock(Unit, ni, fold, stride=stride, nh=nh, **kwargs)]
                else:
                    units += [FoldBlock(Unit, ni, fold, stride=1, nh=nh, **kwargs)]

        self.units = nn.ModuleList(units)

        self.classifier = Classifier(ni, c_out) #*fold
        self.fold = fold
        self.num_nodes = num_nodes
        init_cnn(self)

    def forward(self, x):
        x = self.stem(x)
        xs = self.init(x)
        for unit in self.units:
            xs = unit(*xs)
        x =  torch.cat(xs,1)

        x = self.classifier(x)
        return x