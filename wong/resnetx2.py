#AUTOGENERATED! DO NOT EDIT! File to edit: dev/07_resnetx2.ipynb (unless otherwise specified).

__all__ = ['FoldBlock', 'TransitionBlock', 'ExpandBlock', 'ResNetXExpand', 'ResNetXTransition']

#Cell
from .imports import *
from .core import *
from .config import cfg, assert_cfg


#Cell
class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, Unit:nn.Module, ni:int, fold:int, stride:int=1, **kwargs):
        super(FoldBlock, self).__init__()
        self.ni, self.fold, self.stride = ni, fold, stride
        units = []
        for i in range(fold-1):
            units += [Unit(ni, stride=1, **kwargs)]
        self.units = nn.ModuleList(units)

    def forward(self, *xs):
        xs = list(xs)
        for i in range(self.fold-1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs

#Cell
class TransitionBlock(nn.Module):
    "Transition block of folded ResNet"
    def __init__(self, Unit:nn.Module, ni:int, no:int, fold:int, stride:int=1, **kwargs):
        super(TransitionBlock, self).__init__()
        self.ni, self.no, self.fold, self.stride = ni, no, fold, stride
        units = []
        for i in range(fold):
            units += [Unit(ni, no=no, stride=stride, **kwargs)]
        self.units = nn.ModuleList(units)

    def forward(self, *xs):
        xs = list(xs)
        for i in range(len(xs)):
            xs[i] = self.units[i](xs[i])
        return xs

#Cell
class ExpandBlock(nn.Module):
    "Expand block of folded ResNet"
    def __init__(self, Unit:nn.Module, ni:int, fold1:int, fold2:int, stride:int=1, **kwargs):
        super(ExpandBlock, self).__init__()
        self.ni, self.fold1, self.fold2, self.stride = ni, fold1, fold2, stride
        units = []
        for i in range(fold2 - fold1):
            units += [Unit(ni, stride=1, **kwargs)]
        self.units = nn.ModuleList(units)
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, *xs):
        xs = list(xs)
        if self.stride == 2:
            for i in range(len(xs)):
                xs[i] = self.pool(xs[i])
        if self.fold2 <= self.fold1:
            return xs[:self.fold2]
        xs.reverse()
        for i in range(self.fold2 - self.fold1):
            xs.append(self.units[i](xs[-1]) + xs[-1])
        xs.reverse()
        return xs

#Cell
class ResNetXExpand(nn.Module):
    "A folded resnet, using Expand."
    def __init__(self, Stem, Unit, folds:tuple, ni:int, num_nodes:tuple,
                 bottle_scale:int=1, first_downsample:bool=False, tail_all:bool=True,
                 c_in:int=3, c_out:int=10, **kwargs):
        super(ResNetXExpand, self).__init__()
        num_stages = len(num_nodes)
        nh = int(ni * bottle_scale)
        strides = [1 if i==0 and not first_downsample else 2 for i in range(num_stages)]
        folds = [1] + folds #[fold*exp**i for i in range(num_stages)]

        self.stem = Stem(c_in, no=ni) # , deep_stem

        units = []
        for i, (nu, stride) in enumerate(zip(num_nodes, strides)):
            for j in range(nu):
                if j == 0: # the first node(layer) of each stage
                    units += [ExpandBlock(Unit, ni, fold1 = folds[i], fold2=folds[i+1], stride=stride, nh=nh, **kwargs)]
                else:
                    units += [FoldBlock(Unit, ni, fold=folds[i+1], stride=1, nh=nh, **kwargs)]

        self.units = nn.ModuleList(units)

        if tail_all:
            self.classifier = Classifier(ni*folds[-1], c_out) #
        else:
            self.classifier = Classifier(ni, c_out)
        self.folds = folds
        self.num_nodes = num_nodes
        self.tail_all = tail_all
        init_cnn(self)

    def forward(self, x):
        x = self.stem(x)
        xs = [x] #self.init(x)
        for unit in self.units:
            xs = unit(*xs)
        if self.tail_all:
            x = torch.cat(xs,1)
        else:
            x = xs[0]

        x = self.classifier(x)
        return x


#Cell
class ResNetXTransition(nn.Module):
    "A folded resnet using Transition."
    def __init__(self, Stem, Unit, fold:int, nis:tuple, num_nodes:tuple,
                 bottle_scale:int=1, first_downsample:bool=False, tail_all:bool=True,
                 c_in:int=3, c_out:int=10, **kwargs):
        super(ResNetXTransition, self).__init__()
        num_stages = len(num_nodes)
        nhs = [int(ni * bottle_scale) for ni in nis]
        strides = [1 if i==0 and not first_downsample else 2 for i in range(num_stages)]

        self.stem = Stem(c_in, no=nis[0]) # , deep_stem
        self.expand = ExpandBlock(Unit, ni=nis[0], fold1=1, fold2=fold, stride=strides[0], nh=nhs[0], **kwargs)

        units = []
        for i, (nu, stride) in enumerate(zip(num_nodes, strides)):
            if i != 0:
                units += [TransitionBlock(Unit, ni=nis[i-1], no=nis[i], fold=fold, stride=stride, nh=nhs[i-1], **kwargs)]
            for j in range(nu):
#                 if j == 0: # the first node(layer) of each stage
#                     units += [ExpandBlock(Unit, ni, fold1 = folds[i], fold2=folds[i+1], stride=stride, nh=nh, **kwargs)]
#                 else:
                units += [FoldBlock(Unit, ni=nis[i], fold=fold, stride=1, nh=nhs[i], **kwargs)]

        self.units = nn.ModuleList(units)

        if tail_all:
            self.classifier = Classifier(nis[-1]*fold, c_out) #
        else:
            self.classifier = Classifier(nis[-1], c_out)
        self.fold = fold
        self.num_nodes = num_nodes
        self.tail_all = tail_all
        init_cnn(self)

    def forward(self, x):
        x = self.stem(x)
        xs = self.expand(*[x])
        for unit in self.units:
            xs = unit(*xs)
        if self.tail_all:
            x = torch.cat(xs,1)
        else:
            x = xs[0]

        x = self.classifier(x)
        return x
