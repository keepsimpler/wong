# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_foldnet.ipynb (unless otherwise specified).

__all__ = ['FoldBlock', 'ExpandBlock', 'FoldNet', 'num_units', 'cal_num_params']

# Cell
from .imports import *
from .core import *
from .config import cfg, assert_cfg


# Cell
class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, Unit:nn.Module, ni:int, fold:int, stride:int=1, **kwargs):
        super(FoldBlock, self).__init__()
        self.ni, self.fold, self.stride = ni, fold, stride
        units = []
        for i in range(max(1,fold-1)):
            units += [Unit(ni, stride=1, **kwargs)]
        self.units = nn.ModuleList(units)

    def forward(self, *xs):
        xs = list(xs)
        if self.fold==1:
            xs[0] = xs[0] + self.units[0](xs[0])
            return xs
        for i in range(self.fold-1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs

# Cell
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

# Cell
class FoldNet(nn.Module):
    "A folded resnet, using Expand."
    def __init__(self, Stem, Unit, folds:tuple, ni:int, num_nodes:tuple,
                 bottle_scale:int=1, first_downsample:bool=False, tail_all:bool=True,
                 c_in:int=3, c_out:int=10, **kwargs):
        super(FoldNet, self).__init__()
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


# Cell
def num_units(folds, nodes):
    "calculate the number of all units in the backbone of FoldNet."
    num_units = (folds[0] - 1) * nodes[0]
    for i in range(len(folds)-1):
#         print(num_units)
        num_units += (folds[i+1]-1)*(nodes[i+1]-1) + max(0, folds[i+1]-folds[i])
    return(num_units)

# Cell
def cal_num_params(Stem, Unit, folds, nodes, ni, bottle_scale, tail_all, c_out):
    "calcuate the number of all params of FoldNet, according to hyper-params."
    m0 = Stem(3, no=ni)
    m1 = Unit(ni, nh=ni*bottle_scale)
    m2 = Classifier(ni*folds[-1] if tail_all else ni, c_out)
    return num_params(m0) + num_params(m1) * num_units(folds, nodes) + num_params(m2)