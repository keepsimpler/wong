#AUTOGENERATED! DO NOT EDIT! File to edit: dev/04_resnetx.ipynb (unless otherwise specified).

__all__ = ['get_pred', 'ResNetX']

#Cell
from .imports import *
from .core import *
from .config import cfg

#Cell
def get_pred(l:int, d:int=1):
    "get predecessor layer id."
    assert l >= 1
    if l < d or d == 1:  # if the current layer index is less than the fold depth, or if fold depth == 1
        pred = l - 1
    else:
        remainder = l % (d-1)
        if remainder == 0:
            pred = l - 2 * (d-1)
        else:
            pred = l - 2 * remainder
#         remainder1 = l % (2*(d-1))
#         if 1 <= remainder1 <= d-1:
#             pred = l - 2 * remainder1
#         else:
#             remainder2 = (remainder1 + d-1) % (2*(d-1))
#             pred = l - 2 * remainder2
    return pred

#Cell
class ResNetX(nn.Module):
    "A folded resnet."
    def __init__(self, Start, Unit, fold:int, ni:int, num_nodes:tuple, base:int=64, exp:int=2,
                 bottle_scale:int=1, first_downsample:bool=False, c_in:int=3, c_out:int=10, **kwargs):
        super(ResNetX, self).__init__()
        # fold depth should be less than the sum length of any two neighboring stages

        self.fold = fold
        origin_ni = ni
        num_stages = len(num_nodes)
        nhs = [base * exp ** i for i in range(num_stages)]
        nos = [nh * bottle_scale for nh in nhs]
        strides = [1 if i==0 and not first_downsample else 2 for i in range(num_stages)]
#         print('nhs=', nhs, 'nos=', nos, 'nus=', nus, 'strides=', strides)

        self.start = Start(c_in, ni)

        units = []
        idmappings = []
        cur = 1
        for i, (nh, no, nu, stride) in enumerate(zip(nhs, nos, num_nodes, strides)):
            for j in range(nu):
                if j == 0: # the first node(layer) of each stage
                    units += [Unit(ni, no, nh, stride=stride, **kwargs)]
                else:
                    units += [Unit(no, no, nh, stride=1, **kwargs)]

                pred = get_pred(cur, fold) #
                diff = layer_diff(cur, pred, num_nodes)
                assert diff == 0 or diff == 1 or (diff == 2 and pred == 0), \
                       'cur={}, pred={}, diff={} is not allowed.'.format(cur, pred, diff)
                if diff == 0:
                    idmappings += [IdentityMapping(no, no, stride=1)]
                elif diff == 1:
                    idmappings += [IdentityMapping(ni, no, stride=stride)]
                elif diff == 2:
                    idmappings += [IdentityMapping(origin_ni, no, stride=stride)]
                cur += 1
            ni = no
        self.units = nn.ModuleList(units)
        self.idmappings = nn.ModuleList(idmappings)

        self.classifier = Classifier(nos[-1], c_out)
        init_cnn(self)

    def forward(self, x):
        results = {}
        results[0] = self.start(x)
        cur = 0
        for i, (unit, idmapping) in enumerate(zip(self.units, self.idmappings)):
            cur += 1
            pred = get_pred(cur, self.fold)
            results[cur % (2*self.fold-1)] = unit(results[(cur-1) % (2*self.fold-1)]) + idmapping(results[pred % (2*self.fold-1)])
        x = results[cur % (2*self.fold-1)]

        x = self.classifier(x)
        return x
