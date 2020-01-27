#AUTOGENERATED! DO NOT EDIT! File to edit: dev/04_resnetx.ipynb (unless otherwise specified).

__all__ = ['get_pred', 'layer_diff', 'ResNetX', 'resnet_local_to_pretrained', 'resnetx']

#Cell
from .imports import *
from .core import *
from .config import cfg, assert_cfg

from torchvision.models.utils import load_state_dict_from_url


#Cell
def get_pred(l:int, d:int=1, start_id:int=None, end_id:int=None):
    "get predecessor layer id."
    if start_id is None: start_id = d
    if end_id is None: end_id = l
    assert l >= 1 and start_id >= d and end_id > start_id
    if l < start_id or l > end_id or d == 1:  # if the current layer index is less than the fold depth, or if fold depth == 1
        pred = l - 1
    else:
        remainder = (l-1-(start_id-d)) % (d-1)
        pred = l - 2 * (1+remainder)
    return pred

#Cell
def layer_diff(cur:int, pred:int, num_nodes:tuple):
    "layer difference between the current layer and the predecessor layer."
    assert cur > pred
    num_nodes = (1,) + num_nodes
    cumsum = 0  # start with 0
    for i, num in enumerate(num_nodes):
        if cumsum <= cur < cumsum + num:
            cur_layer = i
        if cumsum <= pred < cumsum + num:
            pred_layer = i
        cumsum += num
    diff = cur_layer - pred_layer
    return diff

#Cell
class ResNetX(nn.Module):
    "A folded resnet."
    def __init__(self, Stem, Unit, Conn, fold:int, ni:int, num_nodes:tuple, start_id:int=None, end_id:int=None,
                 base:int=64, exp:int=2, bottle_scale:int=1, first_downsample:bool=False,
                 c_in:int=3, c_out:int=10, **kwargs):
        super(ResNetX, self).__init__()
        # fold depth should be less than the sum length of any two neighboring stages

        if start_id < fold: start_id = fold
        origin_ni = ni
        num_stages = len(num_nodes)
        nhs = [base * exp ** i for i in range(num_stages)]
        nos = [nh * bottle_scale for nh in nhs]
        strides = [1 if i==0 and not first_downsample else 2 for i in range(num_stages)]
#         print('nhs=', nhs, 'nos=', nos, 'nus=', nus, 'strides=', strides)

        self.stem = Stem(c_in, ni) # , deep_stem

        units = []
        idmappings = []
        cur = 1
        for i, (nh, no, nu, stride) in enumerate(zip(nhs, nos, num_nodes, strides)):
            for j in range(nu):
                if j == 0: # the first node(layer) of each stage
                    units += [Unit(ni, no, nh, stride=stride, **kwargs)]
                else:
                    units += [Unit(no, no, nh, stride=1, **kwargs)]

                pred = get_pred(cur, fold, start_id, end_id) #
                diff = layer_diff(cur, pred, num_nodes)
                assert diff == 0 or diff == 1 or (diff == 2 and pred == 0), \
                       'cur={}, pred={}, diff={} is not allowed.'.format(cur, pred, diff)
#                 print('fold = {} , cur = {} , pred = {} ,diff = {}'.format(fold, cur, pred, diff))
                if diff == 0:
                    idmappings += [Conn(no, no, stride=1)]
                elif diff == 1:
                    idmappings += [Conn(ni, no, stride=stride)]
                elif diff == 2:
                    idmappings += [Conn(origin_ni, no, stride=stride)]
                cur += 1
            ni = no
        self.units = nn.ModuleList(units)
        self.idmappings = nn.ModuleList(idmappings)

        self.classifier = Classifier(nos[-1], c_out)
        self.fold, self.start_id, self.end_id = fold, start_id, end_id
        self.num_nodes = num_nodes
        init_cnn(self)

    def forward(self, x):
        results = {}
        results[0] = self.stem(x)
        cur = 0
        for i, (unit, idmapping) in enumerate(zip(self.units, self.idmappings)):
            cur += 1
            pred = get_pred(cur, self.fold, self.start_id, self.end_id)
            results[cur % (2*self.fold-1)] = unit(results[(cur-1) % (2*self.fold-1)]) + idmapping(results[pred % (2*self.fold-1)])
        x = results[cur % (2*self.fold-1)]

        x = self.classifier(x)
        return x

    def my_load_state_dict(self, state_dict, local_to_pretrained):
        error_msgs = []
        def load(module, prefix=''):
            local_name_params = itertools.chain(module._parameters.items(), module._buffers.items())
            local_state = {k: v.data for k, v in local_name_params if v is not None}

            new_prefix = local_to_pretrained.get(prefix, 'none')
            for name, param in local_state.items():
                key = new_prefix + name
                if key in state_dict:
#                     print(key)
                    input_param = state_dict[key]

                    if input_param.shape != param.shape:
                        # local shape should match the one in checkpoint
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                          'the shape in current model is {}.'
                                          .format(key, input_param.shape, param.shape))
                        continue

                    try:
                        param.copy_(input_param)
                    except Exception:
                        error_msgs.append('While copying the parameter named "{}", '
                                          'whose dimensions in the model are {} and '
                                          'whose dimensions in the checkpoint are {}.'
                                          .format(key, param.size(), input_param.size()))

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(self)
        load = None # break load->load reference cycle



#Cell
def resnet_local_to_pretrained(num_nodes, fold, start_id, end_id):
    "mapping from local state_dict to pretrained state_dict. the pretrained model is restricted to torchvision.models.resnet."
    local_to_pretrained = {  # mapping from the names of local modules to the names of pretrained modules
        'stem.0.': 'conv1.',
        'stem.1.': 'bn1.',
    }

    cumsum = 0
    for i, num in enumerate(num_nodes):
        for j in range(num):
            key = 'units.' + str(cumsum + j) + '.'
            value = 'layer' + str(i+1) + '.' + str(j) + '.'
            downsample0 = 'layer' + str(i+1) + '.0.' + 'downsample.0.'
            downsample1 = 'layer' + str(i+1) + '.0.' + 'downsample.1.'

            pred = get_pred(cumsum + j + 1, fold, start_id, end_id) #
            diff = layer_diff(cumsum + j + 1, pred, num_nodes)
            if diff == 1:
                idmapping0 = 'idmappings.' + str(cumsum + j) + '.unit.0.'
                idmapping1 = 'idmappings.' + str(cumsum + j) + '.unit.1.'
#                     print(idmapping0, downsample0)
#                     print(idmapping1, downsample1)
                local_to_pretrained[idmapping0] = downsample0
                local_to_pretrained[idmapping1] = downsample1

            for a, b in zip(['1.','2.','4.','5.','7.','8.'], ['conv1.','bn1.','conv2.','bn2.','conv3.','bn3.']):
#                     print (key + a, value + b)
                local_to_pretrained[key + a] = value + b

        cumsum += num

    return local_to_pretrained


#Cell
def resnetx(default_cfg:dict, cfg_file:str=None, cfg_list:list=None, pretrained:bool=False, **kwargs):
    "wrapped resnetx"
    assert default_cfg.__class__.__module__ == 'yacs.config' and default_cfg.__class__.__name__ == 'CfgNode'
    cfg = default_cfg
    if cfg_file is not None: cfg.merge_from_file(cfg_file)
    if cfg_list is not None: cfg.merge_from_list(cfg_list)
    assert_cfg(cfg)
    cfg.freeze()

    Stem = getattr(sys.modules[__name__], cfg.GRAPH.STEM)
    Unit = getattr(sys.modules[__name__], cfg.GRAPH.UNIT)
    Conn = getattr(sys.modules[__name__], cfg.GRAPH.CONN)
    # start_id >= fold + 1, fold <= 6
    model = ResNetX(Stem=Stem, Unit=Unit, Conn=Conn, fold=cfg.GRAPH.FOLD, ni=cfg.GRAPH.NI, num_nodes=cfg.GRAPH.NUM_NODES,
                    start_id=cfg.GRAPH.START_ID, end_id=cfg.GRAPH.END_ID, base=cfg.GRAPH.BASE, exp=cfg.GRAPH.EXP, bottle_scale=cfg.GRAPH.BOTTLE_SCALE,
                    first_downsample=cfg.GRAPH.FIRST_DOWNSAMPLE, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfg.URL)
        local_to_pretrained = resnet_local_to_pretrained(cfg.GRAPH.NUM_NODES, cfg.GRAPH.FOLD,cfg.GRAPH.START_ID,cfg.GRAPH.END_ID)
        model.my_load_state_dict(state_dict, local_to_pretrained)
        for param in model.parameters(): # freeze all
            param.requires_grad = False
    return model