from fastai.vision import *
from fastai.distributed import *
from fastai.callbacks import *

from wong.core import *
from wong.resnetx import *
from wong.resnetx2 import *
from wong.config import cfg, assert_cfg

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = untar_data(URLs.CIFAR_100)
print(path)

bs = 32
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=bs, 
                                  device=defaults.device).normalize(cifar_stats)

data.path = '.'

Stem = conv_bn
Unit = mbconv
ni = 64
bottle_scale = 1
tail_all = True
lr = 1e-2

nodes = 16
for fold in (13,):  #2,3,4,5,6,7,8,9,10,11
  num_nodes = [nodes*1,nodes*1+1,nodes*1+1,nodes*1+1]
  folds = [fold] * 4

  model = ResNetX2(Stem=conv_bn, Unit=mbconv, folds=folds, ni=ni, num_nodes=num_nodes, bottle_scale=bottle_scale, tail_all=tail_all, ks=3, c_out=data.c, zero_bn=True)
  params = num_params(model)
  filename = 'mbconv_folds_{}_nodes_{}_ni_{}_scale_{}_params_{}_bs_{}_lr_{}'.format('_'.join(map(str, folds)), 
                                                  '_'.join(map(str, num_nodes)), ni, bottle_scale, params, bs, lr)
  print(filename)
  learn = Learner(data, model, metrics=accuracy, callback_fns=[partial(callbacks.CSVLogger, filename=filename, append=True)]).to_fp16().to_distributed(args.local_rank)
  learn.model_dir = '.'
#   learn.lr_find()
#   print(learn.recorder.losses, learn.recorder.lrs)
#   learn.recorder.plot(suggestion=True)
#   learn.fit_one_cycle(1, lr)
  learn.fit_one_cycle(40, lr, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='model')]) #, tot_epochs=40, start_epoch=0
  learn.save(filename)



