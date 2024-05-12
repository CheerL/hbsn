import fire
import torch
import numpy
import random

from train.hbsn import main as hbsn_train
from train.maskrcnn import main as maskrcnn_train
from train.deeplab import main as deeplab_train
from train.unetpp import main as unetpp_train

RANDOM_SEED = 960717

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    numpy.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    fire.Fire({
        'unetpp': unetpp_train,
        'maskrcnn' : maskrcnn_train,
        'deeplab' : deeplab_train,
        'hbsn' : hbsn_train,
    })