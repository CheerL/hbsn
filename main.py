import random

import fire
import numpy
import torch

from run.deeplab import main as deeplab_train
from run.hbsn import main as hbsn_train
from run.hbsn import main_v2 as hbsn_v2_train
from run.maskrcnn import main as maskrcnn_train
from run.unetpp import main as unetpp_train

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
        'hbsn_v2': hbsn_v2_train
    })