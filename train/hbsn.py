import torch

from config import HBSNetConfig
from data.hbsn_dataset import HBSNDataset
from net.hbsn import HBSNet
from recoder import HBSNRecoder
from train.base_train import training

DATA_DIR = "img/generated"
TEST_DATA_DIR = "img/gen2"
DTYPE = torch.float32

VERSION = "0.10"
# VERSION 0.1
# initial version

# VERSION 0.2
# add learning rate decay

# VERSION 0.3
# add STN

# VERSION 0.3.1
# add STN control and rotation only mode

# VERSION 0.4
# add data augmentation

# VERSION 0.5
# save and load checkpoint, report epoch loss

# VERSION 0.6
# add aug params

# VERSION 0.7
# focus on unit disk

# VERSION 0.8
# add both stn

# VERSION 0.9
# add stn loss to increase the stn effect
# save config in checkpoint and allow to autoload net

# VERSION 0.10
# save optimizer into checkpoint
# allow different dataset in train and test

def main(
    data_dir=DATA_DIR,
    test_data_dir=TEST_DATA_DIR,
    device='cuda:0',
    total_epoches=1000,
    version=VERSION, 
    load='',
    log_dir='',
    log_base_dir='runs/hbsn',
    lr=1e-3,
    weight_norm=1e-5,
    moments=0.9,
    batch_size=32,
    channels=[8,16,32,64,128,256],
    stn_rate=0.0,
    lr_decay_rate=0.5,
    lr_decay_steps=[50,100],
    stn_mode=0,
    is_augment=False,
    comment='',
    augment_rotation=180.0,
    augment_scale=[0.8,1.2],
    augment_translate=[0.1,0.1],
    radius=50
    ):
    args = locals()
    config = HBSNetConfig(args)
    
    dataset = HBSNDataset(config=config)
    if not config.test_data_dir:
        train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=config.batch_size)
    else:
        train_dataloader, _ = dataset.get_dataloader(batch_size=config.batch_size, split_rate=1)
        test_dataset = HBSNDataset(config.test_data_dir, is_augment=False)
        _, test_dataloader = test_dataset.get_dataloader(batch_size=config.batch_size, split_rate=0)
    
    recoder = HBSNRecoder(config, len(train_dataloader), len(test_dataloader))
    net = HBSNet(
        height=dataset.height, width=dataset.width, input_channels=dataset.input_channels, config=config
        )
    training(
        net, recoder, config,
        train_dataloader, test_dataloader
    )

    
    