#! /home/extradisk/linchenran/.pyenv/versions/hbs_seg/bin/python
import os

import click
import torch

from data.hbsn_dataset import HBSNDataset
from net.hbsn import HBSNet
from recoder import HBSNRecoder
from train.base_train import list_para_handle, training

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "img", "generated")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "img", "gen2")

DTYPE = torch.float32
IMAGE_INTERVAL = 100
CHECKPOINT_INTERVAL = 5
LOG_BASE_DIR = 'runs/hbsn'

DEVICE = "cuda:0"
LR_DECAY_STEPS = [50,100]
LR_DECAY_RATE = 0.5
TOTAL_EPOCHES = 1000
LR = 1e-3
WEIGHT_NORM = 1e-5
MOMENTS = 0.9
BATCH_SIZE = 64
CHANNELS = [8, 16, 32, 64, 128, 256]
VERSION = "0.9"
AUGMENT_ROTATION = 180.0
AUGMENT_SCALE = [0.8, 1.2]
AUGMENT_TRANSLATE = [0.1, 0.1]
RADUIS = 50

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

@click.command()
@click.option("--data_dir", default=DATA_DIR, help="Data directory")
@click.option("--test_data_dir", default=TEST_DATA_DIR, help="Test data directory")
@click.option("--device", default=DEVICE, help="Device to run the training")
@click.option("--total_epoches", default=TOTAL_EPOCHES, help="Total epoches to train")
@click.option("--lr", default=LR, help="Learning rate")
@click.option("--weight_norm", default=WEIGHT_NORM, help="Weight decay")
@click.option("--moments", default=MOMENTS, help="Momentum")
@click.option("--batch_size", default=BATCH_SIZE, help="Batch size")
@click.option("--channels", default=str(CHANNELS), help="Channels in each layer", type=str)
@click.option("--version", default=VERSION, help="Version of the model")
@click.option("--lr_decay_rate", default=LR_DECAY_RATE, help="Learning rate decay rate")
@click.option("--lr_decay_steps", default=str(LR_DECAY_STEPS), help="Learning rate decay steps", type=str)
@click.option("--stn_mode", default=0, help="Use STN or not")
@click.option("--is_augment", is_flag=True, help="Use data augmentation or not")
@click.option("--load", default="", help="Load model from checkpoint")
@click.option("--comment", default='')
@click.option("--log_dir", default='', help="Log directory")
@click.option("--augment_rotation", default=AUGMENT_ROTATION, help="Rotation range for data augmentation")
@click.option("--augment_scale", default=str(AUGMENT_SCALE), help="Scale range for data augmentation")
@click.option("--augment_translate", default=str(AUGMENT_TRANSLATE), help="Translate range for data augmentation")
@click.option("--radius", default=RADUIS, help="Radius for mask")
@click.option("--stn_rate", default=0.0, help="STN loss rate")
@click.option("--log_base_dir", default=LOG_BASE_DIR)
def main(data_dir, test_data_dir, device, total_epoches, version, load, log_dir, log_base_dir,
         lr, weight_norm, moments, batch_size, channels, stn_rate,
         lr_decay_rate, lr_decay_steps, stn_mode, is_augment, comment,
         augment_rotation, augment_scale, augment_translate, radius):
    channels = list_para_handle(channels)
    lr_decay_steps = list_para_handle(lr_decay_steps)
    augment_scale = list_para_handle(augment_scale, float)
    augment_translate = list_para_handle(augment_translate, float)
    
    config = {
        "data_dir": data_dir,
        "test_dir": test_data_dir,
        "device": device,
        "total_epoches": total_epoches,
        "lr": lr,
        "weight_norm": weight_norm,
        "moments": moments,
        "batch_size": batch_size,
        "channels": channels,
        "version": version,
        "lr_decay_rate": lr_decay_rate,
        "lr_decay_steps": lr_decay_steps,
        "stn_mode": stn_mode,
        "is_augment": (augment_rotation, augment_scale, augment_translate) if is_augment else False,
        "radius": radius,
        'stn_rate': stn_rate
    }

    dataset = HBSNDataset(
        data_dir, is_augment=is_augment, 
        augment_rotation=augment_rotation, 
        augment_scale=augment_scale, 
        augment_translate=augment_translate
        )
    H, W, C_input, C_output = dataset.get_size()
    if not test_data_dir:
        train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=batch_size)
    else:
        train_dataloader, _ = dataset.get_dataloader(batch_size=batch_size, split_rate=1)
        test_dataset = HBSNDataset(
            test_data_dir, is_augment=False
            )
        _, test_dataloader = test_dataset.get_dataloader(batch_size=batch_size, split_rate=0)
    
    recoder = HBSNRecoder(
        config, len(train_dataloader), len(test_dataloader), 
        log_dir=log_dir, comment=comment, log_base_dir=log_base_dir
        )
    
    net = HBSNet(
        height=H, width=W, input_channels=C_input, output_channels=C_output, stn_rate=stn_rate,
        channels=channels, device=device, dtype=DTYPE, stn_mode=stn_mode, radius=radius
        )
    
    param_dict = net.get_param_dict(lr)
    optimizer = torch.optim.Adam(param_dict, lr=lr, weight_decay=weight_norm, betas=(moments, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=lr_decay_rate)
     
    training(
        net, recoder, optimizer, scheduler,
        train_dataloader, test_dataloader,
        load,
    )
