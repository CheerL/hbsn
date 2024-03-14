#! /home/extradisk/linchenran/.pyenv/versions/hbs_seg/bin/python
import os

import click
import torch
from torch.optim.lr_scheduler import MultiStepLR

from data.dataset import HBSNDataset
from net.hbsn import HBSNet
from summary_logger import HBSNSummary

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "img", "generated")

DTYPE = torch.float32
IMAGE_INTERVAL = 40

DEVICE = "cuda:0"
LR_DECAY_STEPS = [50,100]
LR_DECAY_RATE = 0.1
TOTAL_EPOCHES = 1000
LR = 1e-3
WEIGHT_NORM = 1e-5
MOMENTS = 0.9
BATCH_SIZE = 64
CHANNELS = [8, 16, 32, 64, 128, 256]
VERSION = "0.4"

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



@click.command()
@click.option("--data_dir", default=DATA_DIR, help="Data directory")
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
@click.option("--is_stn", default=0, help="Use STN or not")
@click.option("--is_augment", is_flag=True, help="Use data augmentation or not")
def main(data_dir, device, total_epoches, version,
         lr, weight_norm, moments, batch_size, channels, 
         lr_decay_rate, lr_decay_steps, is_stn, is_augment):
    if isinstance(channels, str):
        if channels == '[]':
            channels = []
        else:
            channels = list(map(int, channels.replace('[', '').replace(']', '').split(",")))
    elif isinstance(channels, int):
        channels = [channels]
    else:
        raise ValueError("Channels should be a list of integers")
        
    if isinstance(lr_decay_steps, str):
        if lr_decay_steps == '[]':
            lr_decay_steps = []
        else:
            lr_decay_steps = list(map(int, lr_decay_steps.replace('[', '').replace(']', '').split(",")))
    elif isinstance(lr_decay_steps, int):
        lr_decay_steps = [lr_decay_steps]
    else:
        raise ValueError("Learning rate decay steps should be a list of integers")
    
    config = {
        "data_dir": data_dir,
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
        "is_stn": is_stn,
        "is_augment": is_augment,
    }

    dataset = HBSNDataset(data_dir, is_augment=is_augment)
    H, W, C_input, C_output = dataset.get_size()
    train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=batch_size)

    net = HBSNet(
        height=H, width=W, input_channels=C_input, output_channels=C_output, 
        channels=channels, device=device, dtype=DTYPE, is_stn=is_stn
        )
    net.initialize()
    
    summary_writer = HBSNSummary(config, len(train_dataloader), len(test_dataloader))
    summary_writer.init_logger()
    summary_writer.init_summary(net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_norm, betas=(moments, 0.999))
    
    if lr_decay_steps and lr_decay_rate != 1:
        scheduler = MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=lr_decay_rate)
    else:
        scheduler = None


    ## Training
    for epoch in range(total_epoches):
        net.train()
        for iteration, (img, hbs) in enumerate(train_dataloader):
            img = img.to(device, dtype=DTYPE)
            hbs = hbs.to(device, dtype=DTYPE)
            output = net(img)
            optimizer.zero_grad()
            loss = net.loss(output, hbs)
            loss.backward()
            optimizer.step()

            summary_writer.add_loss(epoch, iteration, loss)
            if iteration % IMAGE_INTERVAL == 0:
                summary_writer.add_output(img, hbs, output)
        
        # Testing
        net.eval()
        with torch.no_grad():
            for iteration, (img, hbs) in enumerate(test_dataloader):
                img = img.to(device, dtype=DTYPE)
                hbs = hbs.to(device, dtype=DTYPE)
                output = net(img)
                loss = net.loss(output, hbs)
                
                summary_writer.add_loss(epoch, iteration, loss, is_train=False)
                if iteration % IMAGE_INTERVAL == 0:
                    summary_writer.add_output(img, hbs, output, is_train=False)
                    
        if scheduler:
            scheduler.step()

if __name__ == "__main__":
    main()
