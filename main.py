#! /home/extradisk/linchenran/.pyenv/versions/hbs_seg/bin/python
import torch
# from torch.optim.lr_scheduler import StepLR

from net import HBSNet
from summary_logger import HBSNLogger
from data.dataset import HBSNDataset
from loguru import logger
import click

DTYPE = torch.float32
IMAGE_INTERVAL = 5
LOG_PATH = 'logs/summary.log'

DEVICE = "cuda:0"
# lr_decay_step = 5000
# lr_decay_rate = 0.75
TOTAL_EPOCHES = 1000
LR = 1e-4
WEIGHT_NORM = 1e-5
MOMENTS = 0.8
BATCH_SIZE = 64
CHANNELS = [32, 64, 128, 256]


@logger.catch()
@click.command()
@click.option("--device", default=DEVICE, help="Device to run the training")
@click.option("--total_epoches", default=TOTAL_EPOCHES, help="Total epoches to train")
@click.option("--lr", default=LR, help="Learning rate")
@click.option("--weight_norm", default=WEIGHT_NORM, help="Weight decay")
@click.option("--moments", default=MOMENTS, help="Momentum")
@click.option("--batch_size", default=BATCH_SIZE, help="Batch size")
@click.option("--channels", default=CHANNELS, help="Channels in each layer")
def main(device, total_epoches, lr, weight_norm, moments, batch_size, channels):
    logger.add(LOG_PATH, rotation="10 MB", level="INFO")
    logger.info(f'''
    Start training in {device} with config:
        Total epoches: {total_epoches},
        Net Channels: {channels}
        Learning rate: {lr},
        Batch size: {batch_size},
        Weight decay: {weight_norm},
        Momentum: {moments}''')
    
    dataset = HBSNDataset()
    train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=batch_size)

    H, W, C_input, C_output = dataset.get_size()

    net = HBSNet(H, W, C_input, C_output, channels, device=device, dtype=DTYPE)
    net.initialize()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_norm, betas=(moments, 0.999))
    # scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)  # Learning rate decay
    
    log_writer = HBSNLogger(total_epoches, len(train_dataloader), len(test_dataloader), batch_size)
    log_writer.init_summary(net)

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
            # scheduler.step()

            log_writer.add_loss(epoch, iteration, loss)
            if iteration % IMAGE_INTERVAL == 0:
                log_writer.add_output(img, hbs, output)
        
        # Testing
        net.eval()
        with torch.no_grad():
            for iteration, (img, hbs) in enumerate(test_dataloader):
                img = img.to(device, dtype=DTYPE)
                hbs = hbs.to(device, dtype=DTYPE)
                output = net(img)
                loss = net.loss(output, hbs)
                
                log_writer.add_loss(epoch, iteration, loss, is_train=False)
                if iteration % IMAGE_INTERVAL == 0:
                    log_writer.add_output(img, hbs, output, is_train=False)

if __name__ == "__main__":
    main()
