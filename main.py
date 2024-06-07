import os
import random

import fire
import numpy
import torch
from loguru import logger

from net.base_net import BaseNet
from recoder import BaseRecoder
from factory import (
    Config,
    config_factory,
    dataset_factory,
    net_factory,
    recoder_factory,
)

RANDOM_SEED = 960717
IMAGE_INTERVAL = 100
CHECKPOINT_INTERVAL = 5

def run(net, input_data):
    img, gt = input_data
    img = img.to(net.config.device, dtype=net.config.dtype)
    gt = gt.to(net.config.device, dtype=net.config.dtype)
    output_data = net(img)
    loss_dict, output_data = net.loss(output_data, gt)
    return output_data, loss_dict

def epoch_run(net, dataloader, optimizer, recoder, epoch, is_train=True):
    net.train() if is_train else net.eval()
    with (torch.no_grad() if not is_train else torch.enable_grad()):
        for iteration, input_data in enumerate(dataloader):
            output_data, loss_dict = run(net, input_data)
            recoder.add_loss(epoch, iteration, loss_dict, is_train)
            if is_train:
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                optimizer.step()
    
            if iteration % IMAGE_INTERVAL == 0:
                recoder.add_output(epoch, iteration, input_data, output_data, is_train)
        recoder.add_epoch_loss(epoch, is_train)

def save_checkpoint(net: BaseNet, recoder: BaseRecoder, config: Config, optimizer: torch.optim.Optimizer, epoch: int):
    # Save the best model or
    # Save model after `CHECKPOINT_INTERVAL` epochs
    is_best = recoder.update_best(epoch)
    if is_best or epoch % CHECKPOINT_INTERVAL == 0:
        para_path = os.path.join(recoder.config.checkpoint_dir, "best.pth" if is_best else f"epoch_{epoch}.pth")
        net.save(para_path, epoch, recoder.best_epoch, recoder.best_loss, config, optimizer)
        logger.warning(f"{'Best model' if is_best else 'Model'} saved at epoch {epoch} to {para_path}")


def train(type_: str, **config_dict):
    torch.manual_seed(RANDOM_SEED)
    numpy.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    assert type_ in ['hbsn', 'maskrcnn', 'deeplab', 'unetpp'], "Invalid type"

    config = config_factory(type_, config_dict)
    net = net_factory(type_, config)
    train_dataloader, test_dataloader = dataset_factory(type_, config)
    recoder = recoder_factory(type_, config, len(train_dataloader), len(test_dataloader))

    net.initialize()
    recoder.init_recoder(config, net)
    
    param_dict = net.get_param_dict(config.run_config.lr)
    optimizer = torch.optim.Adam(
        param_dict, lr=config.run_config.lr, 
        weight_decay=config.run_config.weight_norm, 
        betas=(config.run_config.moments, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.run_config.lr_decay_steps, 
        gamma=config.run_config.lr_decay_rate
    )

    if config.run_config.load and os.path.exists(config.run_config.load):
        init_epoch, best_epoch, best_loss, optimizer_data = net.load(config.run_config.load)
        if optimizer_data:
            optimizer_data['param_groups'] = param_dict
            # if 'param_groups' in optimizer_data:
            #     del optimizer_data['param_groups']
            optimizer.load_state_dict(optimizer_data)
        
        logger.info(f"Model loaded from {config.run_config.load}")

        recoder.best_epoch = best_epoch
        recoder.best_loss = best_loss
        scheduler.last_epoch = init_epoch # type: ignore
    else:
        init_epoch = -1
    
    for epoch in range(init_epoch+1, config.run_config.total_epoches):
        # Training
        epoch_run(net, train_dataloader, optimizer, recoder, epoch, is_train=True)
        # Testing
        epoch_run(net, test_dataloader, optimizer, recoder, epoch, is_train=False)
        scheduler.step()
        save_checkpoint(net, recoder, config, optimizer, epoch)

if __name__ == '__main__':
    fire.Fire(train)



