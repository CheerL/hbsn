import os

import torch
from loguru import logger

from net.base_net import BaseNet
from recoder import BaseRecoder
from config import BaseConfig

DTYPE = torch.float32
IMAGE_INTERVAL = 20
CHECKPOINT_INTERVAL = 5

def single_run(net, input_data):
    img, gt = input_data
    img = img.to(net.device, dtype=net.dtype)
    gt = gt.to(net.device, dtype=net.dtype)
    output_data = net(img)
    loss_dict, output_data = net.loss(output_data, gt)
    return output_data, loss_dict

def epoch_run(net, dataloader, optimizer, recoder, epoch, is_train=True):
    net.train() if is_train else net.eval()
    with (torch.no_grad() if not is_train else torch.enable_grad()):
        for iteration, input_data in enumerate(dataloader):
            output_data, loss_dict = single_run(net, input_data)
            recoder.add_loss(epoch, iteration, loss_dict, is_train)
            if is_train:
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                optimizer.step()
    
            if iteration % IMAGE_INTERVAL == 0:
                recoder.add_output(epoch, iteration, input_data, output_data, is_train)
        recoder.add_epoch_loss(epoch, is_train)

def save_checkpoint(net, recoder, optimizer, epoch, is_best=False):
    # Save the best model or
    # Save model after `CHECKPOINT_INTERVAL` epochs
    is_best = recoder.update_best(epoch)
    if is_best or epoch % CHECKPOINT_INTERVAL == 0:
        para_path = os.path.join(recoder.checkpoint_dir, f"best.pth" if is_best else f"epoch_{epoch}.pth")
        net.save(para_path, epoch, recoder.best_epoch, recoder.best_loss, recoder.config, optimizer)
        logger.warning(f"{'Best model' if is_best else 'Model'} saved at epoch {epoch} to {para_path}")

def training(
    net: BaseNet, recoder: BaseRecoder, config: BaseConfig,
    train_dataloader, test_dataloader
    ):
    net.initialize()
    recoder.init_recoder(net)
    
    param_dict = net.get_param_dict(config.lr, config.is_freeze, config.finetune_rate)
    optimizer = torch.optim.Adam(param_dict, lr=config.lr, weight_decay=config.weight_norm, betas=(config.moments, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_steps, gamma=config.lr_decay_rate)
    
    load_path = config.load
    if load_path and os.path.exists(load_path):
        init_epoch, best_epoch, best_loss, optimizer_data = net.load(load_path)
        if optimizer_data:
            optimizer.load_state_dict(optimizer_data)
        
        logger.info(f"Model loaded from {load_path}")

        recoder.best_epoch = best_epoch
        recoder.best_loss = best_loss
        scheduler.last_epoch = init_epoch
    else:
        init_epoch = -1
    
    for epoch in range(init_epoch+1, config.total_epoches):
        # Training
        epoch_run(net, train_dataloader, optimizer, recoder, epoch, is_train=True)
        # Testing
        epoch_run(net, test_dataloader, optimizer, recoder, epoch, is_train=False)
        scheduler.step()
        save_checkpoint(net, recoder, optimizer, epoch)

