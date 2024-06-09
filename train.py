import os
import random
from typing import Tuple

import numpy
import torch
import torch.utils.data
from loguru import logger

from factory import (
    Config,
    config_factory,
    dataset_factory,
    net_factory,
    recorder_factory,
)
from net.base_net import BaseNet
from recorder import BaseRecorder

RANDOM_SEED = 960717
IMAGE_INTERVAL = 100
CHECKPOINT_INTERVAL = 5


def run(net: BaseNet, input_data: Tuple[torch.Tensor, torch.Tensor]):
    img, ground_truth = input_data
    img = img.to(net.config.device, dtype=net.config.dtype)
    ground_truth = ground_truth.to(
        net.config.device, dtype=net.config.dtype
    )

    predict = net(img)
    loss_dict, output_data = net.loss(predict, ground_truth)
    return loss_dict, output_data


def epoch_run(
    net: BaseNet,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    recorder: BaseRecorder,
    epoch: int,
    is_train: bool = True,
):
    net.train() if is_train else net.eval()
    with torch.no_grad() if not is_train else torch.enable_grad():
        for iteration, input_data in enumerate(dataloader):
            loss_dict, predict = run(net, input_data)
            recorder.add_loss(epoch, iteration, loss_dict, is_train)
            if is_train:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                optimizer.step()

            if iteration % IMAGE_INTERVAL == 0:
                recorder.add_output(
                    epoch, iteration, input_data, predict, is_train
                )
        recorder.add_epoch_loss(epoch, is_train)


def save_checkpoint(
    net: BaseNet,
    recorder: BaseRecorder,
    config: Config,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):
    # Save the best model or
    # Save model after `CHECKPOINT_INTERVAL` epochs

    def _save_checkpoint(_is_best):
        checkpoint_path = os.path.join(
            recorder.checkpoint_dir,
            "best.pth" if _is_best else f"epoch_{epoch}.pth",
        )
        net.save(
            checkpoint_path,
            epoch,
            recorder.best_epoch,
            recorder.best_loss,
            config,
            optimizer,
        )
        logger.warning(
            f"Model saved at epoch {epoch} to {checkpoint_path}"
        )

    is_best = recorder.update_best(epoch)
    if is_best:
        _save_checkpoint(True)

    if epoch % CHECKPOINT_INTERVAL == 0:
        _save_checkpoint(False)


def initialization(
    net: BaseNet, recorder: BaseRecorder, config: Config
):
    net.initialize()
    recorder.init_recorder(config, net)

    param_dict = net.get_param_dict(config.run_config.lr)
    optimizer = torch.optim.Adam(
        param_dict,
        lr=config.run_config.lr,
        weight_decay=config.run_config.weight_norm,
        betas=(config.run_config.moments, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.run_config.lr_decay_steps,
        gamma=config.run_config.lr_decay_rate,
    )
    return optimizer, scheduler


def load_checkpoint(
    net: BaseNet,
    recorder: BaseRecorder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
):
    if config.run_config.checkpoint_path and os.path.exists(
        config.run_config.checkpoint_path
    ):
        init_epoch, best_epoch, best_loss, optimizer_data = net.load(
            config.run_config.checkpoint_path
        )
        if optimizer_data:
            # optimizer_data["param_groups"] = param_dict
            # if 'param_groups' in optimizer_data:
            #     del optimizer_data['param_groups']
            optimizer.load_state_dict(optimizer_data)

        logger.info(
            f"Model loaded from {config.run_config.checkpoint_path}"
        )

        recorder.best_epoch = best_epoch
        recorder.best_loss = best_loss
        scheduler.last_epoch = init_epoch  # type: ignore
    else:
        init_epoch = -1
    return init_epoch


def train(type_: str, **config_dict):
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    numpy.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Load config then get net, dataloader and recorder
    config = config_factory(type_, config_dict)
    net = net_factory(type_, config)
    train_dataloader, test_dataloader = dataset_factory(type_, config)
    recorder = recorder_factory(
        type_, config, len(train_dataloader), len(test_dataloader)
    )

    optimizer, scheduler = initialization(net, recorder, config)
    init_epoch = load_checkpoint(
        net, recorder, optimizer, scheduler, config
    )

    for epoch in range(
        init_epoch + 1, config.run_config.total_epoches
    ):
        # Training
        epoch_run(
            net,
            train_dataloader,
            optimizer,
            recorder,
            epoch,
            is_train=True,
        )
        # Testing
        epoch_run(
            net,
            test_dataloader,
            optimizer,
            recorder,
            epoch,
            is_train=False,
        )
        scheduler.step()
        save_checkpoint(net, recorder, config, optimizer, epoch)
