"""训练入口：hydra 组合配置 + 自定义训练循环。

用法（仓库根目录）：
    python -m hbsn.train model=hbsn run.total_epoches=100 net.stn_mode=3
    python -m hbsn.train model=tpsn dataset.cat_ids='[16]' net.hbsn_checkpoint=runs/.../best.pth
"""
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from hbsn.config import RecorderSchema, RunSchema, validate_config
from hbsn.nets.base import BaseNet, torch_dtype
from hbsn.recorder import Recorder
from hbsn.registry import get_spec, resolve_model_name

RANDOM_SEED = 960717
IMAGE_INTERVAL = 20
CHECKPOINT_INTERVAL = 5


def run(net: BaseNet, input_data: Tuple[torch.Tensor, torch.Tensor]):
    img, ground_truth = input_data
    img = img.to(net.config.device, dtype=torch_dtype(net.config))
    ground_truth = ground_truth.to(net.config.device, dtype=torch_dtype(net.config))

    predict = net(img)
    loss_dict, output_data = net.loss(predict, ground_truth)
    return loss_dict, output_data


def epoch_run(net, dataloader, optimizer, recorder, epoch, is_train=True):
    net.train() if is_train else net.eval()
    with torch.no_grad() if not is_train else torch.enable_grad():
        for iteration, input_data in enumerate(dataloader):
            loss_dict, output_data = run(net, input_data)
            recorder.add_loss(epoch, iteration, loss_dict, is_train)
            if is_train:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                optimizer.step()

            if iteration % IMAGE_INTERVAL == 0:
                recorder.add_output(epoch, iteration, input_data, output_data, is_train)
    recorder.add_epoch_loss(epoch, is_train)


def initialization(net, recorder, run_cfg):
    """初始化网络与 recorder，返回 Adam 优化器 + MultiStepLR 调度器。"""
    net.initialize()
    recorder.init_recorder(config_to_dict(run_cfg), net)

    param_dict = net.get_param_dict(run_cfg.lr)
    optimizer = torch.optim.Adam(
        param_dict,
        lr=run_cfg.lr,
        weight_decay=run_cfg.weight_norm,
        betas=(run_cfg.moments, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=run_cfg.lr_decay_steps,
        gamma=run_cfg.lr_decay_rate,
    )
    return optimizer, scheduler


def save_checkpoint(net, recorder, run_cfg, config_dict, optimizer, epoch):
    """验证 loss 创新低存 best.pth，每 CHECKPOINT_INTERVAL 轮存 epoch_N.pth。"""

    def _save(_is_best):
        checkpoint_path = os.path.join(
            recorder.checkpoint_dir, "best.pth" if _is_best else f"epoch_{epoch}.pth"
        )
        net.save(
            checkpoint_path,
            epoch,
            recorder.best_epoch,
            recorder.best_loss,
            config_dict,
            optimizer,
        )
        logger.warning(f"Model saved at epoch {epoch} to {checkpoint_path}")

    is_best = recorder.update_best(epoch)
    if is_best:
        _save(True)
    if epoch % CHECKPOINT_INTERVAL == 0:
        _save(False)


def load_checkpoint(net, recorder, optimizer, scheduler, run_cfg):
    if run_cfg.checkpoint_path and os.path.exists(run_cfg.checkpoint_path):
        init_epoch, best_epoch, best_loss, optimizer_data = net.load(
            run_cfg.checkpoint_path
        )
        if optimizer_data:
            optimizer.load_state_dict(optimizer_data)
        logger.info(f"Model loaded from {run_cfg.checkpoint_path}")
        recorder.best_epoch = best_epoch
        recorder.best_loss = best_loss
        scheduler.last_epoch = init_epoch  # type: ignore
    else:
        init_epoch = -1
    return init_epoch


def config_to_dict(cfg) -> dict:
    """OmegaConf → 纯 dict（字符串化，供 recorder 展示与 checkpoint 存档）。"""
    return OmegaConf.to_container(cfg, resolve=True)


@hydra_main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    model_name = resolve_model_name(cfg.model)
    spec = get_spec(model_name)
    net_cfg, dataset_cfg = validate_config(cfg, spec.net_schema, spec.dataset_schema)
    run_cfg = OmegaConf.merge(OmegaConf.structured(RunSchema), cfg.run)
    recorder_cfg = OmegaConf.merge(OmegaConf.structured(RecorderSchema), cfg.recorder)

    net = spec.net.factory(net_cfg)

    dataset = spec.dataset(dataset_cfg)
    if dataset_cfg.test_data_dir:
        test_dataset = spec.dataset(dataset_cfg, is_test=True)
        train_dataloader, _ = dataset.get_dataloader(
            batch_size=run_cfg.batch_size, split_rate=1
        )
        _, test_dataloader = test_dataset.get_dataloader(
            batch_size=run_cfg.batch_size, split_rate=0
        )
    else:
        train_dataloader, test_dataloader = dataset.get_dataloader(
            batch_size=run_cfg.batch_size
        )

    assert train_dataloader is not None, "Train dataloader is None"
    assert test_dataloader is not None, "Test dataloader is None"

    recorder_cfg.log_base_dir = f"runs/{model_name}"
    recorder = Recorder(
        recorder_cfg,
        len(train_dataloader),
        len(test_dataloader),
        run_cfg.total_epoches,
        run_cfg.batch_size,
    )

    optimizer, scheduler = initialization(net, recorder, run_cfg)
    init_epoch = load_checkpoint(net, recorder, optimizer, scheduler, run_cfg)

    # checkpoint 里存档的配置：四类配置纯 dict（OmegaConf 可序列化，无类路径问题）
    checkpoint_config = {
        "net": config_to_dict(net_cfg),
        "dataset": config_to_dict(dataset_cfg),
        "recorder": config_to_dict(recorder_cfg),
        "run": config_to_dict(run_cfg),
    }

    for epoch in range(init_epoch + 1, run_cfg.total_epoches):
        epoch_run(net, train_dataloader, optimizer, recorder, epoch, is_train=True)
        epoch_run(net, test_dataloader, optimizer, recorder, epoch, is_train=False)
        scheduler.step()
        save_checkpoint(net, recorder, run_cfg, checkpoint_config, optimizer, epoch)


if __name__ == "__main__":
    main()
