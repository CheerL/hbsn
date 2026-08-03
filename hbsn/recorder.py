"""Recorder：SummaryWriter + loguru + checkpoint 目录 + matplotlib 结果图。

三份旧版 add_output（HBSNRecorder/CocoHBSNRecorder/TPSNRecorder）合并为一个方法，
按 output_data 结构分发：
- len==2 (predict_hbs, ground_truth_hbs)         → 3 行（hbsn 型）
- len==3 (predict_mask, predict_hbs, gt_hbs)     → 5 行（分割型）
- len==4 (..., predict_pad_mapping)              → 6 行 + 位移场网格（tpsn 型）
"""
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.collections import LineCollection
from torch.utils.tensorboard.writer import SummaryWriter

from hbsn.config.schemas import RecorderSchema
from hbsn.nets.base import torch_dtype


def get_random_index(num, size):
    num = min(num, size)
    index = np.random.choice(size, num, replace=False)
    return index, num


class Recorder(SummaryWriter):
    def __init__(
        self,
        config: RecorderSchema,
        train_size: int,
        test_size: int,
        total_epoches: int,
        batch_size: int,
    ):
        self.config = config
        self.set_log_dir()
        super().__init__(log_dir=self.log_dir)

        self.total_epoches = total_epoches
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size

        self.train_loss = torch.zeros(self.train_size)
        self.test_loss = torch.zeros(self.test_size)
        self.train_iou = torch.zeros(self.train_size)
        self.test_iou = torch.zeros(self.test_size)
        self.train_dice = torch.zeros(self.train_size)
        self.test_dice = torch.zeros(self.test_size)

        self.best_epoch = -1
        self.best_loss = 1e10

        logger.add(self.log_path, level="INFO")

    def set_log_dir(self):
        if self.config.log_dir:
            self.log_dir = self.config.log_dir
        else:
            current_time = datetime.now().astimezone().strftime("%b%d_%H-%M-%S")
            self.log_dir = os.path.join(self.config.log_base_dir, current_time)
            if self.config.comment:
                self.log_dir += f"_{self.config.comment}"

        self.log_path = os.path.join(self.log_dir, "log.log")
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def init_recorder(self, config_dict: dict[str, str], net=None):
        config_info = "\n\t".join(
            f"{k}: {v}" for k, v in config_dict.items()
        )
        logger.info(f"Start training with config:\n\t{config_info}")

        max_width = max([len(s) for s in config_dict]) * 0.1
        max_height = len(config_dict) * 0.25
        fig = plt.figure(figsize=(max_width, max_height), dpi=100)
        plt.text(
            0.5, 0.5, config_info, ha="center", va="center",
            multialignment="left", fontsize=12,
        )
        plt.axis("off")
        self.add_figure("config", fig)
        plt.close(fig)  # 防 matplotlib 全局 registry 累积

        if net and self.config.is_add_graph:
            try:
                empty_input = torch.rand(
                    net.get_input_shape(),
                    requires_grad=False,
                    device=net.config.device,
                    dtype=torch_dtype(net.config),
                )
                self.add_graph(net, empty_input)
            except Exception as e:  # noqa: BLE001 - add_graph 非关键路径，吞所有异常
                logger.error(f"Add graph error: {e}")

    def add_loss(self, epoch, iteration, loss_dict, is_train=True):
        assert "loss" in loss_dict, "loss_dict must contain the key 'loss'"
        prefix, size, total_iteration, logger_func = self._get_info(
            epoch, iteration, is_train=is_train
        )

        if is_train:
            self.train_loss[iteration] = loss_dict["loss"]
        else:
            self.test_loss[iteration] = loss_dict["loss"]

        if "iou" in loss_dict:
            if is_train:
                self.train_iou[iteration] = loss_dict["iou"]
            else:
                self.test_iou[iteration] = loss_dict["iou"]
        if "dice" in loss_dict:
            if is_train:
                self.train_dice[iteration] = loss_dict["dice"]
            else:
                self.test_dice[iteration] = loss_dict["dice"]

        for loss_name, loss_value in loss_dict.items():
            self.add_scalar(f"{loss_name}/{prefix}", loss_value, total_iteration)
        self.flush()

        loss_info = ", ".join(
            f"{name}={value:.4f}" for name, value in loss_dict.items()
        )
        logger_func(f"{prefix}: epoch={epoch}/{self.total_epoches}, iteration={iteration}/{size}, {loss_info}")

    def add_epoch_loss(self, epoch, is_train=True):
        prefix, _, _, logger_func = self._get_info(epoch, is_train=is_train)

        if is_train:
            loss = self.train_loss.mean().item()
            iou = self.train_iou.mean().item() if self.train_iou.numel() else 0.0
            dice = self.train_dice.mean().item() if self.train_dice.numel() else 0.0
        else:
            loss = self.test_loss.mean().item()
            iou = self.test_iou.mean().item() if self.test_iou.numel() else 0.0
            dice = self.test_dice.mean().item() if self.test_dice.numel() else 0.0

        self.add_scalar(f"loss/{prefix}_epoch", loss, epoch)
        if iou:
            self.add_scalar(f"iou/{prefix}_epoch", iou, epoch)
        if dice:
            self.add_scalar(f"dice/{prefix}_epoch", dice, epoch)
        self.flush()

        logger_func(f"{prefix} epoch total: epoch={epoch}/{self.total_epoches}, total loss={loss}")

    def add_output(self, epoch, iteration, input_data, output_data, is_train=True, num=10):
        prefix, _, total_iteration, _ = self._get_info(epoch, iteration, is_train)
        k, num = get_random_index(num, self.batch_size)

        img = input_data[0][k].detach().cpu().numpy()
        if img.ndim == 3:  # HBSN 型灰度图 (B,1,H,W)
            img_k = img[:, 0]
        else:  # COCO 型 RGB (B,3,H,W)
            img_k = img.transpose(0, 2, 3, 1)

        if len(output_data) == 2:  # hbsn
            predict_hbs, ground_truth_hbs = output_data
            n_rows, panels = 3, [("img", img_k), ("gt_hbs", ground_truth_hbs), ("pred_hbs", predict_hbs)]
        elif len(output_data) == 4:  # tpsn（含位移场网格）
            predict_mask, predict_hbs, ground_truth_hbs, predict_pad_mapping = output_data
            panels = [
                ("img", img_k), ("pred_mask", predict_mask), ("gt_mask", input_data[1]),
                ("pred_hbs", predict_hbs), ("gt_hbs", ground_truth_hbs),
                ("mapping", predict_pad_mapping),
            ]
            n_rows = 6
        else:  # seg
            predict_mask, predict_hbs, ground_truth_hbs = output_data
            panels = [
                ("img", img_k), ("pred_mask", predict_mask), ("gt_mask", input_data[1]),
                ("pred_hbs", predict_hbs), ("gt_hbs", ground_truth_hbs),
            ]
            n_rows = 5

        fig = plt.figure(figsize=(num * 2, n_rows * 2), dpi=100 if n_rows == 3 else 200)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for row, (name, data) in enumerate(panels):
            for i in range(num):
                plt.subplot(n_rows, num, row * num + i + 1)
                if name == "img":
                    plt.imshow(data[i], cmap="gray" if data[i].ndim == 2 else None)
                elif name == "mapping":
                    m = data[i, :, ::8, ::8].cpu().numpy()
                    plt.gca().add_collection(
                        LineCollection(m.transpose(1, 2, 0), color="r", linewidths=0.5)
                    )
                    plt.gca().add_collection(
                        LineCollection(m.transpose(2, 1, 0), color="r", linewidths=0.5)
                    )
                    plt.gca().axis("equal")
                else:
                    arr = data[i].detach().cpu().numpy()
                    if arr.ndim == 3:
                        plt.imshow(np.linalg.norm(arr, axis=0), cmap="jet")
                    else:
                        plt.imshow(arr[0], cmap="gray")
                plt.axis("off")
        self.add_figure(f"result/{prefix}", fig, total_iteration)
        plt.close(fig)  # 防 matplotlib 全局 registry 累积
        self.flush()

    def update_best(self, epoch):
        test_epoch_loss = self.test_loss.mean().item()
        if test_epoch_loss < self.best_loss:
            self.best_loss = test_epoch_loss
            self.best_epoch = epoch
            return True
        return False

    def _get_info(self, epoch, iteration=0, is_train=True):
        if is_train:
            prefix, size, logger_func = "Train", self.train_size, logger.info
        else:
            prefix, size, logger_func = "Test", self.test_size, logger.warning
        total_iteration = epoch * size + iteration
        return prefix, size, total_iteration, logger_func
