"""SegHBSNNet：分割骨架 + 内嵌 HBSNet（组合架构，论文核心卖点）。

state_dict 键：hbsn.*（子网）+ 各子类 model.* / mask_conv.* / weight_layer.*。
"""
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from hbsn.nets.base import BaseNet
from hbsn.nets.hbsn import HBSNet


def net_config_from_checkpoint(checkpoint) -> dict:
    """兼容新旧两种 checkpoint 的 config 字段：新格式为 OmegaConf dict，
    旧格式为 pickle 的 Config 对象（含 net_config 属性）。"""
    cfg = checkpoint.get("config")
    if isinstance(cfg, dict):
        return cfg["net"] if "net" in cfg else cfg
    return cfg.net_config


class SegHBSNNet(BaseNet):
    def __init__(self, hbsn: HBSNet, config):
        super().__init__(config)
        self.hbsn = hbsn
        self.build_model()
        self.to(config.device)

    def build_model(self):
        raise NotImplementedError("build_model not implemented")

    def model_forward(self, img: Tensor) -> Tensor | List[Tensor]:
        raise NotImplementedError("model_forward not implemented")

    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, List[Tensor]]:
        results = self.model_forward(img)
        if isinstance(results, Tensor):
            predict_mask = results
            hbs = self.hbsn(self.binarize_mask(predict_mask))
            return predict_mask, hbs
        else:
            predict_mask = results[0]
            hbs = self.hbsn(self.binarize_mask(predict_mask))
            others = results[1:]
            return predict_mask, hbs, others

    def binarize_mask(self, mask: Tensor):
        return torch.sigmoid(self.config.mask_scale * (mask - 0.5))

    def get_hard_mask(self, mask: Tensor):
        return torch.relu(torch.sign(mask - 0.5))

    def get_metrics(self, predict: Tensor, ground_truth: Tensor):
        tp = (predict * ground_truth).sum(dim=(1, 2, 3))
        fp = (predict * (1 - ground_truth)).sum(dim=(1, 2, 3))
        fn = ((1 - predict) * ground_truth).sum(dim=(1, 2, 3))

        f1 = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        iou = tp / (tp + fp + fn)
        return f1, iou

    def loss(
        self, predict: Tuple[Tensor, Tensor], ground_truth: Tensor
    ) -> Tuple[Dict[str, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        predict_mask, predict_hbs = predict
        mse_loss = F.mse_loss(predict_mask, ground_truth)
        f1, iou = self.get_metrics(self.binarize_mask(predict_mask), ground_truth)
        f1 = f1.mean()
        iou = iou.mean()
        dice_loss = 1 - f1
        iou_loss = 1 - iou

        ground_truth_hbs = self.hbsn(ground_truth)
        hbs_loss_dict, (_, ground_truth_hbs) = self.hbsn.loss(
            predict_hbs, ground_truth_hbs
        )

        loss = (
            mse_loss
            + self.config.dice_rate * dice_loss
            + self.config.iou_rate * iou_loss
            + self.config.hbs_loss_rate * hbs_loss_dict["loss"]
        )

        loss_dict: Dict[str, Tensor] = {
            "loss": loss,
            "mse_loss": mse_loss,
            "dice": f1,
            "iou": iou,
            "hbs_loss": hbs_loss_dict["hbs_loss"],
        }
        output_data: Tuple[Tensor, Tensor, Tensor] = (
            predict_mask,
            predict_hbs,
            ground_truth_hbs,
        )
        return loss_dict, output_data

    @property
    def fixable_layers(self):
        return self.hbsn

    @property
    def uninitializable_layers(self):
        return self.hbsn

    @classmethod
    def factory(cls, config):
        if config.hbsn_checkpoint:
            hbsn_checkpoint, hbsn_config, _, _, _ = BaseNet.load_model(
                config.hbsn_checkpoint, config.device
            )
            hbsn = HBSNet.factory(net_config_from_checkpoint(hbsn_checkpoint))
            hbsn.load_state_dict(hbsn_checkpoint["state_dict"])
        else:
            hbsn = HBSNet.factory(config.hbsn_config)
        return cls(hbsn, config)
