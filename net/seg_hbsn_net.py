from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from net.base import BaseNet
from net.hbsn import HBSNet

from config import SegHBSNNetConfig

class SegHBSNNet(BaseNet):
    def __init__(self, hbsn: HBSNet, config: SegHBSNNetConfig):
        super().__init__(config)
        self.config = config
        self.hbsn = hbsn
        self.build_model()
        self.to(self.config.device)
        # self.hbsn.to(self.config.device)

    def _handle_checkpoint(self, checkpoint):
        self.config.load_strict = False
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("hbsn"):
                del checkpoint["state_dict"][key]
        return super()._handle_checkpoint(checkpoint)

    def to(self, device: str):
        super().to(device)
        self.hbsn.to(device)

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
        binarized_mask = torch.sigmoid(self.config.mask_scale * (mask - 0.5))
        return binarized_mask

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
        f1, iou = self.get_metrics(
            self.binarize_mask(predict_mask), ground_truth
        )
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
            + self.hbs_loss_rate * hbs_loss_dict["loss"]
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
    def factory(cls, config: SegHBSNNetConfig):
        if config.hbsn_checkpoint:
            hbsn_checkpoint, hbsn_config, _, _, _ = BaseNet.load_model(
                config.hbsn_checkpoint, config.device
            )
            config.hbsn_config = hbsn_config.net_config
            hbsn = HBSNet.factory(config.hbsn_config)
            hbsn.load_state_dict(hbsn_checkpoint["state_dict"])
        else:
            hbsn = HBSNet.factory(config.hbsn_config)

        return cls(hbsn, config)
