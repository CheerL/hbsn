from re import T
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from net.base_net import BaseNet, BaseNetConfig
from net.stn import STN
from net.unet import UNet


class HBSNetConfig(BaseNetConfig):
    stn_rate = 0.1
    grad_rate = 0.0

    stn_mode = 3
    # stn_mode: 0 - no stn,
    #           1 - pre stn
    #           2 - post stn
    #           3 - both stn

    radius = 50
    channels_down = [8, 8, 16, 32, 64, 128]
    channels_up = [8, 16, 32, 64, 128]
    is_skip = True

    def __init__(self, config_dict: dict = {}):
        super().__init__(config_dict)
        self._output_rate = 2 ** (
            len(self.channels_up) - len(self.channels_down)
        )
        self.output_height = int(self.height * self._output_rate)
        self.output_width = int(self.width * self._output_rate)


class HBSNet(BaseNet):
    def __init__(
        self,
        config: HBSNetConfig,
        backbone: UNet,
        pre_stn: Optional[STN] = None,
        post_stn: Optional[STN] = None,
    ):
        super().__init__(config)
        self.config = config
        self.pre_stn = pre_stn
        self.post_stn = post_stn
        self.backbone = backbone

        self.mask = self.create_mask(self.config.radius)
        self.to(self.config.device)

    def create_mask(self, r):
        x, y = torch.meshgrid(
            torch.arange(self.config.output_width),
            torch.arange(self.config.output_height),
            indexing="ij",
        )
        x = (x - self.config.output_height / 2) / r
        y = (y - self.config.output_width / 2) / r
        mask = (x**2 + y**2) <= 1
        mask.requires_grad = False
        mask = mask.to(self.config.device)
        return mask

    def forward(self, x):
        x = torch.sigmoid(20 * (x - 0.5))
        if self.pre_stn:
            x, pre_theta = self.pre_stn(x)
        x = self.backbone(x)
        if self.post_stn:
            x, post_theta = self.post_stn(x)
        x = torch.masked_fill(x, ~self.mask, 0.0)
        return x

    def loss(
        self, predict: Tensor, ground_truth: Tensor, is_mask=True
    ) -> Tuple[Dict[str, Tensor], Tuple[Tensor, Tensor]]:
        if self.post_stn:
            ground_truth, theta = self.post_stn(ground_truth)
            double_stn_predict, double_theta = self.post_stn(predict)
            stn_loss = F.mse_loss(
                double_stn_predict, predict, reduction="mean"
            )
        else:
            stn_loss = Tensor(0.0)

        output_data: Tuple[Tensor, Tensor] = (predict, ground_truth)
        predict_grad = torch.cat(
            torch.gradient(predict, dim=(2, 3)), dim=1
        )
        ground_truth_grad = torch.cat(
            torch.gradient(ground_truth, dim=(2, 3)), dim=1
        )

        if is_mask:
            predict = torch.masked_select(predict, self.mask)
            ground_truth = torch.masked_select(
                ground_truth, self.mask
            )
            predict_grad = torch.masked_select(
                predict_grad, self.mask
            )
            ground_truth_grad = torch.masked_select(
                ground_truth_grad, self.mask
            )

        hbs_loss = F.mse_loss(predict, ground_truth, reduction="mean")
        grad_loss = F.mse_loss(
            predict_grad, ground_truth_grad, reduction="mean"
        )

        loss = (
            hbs_loss
            + self.config.stn_rate * stn_loss
            + self.config.grad_rate * grad_loss
        )
        loss_dict: Dict[str, Tensor] = {
            "loss": loss,
            "hbs_loss": hbs_loss,
            "stn_loss": stn_loss,
            "grad_loss": grad_loss,
        }
        return loss_dict, output_data

    @classmethod
    def factory(cls, config: HBSNetConfig):
        backbone = UNet(
            n_channels=config.input_channels,
            n_classes=config.output_channels,
            channels_down=config.channels_down,
            channels_up=config.channels_up,
            is_bilinear=True,
            dtype=config.dtype,
            is_skip=config.is_skip,
        )

        pre_stn = (
            STN(
                input_channels=config.input_channels,
                height=config.height,
                width=config.width,
                dtype=config.dtype,
                stn_mode=1,
            )
            if config.stn_mode in [1, 3]
            else None
        )

        post_stn = (
            STN(
                input_channels=config.output_channels,
                height=config.output_height,
                width=config.output_width,
                dtype=config.dtype,
                stn_mode=2,
            )
            if config.stn_mode in [2, 3]
            else None
        )

        return cls(config, backbone, pre_stn, post_stn)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str):
        checkpoint, config, _, _, _ = BaseNet.load_model(
            checkpoint_path, device
        )
        hbsn = HBSNet.factory(config.net_config)
        hbsn.load_state_dict(checkpoint["state_dict"])
        return hbsn, config
