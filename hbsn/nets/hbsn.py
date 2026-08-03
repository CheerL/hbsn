"""HBSNet：UNet backbone + 可选 pre/post STN + 圆盘 mask + hbs/grad/stn 损失。

state_dict 键：pre_stn.* / post_stn.* / backbone.*（mask 为非持久 buffer，不入键）。
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from hbsn.nets.base import BaseNet, output_size, torch_dtype
from hbsn.nets.stn import STN
from hbsn.nets.unet import UNet


class HBSNet(BaseNet):
    def __init__(
        self,
        config,
        backbone: UNet,
        pre_stn: STN | None = None,
        post_stn: STN | None = None,
    ):
        super().__init__(config)
        self.pre_stn = pre_stn
        self.post_stn = post_stn
        self.backbone = backbone

        # 非持久 buffer：.to(device) 自动搬运，且不进 state_dict（与旧版普通属性等价）
        self.register_buffer("mask", self.create_mask(config.radius), persistent=False)
        self.to(config.device)

    def create_mask(self, r):
        output_height, output_width = output_size(self.config)
        x, y = torch.meshgrid(
            torch.arange(output_width),
            torch.arange(output_height),
            indexing="ij",
        )
        x = (x - output_width / 2) / r
        y = (y - output_height / 2) / r
        mask = (x**2 + y**2) <= 1
        return mask.to(self.config.device)

    def forward(self, x):
        x = torch.sigmoid(20 * (x - 0.5))
        if self.pre_stn:
            x, _ = self.pre_stn(x)
        x = self.backbone(x)
        if self.post_stn:
            x, _ = self.post_stn(x)
        x = torch.masked_fill(x, ~self.mask, 0.0)
        return x

    def loss(
        self, predict: Tensor, ground_truth: Tensor, is_mask=True
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor]]:
        if self.post_stn:
            ground_truth, _ = self.post_stn(ground_truth)
            double_stn_predict, _ = self.post_stn(predict)
            stn_loss = F.mse_loss(double_stn_predict, predict, reduction="mean")
        else:
            stn_loss = predict.new_zeros(())

        output_data: tuple[Tensor, Tensor] = (predict, ground_truth)
        predict_grad = torch.cat(torch.gradient(predict, dim=(2, 3)), dim=1)
        ground_truth_grad = torch.cat(torch.gradient(ground_truth, dim=(2, 3)), dim=1)

        if is_mask:
            predict = torch.masked_select(predict, self.mask)
            ground_truth = torch.masked_select(ground_truth, self.mask)
            predict_grad = torch.masked_select(predict_grad, self.mask)
            ground_truth_grad = torch.masked_select(ground_truth_grad, self.mask)

        hbs_loss = F.mse_loss(predict, ground_truth, reduction="mean")
        grad_loss = F.mse_loss(predict_grad, ground_truth_grad, reduction="mean")

        loss = (
            hbs_loss
            + self.config.stn_rate * stn_loss
            + self.config.grad_rate * grad_loss
        )
        loss_dict: dict[str, Tensor] = {
            "loss": loss,
            "hbs_loss": hbs_loss,
            "stn_loss": stn_loss,
            "grad_loss": grad_loss,
        }
        return loss_dict, output_data

    @classmethod
    def factory(cls, config):
        output_height, output_width = output_size(config)
        backbone = UNet(
            n_channels=config.input_channels,
            n_classes=config.output_channels,
            channels_down=config.channels_down,
            channels_up=config.channels_up,
            is_bilinear=True,
            dtype=torch_dtype(config),
            is_skip=config.is_skip,
        )
        pre_stn = (
            STN(
                input_channels=config.input_channels,
                height=config.height,
                width=config.width,
                dtype=torch_dtype(config),
                stn_mode=1,
            )
            if config.stn_mode in [1, 3]
            else None
        )
        post_stn = (
            STN(
                input_channels=config.output_channels,
                height=output_height,
                width=output_width,
                dtype=torch_dtype(config),
                stn_mode=2,
            )
            if config.stn_mode in [2, 3]
            else None
        )
        return cls(config, backbone, pre_stn, post_stn)
