"""TPSN：UNet2D 位移场 + grid_sample 形变 mask，QC/LAP 损失。

state_dict 键：model.Net.* / lap_loss.weight / hbsn.*。
"""

import torch
import torch.nn.functional as F

from hbsn.nets.segmentation import SegHBSNNet
from hbsn.nets.unet import UNet2D as UNet

PAD_SIZE = 16  # 置 [0,0] 即无 padding


def pad_image(img):
    return F.pad(img, [PAD_SIZE] * 4, mode="constant", value=0)


def unpad_image(pad_img):
    return pad_img[:, :, PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE]


class BCLossFunc(torch.nn.Module):
    """QC 损失：有限差分格式的 Beltrami 系数。"""

    def __init__(self, size):
        super().__init__()
        # 非持久 buffer：.to(device) 可搬运，不入 state_dict
        self.register_buffer(
            "hx1", torch.tensor(2.0 / size[0]), persistent=False
        )
        self.register_buffer(
            "hx2", torch.tensor(2.0 / size[1]), persistent=False
        )

    def forward(self, mapping):
        eps = 1e-8
        u_SE, v_SE = mapping[:, 0:1, 1:, 1:], mapping[:, 1:2, 1:, 1:]
        u_NE, v_NE = mapping[:, 0:1, 0:-1, 1:], mapping[:, 1:2, 0:-1, 1:]
        u_SW, v_SW = mapping[:, 0:1, 1:, 0:-1], mapping[:, 1:2, 1:, 0:-1]
        u_NW, v_NW = mapping[:, 0:1, 0:-1, 0:-1], mapping[:, 1:2, 0:-1, 0:-1]
        # forward FDM
        u_x1f = (u_SE - u_SW) / self.hx1
        u_x2f = (u_SE - u_NE) / self.hx2
        v_x1f = (v_SE - v_SW) / self.hx1
        v_x2f = (v_SE - v_NE) / self.hx2
        numerator0 = (u_x1f**2 + v_x1f**2 - v_x2f**2 - u_x2f**2) ** 2 + (
            2 * u_x2f * u_x1f + 2 * v_x1f * v_x2f
        ) ** 2
        denominator0 = ((u_x1f + v_x2f) ** 2 + (v_x1f - u_x2f) ** 2) ** 2
        mu_square0 = numerator0 / (denominator0 + eps)
        # backward FDM
        u_x1b = (u_NE - u_NW) / self.hx1
        u_x2b = (u_SW - u_NW) / self.hx2
        v_x1b = (v_NE - v_NW) / self.hx1
        v_x2b = (v_SW - v_NW) / self.hx2
        numerator1 = (u_x1b**2 + v_x1b**2 - v_x2b**2 - u_x2b**2) ** 2 + (
            2 * u_x2b * u_x1b + 2 * v_x1b * v_x2b
        ) ** 2
        denominator1 = ((u_x1b + v_x2b) ** 2 + (v_x1b - u_x2b) ** 2) ** 2
        mu_square1 = numerator1 / (denominator1 + eps)

        return (torch.exp(mu_square0 + mu_square1) - 1).mean()


class LAPLossFunc(torch.nn.Module):
    """LAP 损失：Laplace 算子 FDM。weight 是 Parameter（进 state_dict，键 lap_loss.weight）。"""

    def __init__(self, size):
        super().__init__()
        kernel = (
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        self.register_buffer(
            "hx1", torch.tensor(2.0 / size[0]), persistent=False
        )
        self.register_buffer(
            "hx2", torch.tensor(2.0 / size[1]), persistent=False
        )

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0) / self.hx1
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0) / self.hx2
        return (torch.cat([x1, x2], dim=1) ** 2).mean()


class TPSN(SegHBSNNet):
    def build_model(self):
        self.model = UNet(3, 2, depth_down=4, depth_hidden=2, is_seg=True)

        # identity grid（对齐 padding 后尺寸）
        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        grid_identity = F.affine_grid(
            theta,
            (
                1,
                2,
                self.config.height + 2 * PAD_SIZE,
                self.config.width + 2 * PAD_SIZE,
            ),
            align_corners=True,
        )[0]
        self.grid_padded_identity = grid_identity.permute((2, 0, 1)).to(
            self.config.device
        )

        # 平凡模板 mask（全 1）
        mask_temp = torch.ones(
            [
                1,
                1,
                self.config.height - 2 * PAD_SIZE,
                self.config.width - 2 * PAD_SIZE,
            ]
        )
        self.mask_simple = pad_image(mask_temp).to(self.config.device)

        # QC / LAP 损失
        self.qc_loss = BCLossFunc(
            [
                self.config.height + 2 * PAD_SIZE,
                self.config.width + 2 * PAD_SIZE,
            ]
        )
        self.lap_loss = LAPLossFunc(
            [
                self.config.height + 2 * PAD_SIZE,
                self.config.width + 2 * PAD_SIZE,
            ]
        )

    def model_forward(self, img: torch.Tensor) -> torch.Tensor:
        padded_img = pad_image(img)
        padded_mask = self.mask_simple.repeat(padded_img.shape[0], 1, 1, 1)
        predict_pad_vector = self.model(padded_img)
        predict_pad_vector = predict_pad_vector * 2 - 1

        predict_pad_mapping = self.grid_padded_identity + predict_pad_vector
        predict_pad_mask = F.grid_sample(
            padded_mask,
            predict_pad_mapping.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        predict_mask = unpad_image(predict_pad_mask)
        return predict_mask, predict_pad_mask, predict_pad_mapping

    def loss(
        self,
        predict: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ground_truth: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
        predict_mask, predict_hbs, (predict_pad_mask, predict_pad_mapping) = (
            predict
        )
        mse_loss = F.mse_loss(predict_pad_mask, pad_image(ground_truth))
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

        qc_loss = self.qc_loss(predict_pad_mapping)
        lap_loss = self.lap_loss(predict_pad_mapping)

        if qc_loss == torch.inf:
            raise ValueError("QC loss is infinite")
        elif lap_loss == torch.inf:
            raise ValueError("LAP loss is infinite")
        elif torch.isnan(qc_loss) or torch.isnan(lap_loss):
            raise ValueError("QC or LAP loss is nan")

        loss = (
            self.config.mse_rate * mse_loss
            + self.config.dice_rate * dice_loss
            + self.config.iou_rate * iou_loss
            + self.config.hbs_loss_rate * hbs_loss_dict["loss"]
            + self.config.qc_loss_rate * qc_loss
            + self.config.lap_loss_rate * lap_loss
        )

        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "dice": f1,
            "iou": iou,
            "hbs_loss": hbs_loss_dict["hbs_loss"],
            "qc_loss": qc_loss,
            "lap_loss": lap_loss,
        }
        return loss_dict, (
            predict_mask,
            predict_hbs,
            ground_truth_hbs,
            predict_pad_mapping,
        )

    def initialize(self):
        super().initialize()
        self.model.zero_initialization()
