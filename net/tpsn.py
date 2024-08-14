from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config import TPSNConfig
from net.hbsn import HBSNet
from net.seg_hbsn_net import SegHBSNNet

# from net.unet import UNet
from net.unet2d import UNet2D as UNet

IMSIZE = [256, 256]
PAD_SIZE = 16  # set it to be [0,0] if no padding is needed.


def pad_image(img):
    """
    INPUT:      img-        B,C,H,W
    OUTPUT:     padded_img- B,C,H+2p,W+2p
    """
    return F.pad(img, [PAD_SIZE] * 4, mode="constant", value=0)


def unpad_image(pad_img):
    """
    INPUT:      padded_img- B,C,H+2p,W+2p
    OUTPUT:     img-        B,C,H,W
    """
    return pad_img[:, :, PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE]


class BCLossFunc(torch.nn.Module):
    def __init__(self, size):
        super(BCLossFunc, self).__init__()
        self.hx1 = torch.tensor(2.0 / size[0])
        self.hx2 = torch.tensor(2.0 / size[1])

    def forward(self, mapping):
        eps = 1e-8
        u_SE, v_SE = mapping[:, 0:1, 1:, 1:], mapping[:, 1:2, 1:, 1:]
        u_NE, v_NE = mapping[:, 0:1, 0:-1, 1:], mapping[:, 1:2, 0:-1, 1:]
        u_SW, v_SW = mapping[:, 0:1, 1:, 0:-1], mapping[:, 1:2, 1:, 0:-1]
        u_NW, v_NW = mapping[:, 0:1, 0:-1, 0:-1], mapping[:, 1:2, 0:-1, 0:-1]
        # print(mapping.shape)
        # print(u_SE.shape, v_SE.shape)
        # print(u_NE.shape, v_NE.shape)
        # print(u_SW.shape, v_SW.shape)
        # print(u_NW.shape, v_NW.shape)
        # forward FDM
        u_x1f = (u_SE - u_SW) / self.hx1
        u_x2f = (u_SE - u_NE) / self.hx2
        v_x1f = (v_SE - v_SW) / self.hx1
        v_x2f = (v_SE - v_NE) / self.hx2
        numerator0 = (u_x1f**2 + v_x1f**2 - v_x2f**2 - u_x2f**2) ** 2 + (
            2 * u_x2f * u_x1f + 2 * v_x1f * v_x2f
        ) ** 2
        dedomenator0 = ((u_x1f + v_x2f) ** 2 + (v_x1f - u_x2f) ** 2) ** 2
        mu_square0 = numerator0 / (dedomenator0 + eps)
        # backward FDM
        u_x1b = (u_NE - u_NW) / self.hx1
        u_x2b = (u_SW - u_NW) / self.hx2
        v_x1b = (v_NE - v_NW) / self.hx1
        v_x2b = (v_SW - v_NW) / self.hx2
        numerator1 = (u_x1b**2 + v_x1b**2 - v_x2b**2 - u_x2b**2) ** 2 + (
            2 * u_x2b * u_x1b + 2 * v_x1b * v_x2b
        ) ** 2
        dedomenator1 = ((u_x1b + v_x2b) ** 2 + (v_x1b - u_x2b) ** 2) ** 2
        mu_square1 = numerator1 / (dedomenator1 + eps)

        return (torch.exp(mu_square0 + mu_square1) - 1).mean()


class LAPLossFunc(torch.nn.Module):
    def __init__(self, size):
        super(LAPLossFunc, self).__init__()
        kernel = (
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        self.hx1 = torch.tensor(2.0 / size[0])
        self.hx2 = torch.tensor(2.0 / size[1])

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0) / (self.hx1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0) / (self.hx2)
        return (torch.cat([x1, x2], dim=1) ** 2).mean()


class TPSN(SegHBSNNet):
    def __init__(self, hbsn: HBSNet, config: TPSNConfig):
        super().__init__(hbsn, config)
        self.config = config

    def build_model(self):
        self.model = UNet(3, 2, depth_down=4, depth_hidden=2, IsSeg=True)
        # self.model = UNet(
        #     3,
        #     2,
        #     [8, 16, 32, 64, 128],
        #     [8, 16, 32, 64, 128],
        #     is_skip=False,
        #     is_sigmoid=True,
        # )
        # The below 3 lines are for making identity grid
        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        grid_identity = F.affine_grid(
            theta,
            (
                1,
                2,
                self.config.height + 2 * PAD_SIZE,
                self.config.width + 2 * PAD_SIZE,
            ),
            align_corners=True
        )[0]
        self.grid_padded_identity = grid_identity.permute((2, 0, 1)).to(
            self.config.device
        )

        # The below 2 lines are for making trival template mask
        mask_temp = torch.ones(
            [
                1,
                1,
                self.config.height - 2 * PAD_SIZE,
                self.config.width - 2 * PAD_SIZE,
            ]
        )
        mask_temp = pad_image(mask_temp)
        self.mask_simple = pad_image(mask_temp).to(self.config.device)

        # define QC and LAP loss
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
        # self

    def model_forward(self, img: torch.Tensor) -> torch.Tensor:
        padded_img = pad_image(img)
        padded_mask = self.mask_simple.repeat(
            padded_img.shape[0], 1, 1, 1
        )  # N, 1, H, W
        predict_pad_vector = self.model(padded_img)
        # predict_pad_vector = torch.sigmoid(predict_pad_vector)
        predict_pad_vector = predict_pad_vector * 2 - 1

        predict_pad_mapping = (
            self.grid_padded_identity + predict_pad_vector
        )  # N, 2, H, W
        predict_pad_mask = F.grid_sample(
            padded_mask,
            predict_pad_mapping.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        predict_mask = unpad_image(predict_pad_mask)
        return predict_mask, predict_pad_mask, predict_pad_mapping

    # def forward(
    #     self, img: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Rewrite because model_forward() need to output 3 torch.tensor for loss computation
    #     """
    #     predict_pad_mask, predict_pad_mapping = self.model_forward(img)
    #     predict_mask = unpad_image(predict_pad_mask)
    #     hbs = self.hbsn(self.binarize_mask(predict_mask))
    #     return predict_mask, hbs, predict_pad_mapping

    def loss(
        self,
        predict: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ground_truth: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
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

        loss = (
            self.config.mse_rate * mse_loss
            + self.config.dice_rate * dice_loss
            + self.config.iou_rate * iou_loss
            + self.config.hbs_loss_rate * hbs_loss_dict["loss"]
            + self.config.qc_loss_rate * qc_loss
            + self.config.lap_loss_rate * lap_loss
        )

        if qc_loss == torch.inf:
            raise ValueError("QC loss is infinite")
        elif lap_loss == torch.inf:
            raise ValueError("LAP loss is infinite")
        elif qc_loss == torch.nan or lap_loss == torch.nan:
            raise ValueError("QC or LAP loss is nan")

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
        self.model.zero_initalization()

    @classmethod
    def factory(cls, config: TPSNConfig):
        return super().factory(config)
