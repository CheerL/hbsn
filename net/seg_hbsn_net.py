from typing import Optional

import torch
from torch.nn import functional as F

from config import SegNetConfig
from net.base_net import BaseNet
from net.hbsn import HBSNet, HBSNet_V2

DTYPE = torch.float32

class SegHBSNNet(BaseNet):
    def __init__(
        self, height=256, width=256, input_channels=3, output_channels=1, 
        dice_rate=0.1, iou_rate=0, hbs_loss_rate=1, mask_scale=10,
        hbsn_checkpoint='', hbsn_version=1,
        hbsn_channels=[64, 128, 256, 512], hbsn_radius=50,
        hbsn_stn_mode=0, hbsn_stn_rate=0.0, 
        dtype=DTYPE, device="cpu", config: Optional[SegNetConfig]=None
        ):
        super().__init__(height, width, input_channels, output_channels, device, dtype, config)
        if config:
            self.dice_rate = config.dice_rate
            self.iou_rate = config.iou_rate
            self.hbs_loss_rate = config.hbs_loss_rate
            self.mask_scale = config.mask_scale
            self.hbsn_checkpoint = config.hbsn_checkpoint
            self.hbsn_radius = config.hbsn_radius
            self.hbsn_stn_mode = config.hbsn_stn_mode
            self.hbsn_stn_rate = config.hbsn_stn_rate
            self.hbsn_version = config.hbsn_version
        else:
            self.dice_rate = dice_rate
            self.iou_rate = iou_rate
            self.hbs_loss_rate = hbs_loss_rate
            self.mask_scale = mask_scale
            self.hbsn_checkpoint = hbsn_checkpoint
            self.hbsn_channels = hbsn_channels
            self.hbsn_radius = hbsn_radius
            self.hbsn_stn_mode = hbsn_stn_mode
            self.hbsn_stn_rate = hbsn_stn_rate
            self.hbsn_version = hbsn_version
        
        # print(self.hbsn_checkpoint)
        if self.hbsn_checkpoint:
            hbsn_checkpoint, hbsn_config, _, _, _ = BaseNet.load_model(self.hbsn_checkpoint, self.device)

            if self.hbsn_version == 1:
                if isinstance(hbsn_config, dict):
                    self.hbsn = HBSNet(
                        self.height, self.width, 1, 2, 
                        channels=hbsn_config['channels'], stn_mode=hbsn_config['stn_mode'], 
                        radius=hbsn_config['radius'], stn_rate=hbsn_config['stn_rate'],
                        device=self.device, dtype=self.dtype)
                else:
                    self.hbsn = HBSNet(
                        self.height, self.width, 1, 2, 
                        channels=hbsn_config.channels, stn_mode=hbsn_config.stn_mode, 
                        radius=hbsn_config.radius, stn_rate=hbsn_config.stn_rate,
                        device=self.device, dtype=self.dtype)

            else:
                self.hbsn = HBSNet_V2(
                    self.height, self.width, 1, 2,
                    channels_down=hbsn_config.channels_down, channels_up=hbsn_config.channels_up,
                    stn_mode=hbsn_config.stn_mode, radius=hbsn_config.radius, stn_rate=hbsn_config.stn_rate,
                    device=self.device, dtype=self.dtype)
            self.hbsn.load_state_dict(hbsn_checkpoint["state_dict"])
        else:
            if self.hbsn_version == 1:
                self.hbsn = HBSNet(
                    self.height, self.width, 1, 2, 
                    self.hbsn_channels, self.hbsn_radius,
                    self.hbsn_stn_mode, self.hbsn_stn_rate,
                    device=self.device, dtype=self.dtype
                )
            else:
                self.hbsn = HBSNet_V2(
                    self.height, self.width, 1, 2,
                    self.hbsn_channels[0], self.hbsn_channels[1], self.hbsn_radius,
                    self.hbsn_stn_mode, self.hbsn_stn_rate,
                    device=self.device, dtype=self.dtype
                )
        
        self.to(self.device)

    def _handle_checkpoint(self, checkpoint):
        self.load_strict = False
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("hbsn"):
                del checkpoint["state_dict"][key]
        return super()._handle_checkpoint(checkpoint)

    def model_forward(self, images):
        raise NotImplementedError("model_forward not implemented")

    def forward(self, images):
        mask = self.model_forward(images)
        mask = self.get_mask(mask)
        hbs = self.hbsn(self.get_mask(mask))
        return mask, hbs

    def get_mask(self, x):
        # eps = 1/scale
        # y = -torch.relu(-torch.relu(x+eps)+2*eps)+eps
        # y = (torch.sin(y*torch.pi/2/eps)+1)/2
        y = torch.sigmoid(self.mask_scale * (x - 0.5))
        return y
    
    def get_hard_mask(self, x):
        return torch.relu(torch.sign(x - 0.5))

    def get_metrics(self, predict, ground_truth):
        tp = (predict * ground_truth).sum(dim=(1,2,3))
        fp = (predict * (1 - ground_truth)).sum(dim=(1,2,3))
        fn = ((1 - predict) * ground_truth).sum(dim=(1,2,3))
        
        f1 = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        iou = tp / (tp + fp + fn)
        return f1, iou

    def loss(self, predict, ground_truth):
        predict_mask, predict_hbs = predict
        mse_loss = F.mse_loss(predict_mask, ground_truth)
        f1, iou = self.get_metrics(self.get_mask(predict_mask), ground_truth)
        f1 = f1.mean()
        iou = iou.mean()
        dice_loss = 1 - f1
        iou_loss = 1 - iou
        loss = mse_loss + self.dice_rate * dice_loss +self.iou_rate * iou_loss
        
        ground_truth_hbs = self.hbsn(ground_truth)
        hbs_loss_dict, (_, ground_truth_hbs) = self.hbsn.loss(predict_hbs, ground_truth_hbs)
        
        loss = loss + self.hbs_loss_rate * hbs_loss_dict["loss"]
        
        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "dice": f1,
            "iou": iou,
            "hbs_loss": hbs_loss_dict["hbs_loss"],
        }
        return loss_dict, (predict_mask, predict_hbs, ground_truth_hbs)

    @property
    def fixable_layers(self):
        return self.hbsn
    
    @property
    def uninitializable_layers(self):
        return self.hbsn
