from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from config import SegNetConfig
from net.seg_hbsn_net import SegHBSNNet


class UnetPP(SegHBSNNet):
    def __init__(
        self, height=256, width=256, input_channels=3, output_channels=1,
        dice_rate=0.1, iou_rate=0, hbs_loss_rate=1, mask_scale=100,
        hbsn_checkpoint='', hbsn_version=1,
        hbsn_channels=[64, 128, 256, 512], hbsn_radius=50,
        hbsn_stn_mode=0, hbsn_stn_rate=0.0, 
        dtype=torch.float32, device="cpu", config: Optional[SegNetConfig]=None
    ):
        super().__init__(
            height, width, input_channels, output_channels,
            dice_rate, iou_rate, hbs_loss_rate, mask_scale,
            hbsn_checkpoint, hbsn_version, hbsn_channels, hbsn_radius, hbsn_stn_mode, hbsn_stn_rate,
            dtype, device, config
        )
        self.model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet', classes=1, activation='sigmoid')
        self.to(self.device)

    def model_forward(self, x):
        x = self.model(x)
        return x
    
    @property
    def fixable_layers(self):
        return nn.ModuleList([
            super().fixable_layers,
            self.model.encoder
        ])
        
    @property
    def uninitializable_layers(self):
        return nn.ModuleList([
            super().uninitializable_layers,
            self.model
        ])
