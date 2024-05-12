import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)

from net.seg_hbsn_net import SegHBSNNet


class DeepLab(SegHBSNNet):
    def __init__(
        self, height=256, width=256, input_channels=3, output_channels=1,
        dice_rate=0.1, iou_rate=0, hbs_loss_rate=1, mask_scale=100,
        hbsn_checkpoint='',
        hbsn_channels=[64, 128, 256, 512], hbsn_radius=50,
        hbsn_stn_mode=0, hbsn_stn_rate=0.0, 
        dtype=torch.float32, device="cpu"
    ):
        super().__init__(
            height, width, input_channels, output_channels,
            dice_rate, iou_rate, hbs_loss_rate, mask_scale,
            hbsn_checkpoint, hbsn_channels, hbsn_radius, hbsn_stn_mode, hbsn_stn_rate,
            dtype, device
        )
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        
        self.mask_conv = nn.Sequential(
            nn.Conv2d(21, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.to(device)

    def model_forward(self, x):
        x = self.model(x)['out']
        mask = self.mask_conv(x)
        return mask
    
    @property
    def fixable_layers(self):
        return nn.ModuleList([
            super().fixable_layers,
            self.model.backbone,
            self.model.aux_classifier
        ])
        
    @property
    def uninitializable_layers(self):
        return nn.ModuleList([
            super().uninitializable_layers,
            self.model
        ])
