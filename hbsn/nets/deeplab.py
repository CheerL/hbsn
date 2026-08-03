"""DeepLabV3(ResNet50) + mask_conv 分割头。state_dict 键：model.* / mask_conv.*。"""
import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from hbsn.nets.segmentation import SegHBSNNet


class DeepLab(SegHBSNNet):
    def build_model(self):
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.mask_conv = nn.Sequential(nn.Conv2d(21, 1, kernel_size=1), nn.Sigmoid())

    def model_forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.model(img)["out"]
        return self.mask_conv(x)

    @property
    def fixable_layers(self):
        return nn.ModuleList([super().fixable_layers, self.model.backbone])

    @property
    def uninitializable_layers(self):
        return nn.ModuleList([super().uninitializable_layers, self.model])
