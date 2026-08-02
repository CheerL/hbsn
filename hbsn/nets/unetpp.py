"""UnetPlusPlus(smp, resnet50) 分割头。state_dict 键：model.*（smp 0.3.3 键集）。"""
import segmentation_models_pytorch as smp
import torch.nn as nn

from hbsn.nets.segmentation import SegHBSNNet


class UnetPP(SegHBSNNet):
    def build_model(self):
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=1,
            activation="sigmoid",
        )

    def model_forward(self, x):
        return self.model(x)

    @property
    def fixable_layers(self):
        return nn.ModuleList([super().fixable_layers, self.model.encoder])

    @property
    def uninitializable_layers(self):
        return nn.ModuleList([super().uninitializable_layers, self.model])
