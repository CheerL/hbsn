import torch
import torch.nn as nn
import torch.nn.functional as F

from net.bce import BCENet
from net.stn import STN

DTYPE = torch.float32

class HBSNet(nn.Module):
    def __init__(
        self, height, width,
        input_channels=1, output_channels=2,
        channels=[16, 32, 64, 128],
        device="cpu", dtype=DTYPE, is_stn=False
        ):
        super(HBSNet, self).__init__()
        self.dtype = dtype
        self.device = device
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.is_stn = is_stn
        
        if is_stn:
            self.stn = STN(input_channels, height, width, dtype=dtype, is_rotation_only=False)
        self.bce = BCENet(input_channels, output_channels, channels, bilinear=True, dtype=dtype)
        

        self.to(device)

    def forward(self, x):
        # x = x.reshape(-1, self.input_channels, self.height, self.width)
        if self.is_stn:
            x = self.stn(x)
        x = self.bce(x)
        # if self.is_stn:
        #     x = self.stn(x)
        return x

    def loss(self, predict, label):
        # if self.is_stn:
        #     label = self.stn(label)
        return F.mse_loss(predict, label, reduction="mean")

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                

