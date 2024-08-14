""" Full assembly of the parts to form the complete network """

import torch
import torch.nn.functional as F
import torch.nn as nn

class UNet2D(nn.Module):
    def __init__(self, n_input=2, n_output=2, n_feature=8, depth_down=2, depth_hidden=1, bilinear=True, IsSeg=True):
        super(UNet2D, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.bilinear = bilinear
        self.Net = nn.Sequential()
        self.Net.add_module("Input", DoubleConv(n_input, n_feature))

        for i in range(depth_down):
            n_this = n_feature * 2**(i)
            n_next = n_feature * 2**(i+1)
            self.Net.add_module(f"Down{i}", Down(n_this, n_next))

        for i in range(depth_hidden):
            n_this = n_feature * 2**(depth_down-i)
            n_next = n_feature * 2**(depth_down-i-1)
            self.Net.add_module(f"Up{i}", DoubleConv(n_this, n_next))

        self.Net.add_module("Output", OutConv(n_next, n_output, IsSeg))
        self.Net.add_module("Upsample", nn.Upsample(scale_factor=2**(depth_down), mode='bilinear', align_corners=True))

        # total_params = sum(p.numel() for p in self.Net.parameters())
        # print(f"Number of parameters: {total_params}")
        
    def forward(self, x):
        vector = self.Net(x)
        return vector
    
    def zero_initalization(self):
        self.Net[-2].conv.weight.data.zero_()
        self.Net[-2].conv.bias.data.zero_()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, IsSeg=True):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.acit = nn.Sigmoid() if IsSeg else None
    def forward(self, x):
        x = self.conv(x)
        if self.acit is not None:
            x = self.acit(x)
        #x = self.Upsample(x)
        return x