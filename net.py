from unittest.mock import DEFAULT
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
DEFAULT_CHANNELS = [16, 32, 64, 128]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dtype=DTYPE):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dtype=dtype),
            nn.BatchNorm2d(mid_channels, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, dtype=dtype),
            nn.BatchNorm2d(out_channels, dtype=dtype),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dtype=DTYPE):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dtype=dtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, dtype=DTYPE):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, dtype=dtype),
                nn.ReLU(inplace=True)
            )
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, dtype=dtype)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=dtype)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, dtype=dtype)

    def forward(self, x, x2):
        x1 = self.up(x)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        out = torch.cat([x2, x1], dim=1)
        return self.conv(out)

class BCENet(nn.Module):
    def __init__(self, n_channels, n_classes, channels=DEFAULT_CHANNELS, bilinear=True, dtype=DTYPE):
        super(BCENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.channels = channels
        self.layers = len(self.channels)

        self.inc = DoubleConv(n_channels, self.channels[0], dtype=dtype)
        # ch1 = 16
        # ch2 = 32
        # ch3 = 64
        # ch4 = 128
        # self.down1 = Down(ch1, ch2, dtype=dtype)
        # self.down2 = Down(ch2, ch3, dtype=dtype)
        # self.down3 = Down(ch3, ch4, dtype=dtype)
        # self.up3 = Up(ch4, ch3, ch3, bilinear, dtype=dtype)
        # self.up2 = Up(ch3, ch2, ch2, bilinear, dtype=dtype)
        # self.up1 = Up(ch2, ch1, ch1, bilinear, dtype=dtype)
        self.downs = nn.ModuleList([
            Down(self.channels[i], self.channels[i+1], dtype=dtype) 
            for i in range(self.layers - 1)
        ])

        self.ups = nn.ModuleList([
            Up(self.channels[i+1], self.channels[i], self.channels[i], bilinear, dtype=dtype) 
            for i in range(self.layers - 1)
        ])
        self.outc = nn.Conv2d(self.channels[0], n_classes, kernel_size=1, dtype=dtype)

    def forward_down(self, x):
        output = [x]
        for i in range(self.layers - 1):
            x = self.downs[i](x)
            output.append(x)
        return output
    
    def forward_up(self, output):
        x = output[-1]
        for i in range(self.layers - 1, 0, -1):
            x = self.ups[i-1](x, output[i-1])
        return x

    def forward(self, x):
        x = self.inc(x)
        # x_1 = self.down1(x)
        # x_2 = self.down2(x_1)
        # x_3 = self.down3(x_2)
        # x_2 = self.up3(x_3, x_2)
        # x_1 = self.up2(x_2, x_1)
        # x = self.up1(x_1, x)
        x = self.forward_down(x)
        x = self.forward_up(x)
        x = self.outc(x)
        return x

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.outc = torch.utils.checkpoint(self.outc)


class HBSNet(nn.Module):
    def __init__(self, height, width,  input_channels=1, output_channels=2, channels=DEFAULT_CHANNELS, device="cpu", dtype=DTYPE):
        super(HBSNet, self).__init__()
        self.dtype = dtype
        self.device = device
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bce = BCENet(input_channels, output_channels, channels, bilinear=True, dtype=dtype)
        self.to(device)

    def forward(self, x):
        # x = x.reshape(-1, self.input_channels, self.height, self.width)
        x = self.bce(x)
        return x

    def loss(self, predict, label):
        return F.mse_loss(predict, label, reduction="mean")

    def initialize(self):
        for m in self.bce.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

