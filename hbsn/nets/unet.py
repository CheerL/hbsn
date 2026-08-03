"""UNet 与 UNet2D 两个变体的合并实现。

state_dict 键约束（checkpoint 兼容，勿改属性名）：
- UNet（HBSNet backbone）：inc/downs/ups/outc
- UNet2D（TPSN）：Net 这个 Sequential 的 Input/DownN/UpN/Output/Upsample
类名不影响 state_dict，可自由调整（unet2d 的 DoubleConv/Down 改名为
ConvBlock/DownConv，仅避免与 UNet 版同名冲突）。
"""

import torch
from torch import nn

# ---------------------------------------------------------------- UNet 版（带 BN）


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dtype=torch.float32):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dtype=dtype),
            nn.BatchNorm2d(mid_channels, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, dtype=dtype),
            nn.BatchNorm2d(out_channels, dtype=dtype),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dtype=dtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, dtype=torch.float32):
        super().__init__()
        self.is_bilinear = bilinear
        self.is_skip = skip_channels > 0

        if self.is_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, dtype=dtype),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=dtype
            )

        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, dtype=dtype)

    def forward(self, x, x2):
        x = self.up(x)
        if self.is_skip:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        channels_down,
        channels_up,
        is_bilinear=True,
        dtype=torch.float32,
        is_skip=True,
        is_sigmoid=False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.is_bilinear = is_bilinear
        self.is_skip = is_skip
        self.is_sigmoid = is_sigmoid
        self.channels_down = channels_down
        self.channels_up = channels_up
        self.layers_down = len(channels_down)
        self.layers_up = len(channels_up)

        self.inc = DoubleConv(n_channels, channels_down[0], dtype=dtype)
        self.downs = nn.ModuleList(
            [
                DownBlock(channels_down[i], channels_down[i + 1], dtype=dtype)
                for i in range(self.layers_down - 1)
            ]
        )
        self.ups = nn.ModuleList(
            [
                UpBlock(
                    channels_up[i + 1],
                    channels_down[i + self.layers_down - self.layers_up] if is_skip else 0,
                    channels_up[i],
                    bilinear=is_bilinear,
                    dtype=dtype,
                )
                for i in range(self.layers_up - 1)
            ]
        )
        self.outc = nn.Conv2d(channels_up[0], n_classes, kernel_size=1, dtype=dtype)

    def encode(self, x):
        features = [x]
        for i in range(self.layers_down - 1):
            x = self.downs[i](x)
            features.append(x)
        return features

    def decode(self, features):
        x = features[-1]
        for i in range(self.layers_up - 1):
            x = self.ups[self.layers_up - 2 - i](
                x, features[self.layers_down - 2 - i]
            )
        return x

    def forward(self, x):
        x = self.inc(x)
        x = self.encode(x)
        x = self.decode(x)
        x = self.outc(x)
        if self.is_sigmoid:
            x = torch.sigmoid(x)
        return x


# -------------------------------------------------------------- UNet2D 版（无 BN）


class ConvBlock(nn.Module):
    """(conv => ReLU) * 2，无 BN（TPSN 用，键名 double_conv）"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """Downscaling with maxpool then conv block（键名 maxpool_conv）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_seg=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.activation = nn.Sigmoid() if is_seg else None

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, n_input=2, n_output=2, n_feature=8, depth_down=2, depth_hidden=1, bilinear=True, is_seg=True):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.bilinear = bilinear
        self.Net = nn.Sequential()
        self.Net.add_module("Input", ConvBlock(n_input, n_feature))
        for i in range(depth_down):
            self.Net.add_module(
                f"Down{i}", DownConv(n_feature * 2**i, n_feature * 2 ** (i + 1))
            )
        n_next = n_feature * 2**depth_down
        for i in range(depth_hidden):
            n_this = n_feature * 2 ** (depth_down - i)
            n_next = n_feature * 2 ** (depth_down - i - 1)
            self.Net.add_module(f"Up{i}", ConvBlock(n_this, n_next))
        self.Net.add_module("Output", OutConv(n_next, n_output, is_seg))
        self.Net.add_module(
            "Upsample",
            nn.Upsample(scale_factor=2**depth_down, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        return self.Net(x)

    def zero_initialization(self):
        self.Net[-2].conv.weight.data.zero_()
        self.Net[-2].conv.bias.data.zero_()
