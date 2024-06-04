import torch
import torch.nn as nn


DTYPE = torch.float32

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

class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dtype=DTYPE):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dtype=dtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, dtype=DTYPE):
        super().__init__()
        self.bilinear = bilinear
        self.dtype = dtype
        self.skip = skip_channels > 0
        # if bilinear, use the normal convolutions to reduce the number of channels
        conv_in_channels = in_channels // 2 + skip_channels

        if self.bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, dtype=dtype),
                nn.ReLU(inplace=True)
            )
            self.conv = DoubleConv(conv_in_channels, out_channels, dtype=dtype)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=dtype)
            self.conv = DoubleConv(conv_in_channels, out_channels, dtype=dtype)

    def forward(self, x, x2):
        
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = self.up(x)
        if self.skip:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels, bilinear=True, dtype=DTYPE):
        super().__init__()
        self.dtype=dtype
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.channels = channels
        self.layers = len(self.channels)

        self.inc = DoubleConv(n_channels, self.channels[0], dtype=dtype)
        self.downs = nn.ModuleList([
            DownBlock(self.channels[i], self.channels[i+1], dtype=dtype) 
            for i in range(self.layers - 1)
        ])

        self.ups = nn.ModuleList([
            UpBlock(self.channels[i+1], self.channels[i], self.channels[i], bilinear, dtype=dtype) 
            for i in range(self.layers - 1)
        ])
        self.outc = nn.Conv2d(self.channels[0], n_classes, kernel_size=1, dtype=dtype)

    def encode(self, x):
        features = [x]
        for i in range(self.layers - 1):
            x = self.downs[i](x)
            features.append(x)
        return features
    
    def decode(self, features):
        x = features[-1]
        for i in range(self.layers - 1, 0, -1):
            x = self.ups[i-1](x, features[i-1])
        return x

    def forward(self, x):
        x = self.inc(x)
        x = self.encode(x)
        x = self.decode(x)
        x = self.outc(x)
        return x

class AsymmetricUNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels_down, channels_up, bilinear=True, dtype=DTYPE, skip=True):
        super().__init__()
        self.dtype=dtype
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip = skip
        
        self.channels_down = channels_down
        self.channels_up = channels_up
        self.layers_down = len(self.channels_down)
        self.layers_up = len(self.channels_up)

        self.inc = DoubleConv(n_channels, self.channels_down[0], dtype=dtype)
        self.downs = nn.ModuleList([
            DownBlock(self.channels_down[i], self.channels_down[i+1], dtype=dtype) 
            for i in range(self.layers_down - 1)
        ])

        self.ups = nn.ModuleList([
            UpBlock(
                self.channels_up[i+1], 
                self.channels_down[i+1] if self.skip else 0, 
                self.channels_up[i], 
                bilinear=self.bilinear, dtype=dtype
                ) 
            for i in range(self.layers_up - 1)
        ])

        self.outc = nn.Conv2d(self.channels_up[0], n_classes, kernel_size=1, dtype=dtype)

    def encode(self, x):
        features = [x]
        for i in range(self.layers_down - 1):
            x = self.downs[i](x)
            features.append(x)
        return features
    
    def decode(self, features):
        x = features[-1]
        for i in range(self.layers_up - 1):
            x = self.ups[self.layers_up - 2 - i](x, features[self.layers_down - 2 - i])
        return x

    def forward(self, x):
        x = self.inc(x)
        x = self.encode(x)
        x = self.decode(x)
        x = self.outc(x)
        return x
