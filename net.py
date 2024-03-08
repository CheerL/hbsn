import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32

# def qc_loss(predict_map, img, Dx, Dy, face, vertex, k=1.0, alpha=1.0, beta=1.0, gamma=1e-8, label=None):
#     """
#     predict_map: N x H x W x 2 tensor
#     img: N x 1 x H x W tensor
#     """
#     mu = bc_metric(predict_map, Dx, Dy)
#     N,C,H,W = img.shape
#     face_is_inside = img.reshape(N, -1)[:, face].all(2)
#     # vertex_is_inside = img > 0.5
#     # mu loss
#     mu = mu.masked_select(face_is_inside)
#     nan_mask = mu.isnan()
#     if nan_mask.sum() > 0:
#         mu_loss = torch.norm(mu.masked_select(~nan_mask))
#     else:
#         mu_loss = torch.norm(mu)

#     # img loss
#     img_loss = torch.norm(check_inside_unit_disk(predict_map) - img)
    
#     # area_loss
    
#     area = get_area(predict_map, Dx, Dy).abs()
#     eps = 1e-9
#     area_loss = torch.tanh(gamma / (area.masked_select(face_is_inside) + eps)).norm()

#     # label loss
#     if label is not None:
#         label_loss = torch.norm((predict_map - label) * (img.reshape(N,H,W,1)))
#     else:
#         label_loss = 0

#     total_loss = k * img_loss + alpha * mu_loss + beta * label_loss + area_loss
#     return total_loss, img_loss, mu_loss, label_loss, area_loss

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
    def __init__(self, n_channels, n_classes, bilinear=True, dtype=DTYPE):
        super(BCENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ch1 = 8
        ch2 = 16
        ch3 = 32
        ch4 = 64
        # ch5 = 32
        self.inc = DoubleConv(n_channels, ch1, dtype=dtype)
        self.down1 = Down(ch1, ch2, dtype=dtype)
        self.down2 = Down(ch2, ch3, dtype=dtype)
        self.down3 = Down(ch3, ch4, dtype=dtype)
        # self.down4 = Down(ch4, ch5, dtype=dtype)
        # self.up4 = Up(ch5, ch4, ch4, bilinear, dtype=dtype)
        self.up3 = Up(ch4, ch3, ch3, bilinear, dtype=dtype)
        self.up2 = Up(ch3, ch2, ch2, bilinear, dtype=dtype)
        self.up1 = Up(ch2, ch1, ch1, bilinear, dtype=dtype)
        self.outc = nn.Conv2d(ch1, n_classes, kernel_size=1, dtype=dtype)

    def forward(self, x):
        x_0 = self.inc(x)
        x_1 = self.down1(x_0)
        x_2 = self.down2(x_1)
        x_3 = self.down3(x_2)
        # x_4 = self.down3(x_3)
        # x_3 = self.up4(x_4, x_3)
        x_2 = self.up3(x_3, x_2)
        x_1 = self.up2(x_2, x_1)
        x_0 = self.up1(x_1, x_0)
        x_out = self.outc(x_0)
        return x_out

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
    def __init__(self, height, width, device="cpu", dtype=DTYPE):
        super(HBSNet, self).__init__()
        self.height = height
        self.width = width
        self.bce = BCENet(1, 2, True, dtype=dtype)
        self.to(device)

    def forward(self, x):
        x.reshape(-1, 1, self.height, self.width)
        x = self.bce(x)
        x = x.transpose(1, 3)
        return x

    def loss(self, predict, label):
        return F.mse_loss(predict, label, reduction="sum")

    def initialize(self):
        for m in self.bce.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

