"""空间变换网络（STN）。state_dict 键：localization/fc_loc1/fc_loc2/fc_loc。"""

import torch
import torch.nn.functional as F
from torch import nn


class STN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        height=256,
        width=256,
        stn_mode=1,
        dtype=torch.float32,
    ):
        super().__init__()
        assert stn_mode in [0, 1, 2], "stn_mode should be 0, 1, or 2"

        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.dtype = dtype
        self.stn_mode = stn_mode
        # 0 - free affine, 1 - rotation+translation+scale, 2 - rotation only

        loc_conv1_kernel_size = 7
        loc_conv1_out_channels = 8
        loc_conv2_kernel_size = 5
        loc_conv2_out_channels = 10
        loc_maxpool_kernel_size = 2
        loc_maxpool_stride = 4

        self.localization = nn.Sequential(
            nn.Conv2d(
                input_channels,
                loc_conv1_out_channels,
                kernel_size=loc_conv1_kernel_size,
                dtype=dtype,
            ),
            nn.MaxPool2d(loc_maxpool_kernel_size, stride=loc_maxpool_stride),
            nn.ReLU(True),
            nn.Conv2d(
                loc_conv1_out_channels,
                loc_conv2_out_channels,
                kernel_size=loc_conv2_kernel_size,
                dtype=dtype,
            ),
            nn.MaxPool2d(loc_maxpool_kernel_size, stride=loc_maxpool_stride),
            nn.ReLU(True),
        )

        def get_loc_output_size(s):
            # 沿 localization 链逐层推算输出尺寸：conv 减 (k-1)，池化按 ceil 模式
            # (s + stride - kernel) // stride；链为 conv1(k=7) → pool(k=2,s=4) →
            # conv2(k=5) → pool(k=2,s=4)。128×128 输入 → 2×2，256×256 → 7×7。
            s = s - (loc_conv1_kernel_size - 1)
            s = (
                s + (loc_maxpool_stride - loc_maxpool_kernel_size)
            ) // loc_maxpool_stride
            s = s - (loc_conv2_kernel_size - 1)
            s = (
                s + (loc_maxpool_stride - loc_maxpool_kernel_size)
            ) // loc_maxpool_stride
            return s

        output_height = get_loc_output_size(height)
        output_width = get_loc_output_size(width)
        self.fc_loc_input_size = (
            loc_conv2_out_channels * output_height * output_width
        )

        if self.stn_mode == 0:
            fc_loc_output_size = 6
            fc_bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=dtype)
        elif self.stn_mode == 1:
            fc_loc_output_size = 4
            fc_bias = torch.tensor([0, 1, 0, 0], dtype=dtype)
        elif self.stn_mode == 2:
            fc_loc_output_size = 1
            fc_bias = torch.tensor([0], dtype=dtype)

        self.fc_loc1 = nn.Linear(self.fc_loc_input_size, 32, dtype=dtype)
        self.fc_loc2 = nn.Linear(32, fc_loc_output_size, dtype=dtype)
        self.fc_loc2.weight.data.zero_()
        self.fc_loc2.bias.data.copy_(fc_bias)
        self.fc_loc = nn.Sequential(self.fc_loc1, nn.ReLU(True), self.fc_loc2)

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_loc_input_size)
        loc = self.fc_loc(xs)

        if self.stn_mode == 0:
            p = loc.view(-1, 2, 3)
            theta = p  # 全仿射参数；下游 loss 只用 predict，theta 仅占位
        else:
            if self.stn_mode == 1:
                theta, scale, dx, dy = loc.split(1, dim=1)
                dx = torch.sigmoid(dx) - 0.5
                dy = torch.sigmoid(dy) - 0.5
            elif self.stn_mode == 2:
                theta = loc.view(-1)
                scale = 1
                dx = dy = torch.zeros_like(theta)

            p = torch.stack(
                [
                    torch.stack(
                        [
                            torch.cos(theta) / scale,
                            torch.sin(theta) / scale,
                            dx,
                        ],
                        dim=1,
                    ),
                    torch.stack(
                        [
                            -torch.sin(theta) / scale,
                            torch.cos(theta) / scale,
                            dy,
                        ],
                        dim=1,
                    ),
                ],
                dim=1,
            ).reshape(-1, 2, 3)

        grid = F.affine_grid(p, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x, theta
