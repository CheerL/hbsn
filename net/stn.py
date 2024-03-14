from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32

class STN(nn.Module):
    def __init__(self, input_channels=1, height=256, width=256, dtype=DTYPE):
        super(STN, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.dtype = dtype
        
        # self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        
        loc_conv1_kernel_size = 7
        loc_conv1_out_channels = 8
        loc_conv2_kernel_size = 5
        loc_conv2_out_channels = 10
        loc_maxpool_kernel_size = 2
        loc_maxpool_stride = 4

        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, loc_conv1_out_channels, kernel_size=loc_conv1_kernel_size, dtype=dtype),
            nn.MaxPool2d(loc_maxpool_kernel_size, stride=loc_maxpool_stride),
            nn.ReLU(True),
            nn.Conv2d(loc_conv1_out_channels, loc_conv2_out_channels, kernel_size=loc_conv2_kernel_size, dtype=dtype),
            nn.MaxPool2d(loc_maxpool_kernel_size, stride=loc_maxpool_stride),
            nn.ReLU(True)
        )
        
        def get_loc_output_size(s):
            s = s - (loc_conv1_kernel_size - 1)
            s = (s + (loc_maxpool_stride - loc_maxpool_kernel_size)) // loc_maxpool_stride
            s = s - (loc_conv2_kernel_size - 1)
            s = (s + (loc_maxpool_stride - loc_maxpool_kernel_size)) // loc_maxpool_stride
            return s
        
        output_height = get_loc_output_size(height)
        output_width = get_loc_output_size(width)
        self.fc_loc_input_size = loc_conv2_out_channels * output_height * output_width

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_input_size, 32, dtype=dtype),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2, dtype=dtype)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_loc_input_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(),  align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return x
