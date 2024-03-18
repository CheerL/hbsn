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
        device="cpu", dtype=DTYPE, stn_mode=0, radius=50
        ):
        # stn_mode: 0 - no stn, 
        #           1 - pre stn
        #           2 - post stn
        #           3 - both stn
        super(HBSNet, self).__init__()
        self.dtype = dtype
        self.device = device
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stn_mode = stn_mode
        
        if stn_mode == 1 or stn_mode == 3:
            self.pre_stn = STN(input_channels, height, width, dtype=dtype, is_rotation_only=False)
        
        if stn_mode == 2 or stn_mode == 3:
            self.post_stn = STN(output_channels, height, width, dtype=dtype, is_rotation_only=True)
        
        self.bce = BCENet(input_channels, output_channels, channels, bilinear=True, dtype=dtype)
        self.mask = self.create_mask(radius)
        self.to(device)

    def create_mask(self, r):
        x, y = torch.meshgrid(torch.arange(self.width),torch.arange(self.height), indexing='ij')
        x = (x - self.width/2) / r
        y = (y - self.height/2) / r
        mask = (x**2 + y**2) <= 1
        mask.requires_grad = False
        mask = mask.to(self.device)
        return mask

    def forward(self, x):
        if self.stn_mode == 1 or self.stn_mode == 3:
            x = self.pre_stn(x)
            
        x = self.bce(x)
        
        if self.stn_mode == 2 or self.stn_mode == 3:
            x = self.post_stn(x)

        x = torch.masked_fill(x, ~self.mask, 0.0)
        return x

    def loss(self, predict, label, is_mask=True):
        if self.stn_mode == 2 or self.stn_mode == 3:
            label = self.post_stn(label)

        if is_mask:
            loss= F.mse_loss(
                torch.masked_select(predict, self.mask), 
                torch.masked_select(label, self.mask), 
                reduction="mean")
        else:
            loss = F.mse_loss(predict, label, reduction="mean")
        return loss

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

    def save(self, path, epoch, best_epoch, best_loss):
        torch.save({
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]

        return epoch, best_epoch, best_loss
