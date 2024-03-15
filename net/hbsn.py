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
        device="cpu", dtype=DTYPE, stn_mode=0
        ):
        # stn_mode: 0 - no stn, 
        #         1 - stn control, 
        #         2 - stn rotation only
        super(HBSNet, self).__init__()
        self.dtype = dtype
        self.device = device
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stn_mode = stn_mode
        
        if stn_mode == 1:
            self.stn = STN(input_channels, height, width, dtype=dtype, is_rotation_only=False)
        elif stn_mode == 2:
            self.stn = STN(output_channels, height, width, dtype=dtype, is_rotation_only=True)
        self.bce = BCENet(input_channels, output_channels, channels, bilinear=True, dtype=dtype)

        self.to(device)

    def forward(self, x):
        if self.stn_mode == 1:
            x = self.stn(x)
        x = self.bce(x)
        if self.stn_mode == 2:
            x = self.stn(x)
        return x

    def loss(self, predict, label):
        if self.stn_mode == 2:
            label = self.stn(label)
        return F.mse_loss(predict, label, reduction="mean")

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

    def save(self, path, epoch, best_epoch, best_loss, optimizer=None, scheduler=None):
        torch.save({
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None
        }, path)
        
    def load(self, path, optimizer=None, scheduler=None):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        
        if optimizer and checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        if scheduler and checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])
        return epoch, best_epoch, best_loss
