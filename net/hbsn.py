from typing import Optional

import torch
import torch.nn.functional as F

from config import HBSNetConfig
from net.base_net import BaseNet
from net.stn import STN
from net.unet import UNet

DTYPE = torch.float32

class HBSNet(BaseNet):
    def __init__(
        self, height=256, width=256,
        input_channels=1, output_channels=2,
        channels=[16, 32, 64, 128], radius=50,
        stn_mode=0, stn_rate=0.0,
        device="cpu", dtype=DTYPE, config: Optional[HBSNetConfig]=None
        ):
        # stn_mode: 0 - no stn, 
        #           1 - pre stn
        #           2 - post stn
        #           3 - both stn
        super().__init__(height, width, input_channels, output_channels, device, dtype, config)
        if config:
            self.stn_mode = config.stn_mode
            self.stn_rate = config.stn_rate
            self.channels = config.channels
            self.radius = config.radius
        else:
            self.stn_mode = stn_mode
            self.stn_rate = stn_rate
            self.channels = channels
            self.radius = radius
        
        self.pre_stn = (
            STN(self.input_channels, self.height, self.width, dtype=self.dtype)
            if self.stn_mode in [1, 3] else
            lambda x: (x, None)
        )
        
        self.post_stn = (
            STN(self.output_channels, self.height, self.width, dtype=self.dtype, stn_mode=2)
            if self.stn_mode in [2, 3] else
            lambda x: (x, None)
        )
        
        self.bce = UNet(self.input_channels, self.output_channels, self.channels, bilinear=True, dtype=self.dtype)
        self.mask = self.create_mask(self.radius)
        self.to(self.device)

    def create_mask(self, r):
        x, y = torch.meshgrid(torch.arange(self.width),torch.arange(self.height), indexing='ij')
        x = (x - self.width/2) / r
        y = (y - self.height/2) / r
        mask = (x**2 + y**2) <= 1
        mask.requires_grad = False
        mask = mask.to(self.device)
        return mask

    def forward(self, x):
        x, pre_theta = self.pre_stn(x)
        x = self.bce(x)
        x, post_theta = self.post_stn(x)
        x = torch.masked_fill(x, ~self.mask, 0.0)
        return x

    def loss(self, predict_hbs, ground_truth_hbs, is_mask=True):
        ground_truth_hbs, theta = self.post_stn(ground_truth_hbs)
        double_stn_predict_hbs, double_theta = self.post_stn(predict_hbs)
        stn_loss = F.mse_loss(double_stn_predict_hbs, predict_hbs, reduction="mean")

        if is_mask:
            hbs_loss = F.mse_loss(
                torch.masked_select(predict_hbs, self.mask), 
                torch.masked_select(ground_truth_hbs, self.mask), 
                reduction="mean")
        else:
            hbs_loss = F.mse_loss(predict_hbs, ground_truth_hbs, reduction="mean")

        loss = hbs_loss + stn_loss * self.stn_rate
        loss_dict = {
            "loss": loss,
            "hbs_loss": hbs_loss,
            "stn_loss": stn_loss
        }
        return loss_dict, (predict_hbs, ground_truth_hbs)

    @staticmethod
    def load_model(path, device=None, dtype=DTYPE):
        checkpoint, config, epoch, best_epoch, best_loss = BaseNet.load_model(path, device)
    
        from data.hbsn_dataset import HBSNDataset
        if config['is_augment']:
            augment_rotation, augment_scale, augment_translate = config['is_augment']
            dataset = HBSNDataset(
                config['data_dir'], is_augment=True,
                augment_rotation=augment_rotation, 
                augment_scale=augment_scale, 
                augment_translate=augment_translate
            )
        else:
            dataset = HBSNDataset(config['data_dir'], is_augment=False)

        H, W, C_input, C_output = dataset.get_size()
        train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=config['batch_size'])
        
        net = HBSNet(
            height=H, width=W, input_channels=C_input, output_channels=C_output, 
            channels=config['channels'], device=device or config['device'], 
            dtype=dtype, stn_mode=config['stn_mode'], radius=config['radius'],
            stn_rate=config['stn_rate']
            )
        net.load_state_dict(checkpoint["state_dict"])

        return net, epoch, best_epoch, best_loss, dataset, train_dataloader, test_dataloader


class HBSNet_FCN(HBSNet):
    def __init__(
        self, height=256, width=256,
        input_channels=1, output_channels=2,
        channels=[16, 32, 64, 128], radius=50,
        stn_mode=0, stn_rate=0.0,
        device="cpu", dtype=DTYPE, config: Optional[HBSNetConfig]=None
    ):
        super().__init__(
            height, width, input_channels, output_channels,
            channels, radius, stn_mode, stn_rate, device, dtype, config
        )
        self.bce = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=False)
        self.bce.backbone.conv1 = torch.nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bce.classifier._modules['4'] = torch.nn.Conv2d(512, self.output_channels, kernel_size=(1, 1), stride=(1, 1))
        
        self.to(self.device)
        
    def forward(self, x):
        x, pre_theta = self.pre_stn(x)
        x = self.bce(x)['out']
        x, post_theta = self.post_stn(x)
        x = torch.masked_fill(x, ~self.mask, 0.0)
        return x