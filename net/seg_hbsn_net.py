from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from net.base_net import BaseNet
from net.hbsn import HBSNet

DTYPE = torch.float32

class SegHBSNNet(BaseNet):
    def __init__(
        self, height=256, width=256, input_channels=3, output_channels=1, 
        dice_rate=0.1, iou_rate=0, hbs_loss_rate=1, mask_scale=100,
        hbsn_checkpoint='',
        hbsn_channels=[64, 128, 256, 512], hbsn_radius=50,
        hbsn_stn_mode=0, hbsn_stn_rate=0.0, 
        dtype=DTYPE, device="cpu"
        ):
        super().__init__(height, width, input_channels, output_channels, device, dtype)
        self.dice_rate = dice_rate
        self.iou_rate = iou_rate
        self.hbs_loss_rate = hbs_loss_rate
        self.mask_scale = mask_scale
        
        if hbsn_checkpoint:
            checkpoint, config, _, _, _ = BaseNet.load_model(hbsn_checkpoint, device)

            self.hbsn = HBSNet(
                height, width, 1, 2, 
                channels=config['channels'], stn_mode=config['stn_mode'], 
                radius=config['radius'], stn_rate=config['stn_rate'],
                device=device, dtype=dtype
            )
            self.hbsn.load_state_dict(checkpoint["state_dict"])
        else:
            self.hbsn = HBSNet(
                height, width, 1, 2, 
                hbsn_channels, hbsn_radius, hbsn_stn_mode, hbsn_stn_rate,
                device=device, dtype=dtype
            )
        
        self.to(device)

    def model_forward(self, images):
        raise NotImplementedError("model_forward not implemented")

    def forward(self, images):
        mask = self.model_forward(images)
        mask = self.get_mask(mask)
        hbs = self.hbsn(mask)
        return mask, hbs

    def get_mask(self, x):
        # eps = 1/scale
        # y = -torch.relu(-torch.relu(x+eps)+2*eps)+eps
        # y = (torch.sin(y*torch.pi/2/eps)+1)/2
        y = F.sigmoid(self.mask_scale * (x - 0.5))
        return y

    def get_metrics(self, predict, ground_truth):
        tp = (predict * ground_truth).sum()
        # tn = ((1 - predict) * (1 - ground_truth)).sum()
        fp = (predict * (1 - ground_truth)).sum()
        fn = ((1 - predict) * ground_truth).sum()
        
        f1 = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        iou = tp / (tp + fp + fn)
        return f1, iou

    def loss(self, predict, ground_truth):
        predict_mask, predict_hbs = predict
        mse_loss = F.mse_loss(predict_mask, ground_truth)
        f1, iou = self.get_metrics(predict_mask, ground_truth)
        dice_loss = 1 - f1
        iou_loss = 1 - iou
        loss = mse_loss + self.dice_rate * dice_loss +self.iou_rate * iou_loss
        
        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "dice": f1,
            "iou": iou,
        }
        
        ground_truth_hbs = self.hbsn(ground_truth)
        hbs_loss_dict, (_, ground_truth_hbs) = self.hbsn.loss(predict_hbs, ground_truth_hbs)
        loss_dict['loss'] += self.hbs_loss_rate * hbs_loss_dict["loss"]
        loss_dict['hbs_loss'] = hbs_loss_dict['hbs_loss']
        
        return loss_dict, (predict_mask, predict_hbs, ground_truth_hbs)

    @property
    def fixable_layers(self):
        return self.hbsn
    
    @property
    def uninitializable_layers(self):
        return self.hbsn
