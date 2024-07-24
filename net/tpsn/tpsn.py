from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from net.seg_hbsn_net import SegHBSNNet

# import segmentation_models_pytorch as smp
# from net.unet import UNet
from net.tpsn.unet import UNet2D as UNet

imsize = [256,256]
pad_size = 16 # set it to be [0,0] if no padding is needed.

def PADDEN_IMAGE(img):
    '''
        INPUT:      img-        B,C,H,W
        OUTPUT:     padded_img- B,C,H+2p,W+2p
    '''
    return F.pad(img,[pad_size]*4, mode='constant', value=0)
def UNPAD_IMAGE(pad_img):
    '''
        INPUT:      padded_img- B,C,H+2p,W+2p
        OUTPUT:     img-        B,C,H,W
    '''
    return pad_img[:,:,pad_size:-pad_size,pad_size:-pad_size]

class BCLossFunc(torch.nn.Module):
    def __init__(self, size):
        super(BCLossFunc, self).__init__()
        self.hx1 = torch.tensor(2.0/size[0])
        self.hx2 = torch.tensor(2.0/size[1])
    def forward(self, mapping):
        u_SE,v_SE = mapping[:,1:, 1:, 0:1], mapping[:,1:, 1:, 1:2]
        u_NE,v_NE = mapping[:,0:-1,1:,0:1], mapping[:,0:-1,1:,1:2]
        u_SW,v_SW = mapping[:,1:,0:-1,0:1], mapping[:,1:,0:-1,1:2]
        u_NW,v_NW = mapping[:,0:-1, 0:-1, 0:1], mapping[:,0:-1, 0:-1,1:2]
        # forward FDM
        u_x1f = (u_SE-u_SW)/ self.hx1
        u_x2f = (u_SE-u_NE)/ self.hx2
        v_x1f = (v_SE-v_SW)/ self.hx1
        v_x2f = (v_SE-v_NE)/ self.hx2
        numerator0 = (u_x1f**2 + v_x1f**2 - v_x2f**2 - u_x2f**2)**2 + (2*u_x2f*u_x1f + 2*v_x1f*v_x2f)**2
        dedomenator0 = ((u_x1f+v_x2f)**2 + (v_x1f-u_x2f)**2)**2
        mu_square0 = numerator0/dedomenator0
        # backward FDM
        u_x1b = (u_NE-u_NW)/ self.hx1
        u_x2b = (u_SW-u_NW)/ self.hx2
        v_x1b = (v_NE-v_NW)/ self.hx1
        v_x2b = (v_SW-v_NW)/ self.hx2              
        numerator1 = (u_x1b**2 + v_x1b**2 - v_x2b**2 - u_x2b**2)**2 + (2*u_x2b*u_x1b + 2*v_x1b*v_x2b)**2
        dedomenator1 = ((u_x1b+v_x2b)**2 + (v_x1b-u_x2b)**2)**2
        mu_square1 = numerator1/dedomenator1
        
        return (torch.exp(mu_square0+mu_square1)-1).mean()


class LAPLossFunc(torch.nn.Module):
    def __init__(self, size):
        super(LAPLossFunc, self).__init__()
        kernel = torch.tensor([[0.,  1., 0.],
                               [1., -4., 1.],
                               [0.,  1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        self.hx1 = torch.tensor(2.0/size[0])
        self.hx2 = torch.tensor(2.0/size[1])
    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        # print(x1.shape, x2.shape, x.shape, 125)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0) / (self.hx1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0) / (self.hx2)
        return (torch.cat([x1, x2], dim=1) ** 2).mean()

class TPSN(SegHBSNNet):
    def build_model(self):
        self.model = UNet(
            3, 2, IsSeg=True
        )
        # The below 3 lines are for making identity grid
        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        grid_identity = F.affine_grid(theta, (1, 2, imsize[0]+2*pad_size, imsize[1]+2*pad_size))[0]
        self.grid_padded_identity = grid_identity.permute((2, 0, 1)).to(self.config.device)

        # The below 2 lines are for making trival template mask
        mask_temp = torch.ones([1,1] + imsize)
        self.mask_simple = PADDEN_IMAGE(mask_temp).to(self.config.device)

        # define QC and LAP loss
        # TODO
        self.qc_loss_rate = 0.01
        self.lap_loss_rate = 0.0001

        self.qc_loss = BCLossFunc([imsize[0]+2*pad_size, imsize[1]+2*pad_size])
        self.lap_loss = LAPLossFunc([imsize[0]+2*pad_size, imsize[1]+2*pad_size])
        # self

    def model_forward(self, img: torch.Tensor) -> torch.Tensor:
        # print(img.shape, 1)
        padded_img = PADDEN_IMAGE(img)
        # print(padded_img.shape, 2)
        padded_mask = self.mask_simple.repeat(padded_img.shape[0],1,1,1) # N, 1, H, W
        # print(padded_mask.shape, padded_mask[0], 1)
        predict_pad_vector = self.model(padded_img)
        predict_pad_vector = predict_pad_vector * 2 - 1
        print(predict_pad_vector[0, 0, pad_size:-pad_size,pad_size:-pad_size], predict_pad_vector.max())
        # print(predict_pad_vector.device, self.grid_padded_identity.device, 3)
        predict_pad_mapping = self.grid_padded_identity + predict_pad_vector  # N, 2, H, W
        # print(padded_mask.shape, predict_pad_mapping.shape, 4)
        predict_pad_mask = F.grid_sample(padded_mask, predict_pad_mapping.permute(0,2,3,1), mode='bilinear',
                                     padding_mode='border', align_corners=True)
        return predict_pad_mask, predict_pad_mapping

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Rewrite because model_forward() need to output 3 torch.tensor for loss computation
        '''
        predict_pad_mask, predict_pad_mapping = self.model_forward(img)
        # print(predict_pad_mask.shape, predict_pad_mapping.shape, 55)
        # predict_mask = self.binarize_mask(predict_mask)
        predict_mask = UNPAD_IMAGE(predict_pad_mask)
        # print(predict_mask.shape, 5)
        hbs = self.hbsn(self.binarize_mask(predict_mask))
        return predict_mask, hbs, predict_pad_mask, predict_pad_mapping

    def loss(
        self,
        predict: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ground_truth: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        predict_mask, predict_hbs, predict_pad_mask, predict_pad_mapping = predict
        mse_loss = F.mse_loss(predict_pad_mask, PADDEN_IMAGE(ground_truth))
        f1, iou = self.get_metrics(
            self.binarize_mask(predict_mask), ground_truth
        )
        f1 = f1.mean()
        iou = iou.mean()
        dice_loss = 1 - f1
        iou_loss = 1 - iou

        ground_truth_hbs = self.hbsn(ground_truth)
        hbs_loss_dict, (_, ground_truth_hbs) = self.hbsn.loss(
            predict_hbs, ground_truth_hbs
        )

        qc_loss = self.qc_loss(predict_pad_mapping)
        lap_loss = self.lap_loss(predict_pad_mapping)
        # print(qc_loss, lap_loss, 77)

        loss = (
            mse_loss
            + self.config.dice_rate * dice_loss
            + self.config.iou_rate * iou_loss
            + self.config.hbs_loss_rate * hbs_loss_dict["loss"]
            + self.qc_loss_rate * qc_loss
            + self.lap_loss_rate * lap_loss
        )

        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "dice": f1,
            "iou": iou,
            "hbs_loss": hbs_loss_dict["hbs_loss"],
            "qc_loss": qc_loss,
            "lap_loss": lap_loss,
        }
        return loss_dict, (
            predict_mask,
            predict_hbs,
            ground_truth_hbs,
        )

    def initialize(self):
        super().initialize()
        
        self.model.Net[-2].conv.weight.data.zero_()
        self.model.Net[-2].conv.bias.data.zero_()