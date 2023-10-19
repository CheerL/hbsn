import torch
import torch.nn as nn
import torch.nn.functional as F
from bc import bc_metric, diff_operator
from unet import Unet
from utils import create_rect_mesh

def move_pixels_inverse_torch(I, f):
    '''
    INPUT:
        I: N x C x H x W tensor
            I is the input image
        f: N x H x W x 2 tensor
            f is the map, f(J) = I
    OUTPUT:
        J: N x C x H x W tensor
            J is the output image
    '''
    _, _, H, W = I.shape
    center = torch.DoubleTensor([W,H]) / 2
    target_xy = (f - center) / center
    J = F.grid_sample(I, target_xy, mode='bilinear', padding_mode='zeros',align_corners=False)
    return J

def check_inside_unit_disk(vertex: torch.Tensor, k=1e9):
    '''
    INPUT:
        vertex: N x H x W x 2 tensor
            original cordination of each pixel
    OUTPUT:
        result: N X H x W tensor
            whether the pixel is inside the unit circle
    '''
    # return torch.tanh((1 - torch.norm(vertex, 2, 3))*k) / 2 + 0.5
    return torch.sigmoid((1 - torch.norm(vertex, 2, 3))*k)

def qc_loss(predict_map, img, Dx, Dy, alpha=1., beta=1., label=None):
    '''
    predict_map: N x H x W x 2 tensor
    img: N x H x W tensor
    '''
    mu = bc_metric(predict_map, Dx, Dy)
    nan_mask = mu.isnan()
    # print(nan_mask.sum())
    if nan_mask.sum() > 0:
        mu_loss = torch.norm(mu.masked_select(~nan_mask))
    else:
        mu_loss = torch.norm(mu)
    
    img_loss = torch.norm(check_inside_unit_disk(predict_map) - img)

    if label is not None:
        label_loss = torch.norm(predict_map - label)
    else:
        label_loss = 0

    return img_loss+ alpha * mu_loss + beta * label_loss

class ConformalNet(nn.Module):
    def __init__(self, height, width, alpha=1., beta=1., device='cpu'):
        super(ConformalNet, self).__init__()
        self.height = height
        self.width = width
        self.alpha = alpha
        self.beta = beta
        self.face, self.vertex = create_rect_mesh(height, width)
        self.Dx, self.Dy = diff_operator(self.face, self.vertex.reshape(-1,2))
        self.unet = Unet(2)
        
        self.Dx = self.Dx.to(device)
        self.Dy = self.Dy.to(device)
        self.unet.to(device)
        
    def forward(self, x):
        x.reshape(-1, 1, self.height, self.width)
        x = self.unet(x)
        x = x.transpose(1, 3)
        return x

    def loss(self, predict, img, label=None):
        return qc_loss(predict, img, self.Dx, self.Dy, self.alpha, self.beta, label)