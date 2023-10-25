import torch
import torch.nn as nn
import torch.nn.functional as F
from bc import bc_metric, diff_operator, get_area
from unet import UNet
from utils import create_rect_mesh

DTYPE = torch.float32

def move_pixels_inverse_torch(I, f):
    """
    INPUT:
        I: N x C x H x W tensor
            I is the input image
        f: N x H x W x 2 tensor
            f is the map, f(J) = I
    OUTPUT:
        J: N x C x H x W tensor
            J is the output image
    """
    _, _, H, W = I.shape
    center = torch.tensor([W, H]) / 2
    center.type(f.dtype)
    target_xy = (f - center) / center
    J = F.grid_sample(
        I, target_xy, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return J


def check_inside_unit_disk(vertex: torch.Tensor, k=1e7):
    """
    INPUT:
        vertex: N x H x W x 2 tensor
            original cordination of each pixel
    OUTPUT:
        result: N X H x W tensor
            whether the pixel is inside the unit circle
    """
    # return torch.tanh((1 - torch.norm(vertex, 2, 3))*k) / 2 + 0.5
    return torch.sigmoid((1 - torch.norm(vertex, 2, 3)) * k)


def qc_loss(predict_map, img, Dx, Dy, face, vertex, k=1.0, alpha=1.0, beta=1.0, gamma=1e-8, label=None):
    """
    predict_map: N x H x W x 2 tensor
    img: N x 1 x H x W tensor
    """
    mu = bc_metric(predict_map, Dx, Dy)
    N,C,H,W = img.shape
    face_is_inside = img.reshape(N, -1)[:, face].all(2)
    # vertex_is_inside = img > 0.5
    # mu loss
    mu = mu.masked_select(face_is_inside)
    nan_mask = mu.isnan()
    if nan_mask.sum() > 0:
        mu_loss = torch.norm(mu.masked_select(~nan_mask))
    else:
        mu_loss = torch.norm(mu)

    # img loss
    img_loss = torch.norm(check_inside_unit_disk(predict_map) - img)
    
    # area_loss
    
    area = get_area(predict_map, Dx, Dy).abs()
    eps = 1e-9
    area_loss = torch.tanh(gamma / (area.masked_select(face_is_inside) + eps)).norm()

    # label loss
    if label is not None:
        label_loss = torch.norm((predict_map - label) * (img.reshape(N,H,W,1)))
    else:
        label_loss = 0

    total_loss = k * img_loss + alpha * mu_loss + beta * label_loss + area_loss
    return total_loss, img_loss, mu_loss, label_loss, area_loss


class ConformalNet(nn.Module):
    def __init__(self, height, width, k=1.0, alpha=1.0, beta=1.0, device="cpu", dtype=DTYPE):
        super(ConformalNet, self).__init__()
        self.height = height
        self.width = width
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.face, self.vertex = create_rect_mesh(height, width)
        self.Dx, self.Dy = diff_operator(self.face, self.vertex.reshape(-1, 2))
        self.unet = UNet(1, 2, False, dtype=dtype)

        self.Dx = self.Dx.to(device)
        self.Dy = self.Dy.to(device)
        self.unet.to(device)

    def forward(self, x):
        x.reshape(-1, 1, self.height, self.width)
        x = self.unet(x)
        x = x.transpose(1, 3)
        return x

    def loss(self, predict, img, label=None):
        return qc_loss(
            predict, img, self.Dx, self.Dy, self.k, self.alpha, self.beta, label
        )
        
    def initialize(self):
        for m in self.unet.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

