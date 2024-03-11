import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata


def move_image(I, f, vertex=None, version='torch'):
    """
    INPUT:
        I: H x W tensor
            I is the input image
        f: N x H x W x 2 tensor
            f is the map, f(J) = I
    OUTPUT:
        J: N x C x H x W tensor
            J is the output image
    """
    if I.ndim == 2:
        N = 1
        C = 1
        H, W = I.shape
    elif I.ndim == 3:
        N = 1
        H, W, C = I.shape
    elif isinstance(I, torch.Tensor) and I.ndim == 4:
        assert version == 'torch'
        N, C, H, W = I.shape
    else:
        raise ValueError('I should be 2 or 3 dimension')
    
    if version == 'torch':
        if N == 1:
            I = torch.DoubleTensor(I).reshape(N, H, W, C).permute(0, 3, 1, 2)
            f = torch.tensor(f).reshape(N, H, W, 2)
            J = move_image_torch(I, f)
            J = J.permute(0, 2, 3, 1)
            if C == 1:
                J = J.reshape(H, W)
            else:
                J = J.reshape(H, W, C)
            J = J.numpy().copy()
        else:
            J = move_image_torch(I, f)
            
    elif version == 'scipy':
        J = move_image_scipy(I.reshape(H, W, C), f, vertex)
    else:
        raise ValueError('version should be torch or scipy')

    return J

def move_image_scipy(I: np.ndarray, f: np.ndarray, vertex: np.ndarray = None):
    '''
    INPUT:
        I: H x W x C numpy array
            I is the input image
        f: H x W x 2 numpy array
            f is the map, f(J) = I
        vertex: (optional) H x W x 2 numpy array
            original cordination of each pixel
    OUTPUT:
        J: H x W x C numpy array
            J is the output image
    '''
    H, W, C = I.shape

    if vertex is None:
        vx = np.linspace(-1, 1, W)
        vy = np.linspace(-1, 1, H)
        vx, vy = np.meshgrid(vx, vy)
        vertex = np.stack([vx, vy], axis=-1)

    f = f.reshape(H*W,2)
    vertex = vertex.reshape(H*W,2)
    I = I.reshape(H*W, C)
    J = griddata(f, I, vertex, fill_value=0).reshape(H, W, C)
    return J


def move_image_torch(I, f):
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
    # _, _, H, W = I.shape
    # center = torch.tensor([W, H]) / 2
    # center.type(f.dtype)
    # target_xy = (f - center) / center
    J = F.grid_sample(
        I, f, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return J

