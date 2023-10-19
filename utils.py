
from scipy.interpolate import griddata
import numpy as np
import cv2
import torch
from scipy.spatial import Delaunay

def move_pixels_numpy(I: np.ndarray, f: np.ndarray, vertex: np.ndarray):
    '''
    INPUT:
        I: C x H x W numpy array
            I is the input image
        f: H x W x 2 numpy array
            f is the map, f(J) = I
        vertex: H x W x 2 numpy array
            original cordination of each pixel
    OUTPUT:
        J: C x H x W numpy array
            J is the output image
    '''
    if I.ndim == 2:
        C = 1
        H, W = I.shape
    elif I.ndim == 3:
        C, H, W = I.shape
    else:
        raise ValueError('I should be 2 or 3 dimension')

    f = f.reshape(H*W,2)
    vertex = vertex.reshape(H*W,2)
    I = I.reshape(H*W)
    J = griddata(f, I, vertex, fill_value=0)
    
    if C == 1:
        return J.reshape(H,W)
    else:
        return J.reshape(C,H,W)
    
def create_rect_mesh(h:int, w:int):
    '''
    Create a rectangle delaunay mesh grid with size h x w
    
    INPUT:
        h: int
            height of the image
        w: int
            width of the image
    OUTPUT:
        face: N x 3 numpy array
            face is the index of the vertex that form a triangle
        vertex: h x w x 2 numpy array
            coordinate of each vertex
    '''
    vy, vx = torch.meshgrid(torch.arange(h), torch.arange(w))
    vertex = torch.stack([vx, vy], dim=-1).double()

    tri = Delaunay(vertex.reshape(-1,2), incremental=True)
    face = torch.from_numpy(tri.simplices).int()
    return face, vertex

def read_image(path: str, gray:bool=True, 
               binary_threshold: int = 0, 
               noramlize: bool = False, 
               CHW=False):
    '''
    INPUT:
        path: str
            path to the image
        gray: bool
            if True, then the image will be converted to grayscale
        binary_threshold: int
            if not 0, then the image will be binarized according to this threshold
        noramlize: bool
            if True, then the image will be normalized to [0,1], otherwise [0,255]
        CHW: bool
            if True, then the image will be converted to C x H x W
    OUTPUT:
        I: C x H x W or H x W numpy array (depends on CHW)
            I is the image
    '''
    I = cv2.imread(path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    H, W, C = I.shape
    
    if gray:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    if binary_threshold:
        _, I = cv2.threshold(I, binary_threshold, 255, cv2.THRESH_BINARY)

    if noramlize:
        I = I / 255.0

    if CHW:
        if not gray:
            I = I.transpose(2,0,1)
        else:
            I = I.reshape(1,H,W)

    return I

def image_meshgen(height, width, normal=False):
    """
    Inputs:
        height: int
        width: int
    Outputs:
        face : m x 3 index of trangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
        boundary_mask: m x 1 mask of boundary faces, 
                       1 for faces having boundary points, 0 for others
    """    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    y = y[::-1, :]
    if normal:
        x = x / (width-1)
        y = y / (height-1)
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    vertex = np.hstack((x, y))
    # vy, vx = torch.meshgrid(torch.arange(height), torch.arange(width))
    # vertex = torch.stack([vx, vy], dim=-1)

    face = np.zeros(((height-1)*(width-1)*2, 3))
    ind = np.arange(height*width).reshape((height, width))
    mid = ind[0:-1, 1:]
    left1 = ind[0:-1, 0:-1]
    left2 = ind[1:, 1:]
    right = ind[1:, 0:-1]
    face[0::2, 0] = left1.reshape(-1)
    face[0::2, 1] = right.reshape(-1)
    face[0::2, 2] = mid.reshape(-1)
    face[1::2, 0] = left2.reshape(-1)
    face[1::2, 1] = mid.reshape(-1)
    face[1::2, 2] = right.reshape(-1)

    # boundary is 1, the other faces is 0
    boundary_mask = np.zeros(((height-1)*(width-1)*2, 3))
    ind_mask = np.arange(height*width).reshape((height, width)) + 1
    ind_mask[1:-1, 1:-1] = 0
    ind_mask[ind_mask > 0] = 1
    mid_mask = ind_mask[0:-1, 1:]
    left1_mask = ind_mask[0:-1, 0:-1]
    left2_mask = ind_mask[1:, 1:]
    right_mask = ind_mask[1:, 0:-1]
    boundary_mask[0::2, 0] = left1_mask.reshape(-1)
    boundary_mask[0::2, 1] = right_mask.reshape(-1)
    boundary_mask[0::2, 2] = mid_mask.reshape(-1)
    boundary_mask[1::2, 0] = left2_mask.reshape(-1)
    boundary_mask[1::2, 1] = mid_mask.reshape(-1)
    boundary_mask[1::2, 2] = right_mask.reshape(-1)
    boundary_mask = (np.sum(boundary_mask, axis = 1) > 0) + 0.0

    return face, vertex, boundary_mask