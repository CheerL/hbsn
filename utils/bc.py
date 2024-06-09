import torch


def bc_metric(f_map: torch.Tensor, Dx, Dy):
    """
    Compute the Beltrami coefficient of a map f

    INPUT:
        f_map: N x H x W x 2 tensor
            f_map is the coridinate of vertex after mapping
            f_map = f(vertex)
    OUTPUT:
        mu: N x M complex tensor
            mu is the Beltrami coefficient
            M is the number of faces
    """
    N, H, W, _ = f_map.shape
    f_map = f_map.reshape(N, -1, 2).transpose(0, 1).double()

    fx = f_map[:, :, 0]
    fy = f_map[:, :, 1]
    fz = torch.ones(H * W, N, dtype=f_map.dtype, device=f_map.device)

    dXdu = Dx.mm(fx).transpose(0, 1)
    dXdv = Dy.mm(fx).transpose(0, 1)
    dYdu = Dx.mm(fy).transpose(0, 1)
    dYdv = Dy.mm(fy).transpose(0, 1)
    dZdu = Dx.mm(fz).transpose(0, 1)
    dZdv = Dy.mm(fz).transpose(0, 1)

    E = dXdu.pow(2) + dYdu.pow(2) + dZdu.pow(2)
    G = dXdv.pow(2) + dYdv.pow(2) + dZdv.pow(2)
    F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv

    mu = (E - G + 2j * F) / (E + G + 2 * torch.sqrt(E * G - F.pow(2)))
    return mu


def get_area(f_map, Dx, Dy):
    N, H, W, _ = f_map.shape
    f_map = f_map.reshape(N, -1, 2).transpose(0, 1).double()
    fx = f_map[:, :, 0]
    fy = f_map[:, :, 1]

    e1x = Dx.mm(fx).transpose(0, 1)
    e1y = Dx.mm(fy).transpose(0, 1)
    e2x = Dy.mm(fx).transpose(0, 1)
    e2y = Dy.mm(fy).transpose(0, 1)

    area = (e1x * e2y - e1y * e2x) / 2
    return area


def diff_operator(face, vertex, dtype=torch.float64):
    """
    Compute Dx, Dy for further Beltrami coefficient computation.
    Seperate this function out since face and vertex are always constant
    for a given mesh and we don't need to compute Dx, Dy every time we
    compute BC.

    INPUT:
        face: M x 3 tensor
            face is the index of the vertex that form a triangle
        vertex: N x 2 tensor
            coordinate of each vertex. N = H*W

    OUTPUT:
        Dx: M x N sparse tensor
        Dy: M x N sparse tensor
    """
    m = vertex.size(0)
    n = face.size(0)

    Mi = torch.arange(n).repeat_interleave(3)
    Mj = face.flatten()

    e1 = vertex[face[:, 2]] - vertex[face[:, 1]]
    e2 = vertex[face[:, 0]] - vertex[face[:, 2]]
    e3 = vertex[face[:, 1]] - vertex[face[:, 0]]
    area = (e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]) / 2

    Mx = (
        torch.stack(
            [
                e1[:, 1] / area / 2,
                e2[:, 1] / area / 2,
                e3[:, 1] / area / 2,
            ]
        )
        .t()
        .flatten()
    )
    My = (
        torch.stack(
            [
                -e1[:, 0] / area / 2,
                -e2[:, 0] / area / 2,
                -e3[:, 0] / area / 2,
            ]
        )
        .t()
        .flatten()
    )

    indices = torch.stack([Mi, Mj])
    Dx = torch.sparse_coo_tensor(
        indices=indices, values=Mx, size=(n, m), dtype=dtype
    )
    Dy = torch.sparse_coo_tensor(
        indices=indices, values=My, size=(n, m), dtype=dtype
    )

    return Dx, Dy
