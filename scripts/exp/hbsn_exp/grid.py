"""GHBS 网格与经典/HBSN 场互转。

统一网格 = 128×128 图、圆心 (64,64)、圆盘半径 50 像素（与 HBSN 输出的 mask 一致），
单位圆盘内像素数 ≈ π·50² ≈ 7845 —— 即 spec 的 GHBS 网格 N。

经典 hbs 库输出定义在三角网格（DiskMesh 面中心），HBSN 输出在 128×128 像素场。
grid_bridge 在两者间插值：
- tri_to_pixel: 三角面中心值 → 像素场（LinearNDInterpolator）
- pixel_to_tri: 像素场 → 三角面中心（反插值，供同一网格上比较）
"""

import numpy as np
from hbs.mesh import get_unit_disk
from scipy.interpolate import LinearNDInterpolator, griddata

H = W = 128  # 像素网格分辨率（与 HBSN 输出一致）
R = 50.0  # 圆盘半径（像素）
CX = CY = 64.0  # 圆心（像素坐标）

# spec 网格目标：单位圆盘内像素数 ≈ π·50² ≈ 7845
# get_unit_disk(density=1/50) 生成 ~7353 内点，三角面 ~15000
DISK_DENSITY = 1 / 50
DISK_CIRCLE_NUM = 300


def get_ghbs_grid():
    """GHBS 像素网格：返回 (z, mask)。z 为 128×128 复数场（单位圆盘坐标），mask 为圆盘掩膜。"""
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx - CX) / R
    v = (CY - yy) / R
    z = u + 1j * v
    mask = np.abs(z) <= 1.0
    return z, mask


_disk_cache = None


def get_disk():
    """单位圆盘 DiskMesh（hbs 库网格，density≈1/50 → 内点数≈GHBS 像素数）。

    模块级缓存：Delaunay 构建很贵，E2 等实验每形状都调用，不能重复构建。
    """
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = get_unit_disk(
            density=DISK_DENSITY, circle_point_num=DISK_CIRCLE_NUM
        )
    return _disk_cache


def tri_to_pixel(hbs_tri, disk):
    """三角面中心值（复数场）→ 128×128 像素场（圆盘内）。"""
    z, mask = get_ghbs_grid()
    pts = np.stack([z.real, z.imag], axis=-1).reshape(-1, 2)
    interp = LinearNDInterpolator(
        disk.face_center, np.stack([hbs_tri.real, hbs_tri.imag], axis=-1)
    )
    val = interp(pts).reshape(H, W, 2)
    out = (val[..., 0] + 1j * val[..., 1]) * mask
    out = np.nan_to_num(out)
    return out


def pixel_to_tri(pixel_field, disk):
    """128×128 像素场 → 三角面中心值（同一网格上对比用）。"""
    z, mask = get_ghbs_grid()
    complex_pts = np.stack([z.real, z.imag], axis=-1)
    val = pixel_field  # (H,W) 复数
    re = griddata(
        complex_pts.reshape(-1, 2)[mask.ravel()],
        val.real[mask],
        disk.face_center,
        method="linear",
    )
    im = griddata(
        complex_pts.reshape(-1, 2)[mask.ravel()],
        val.imag[mask],
        disk.face_center,
        method="linear",
    )
    return (np.nan_to_num(re) + 1j * np.nan_to_num(im)).astype(complex)


def disk_mask_pixels():
    """圆盘内像素数（GHBS 网格 N）。"""
    _, mask = get_ghbs_grid()
    return int(mask.sum())
