"""HBS 场数值操作：L² 距离、I₂ 积分、残差作用、对称度量、翻转检测。

场统一为 128×128 复数场（单位圆盘坐标，圆盘外 0），z/mask 来自 grid.get_ghbs_grid。
"""

import numpy as np
from scipy.ndimage import map_coordinates

from .grid import CX, CY, R, get_ghbs_grid


def d2(a, b, z=None, mask=None):
    """圆盘掩膜上 L² 距离。"""
    if z is None:
        z, mask = get_ghbs_grid()
    diff = a - b
    return np.sqrt(np.sum(np.abs(diff[mask]) ** 2))


def i2(field, z=None, mask=None):
    """I₂(B) = ∫_D B(z)/z dz ≈ Σ B(z)/z · ΔA（z=0 处 B(0)=0）。

    ΔA = (1/R)²：单位圆盘坐标下像素面积（R=50 像素 ↔ 单位半径）。
    """
    if z is None:
        z, mask = get_ghbs_grid()
    da = (1.0 / R) ** 2
    zz = np.where(z == 0, 1.0, z)  # z=0 处先除 1 避免警告，再置 0
    integrand = field / zz
    integrand[z == 0] = 0.0  # 奇异点处理（B(0)=0 正则性）
    return np.sum(integrand[mask]) * da


def rotate_mu(field, theta, z=None, mask=None):
    """残差作用 R_θ B(z) = e^{-2iθ} B(e^{iθ} z)。

    对像素场做逆旋转插值（map_coordinates）+ 乘相位；圆盘外补 0。
    """
    if z is None:
        z, mask = get_ghbs_grid()
    wz = z * np.exp(1j * theta)  # 源点 w = e^{iθ} z
    src_col = wz.real * R + CX
    src_row = CY - wz.imag * R
    re = map_coordinates(
        field.real, [src_row, src_col], order=1, mode="constant", cval=0
    )
    im = map_coordinates(
        field.imag, [src_row, src_col], order=1, mode="constant", cval=0
    )
    out = (re + 1j * im) * np.exp(-2j * theta)
    return out * mask


def sym(field, n_theta=360, z=None, mask=None):
    """对称度量 sym(B) = min_θ ‖R_θ B − B‖∞（θ 均匀网格，近似 Σ 距离）。"""
    if z is None:
        z, mask = get_ghbs_grid()
    best = np.inf
    for k in range(1, n_theta):  # 排除平凡旋转 θ=0（R_0 B=B 恒使 sym=0）
        theta = 2 * np.pi * k / n_theta
        r = rotate_mu(field, theta, z, mask)
        err = np.max(np.abs(r - field)[mask])
        if err < best:
            best = err
    return best


def is_flip(a, b, z=None, mask=None):
    """180° 翻转检测：B' 与 R_π B' 更近视为翻转。

    经典归一化（hbs 库 N2）在 Im I₂ 变号时对域旋转 π：B → R_πB = B(-z)，
    **不是**全局负号 -B。实证见 E1 诊断（d(a,Rπb)≪d(a,-b)）。
    """
    rpi_b = rotate_mu(b, np.pi, z, mask)
    return d2(a, rpi_b, z, mask) < d2(a, b, z, mask)


def flip_count(path, z=None, mask=None):
    """沿参数路径（场序列）统计 180° 翻转次数。"""
    n = 0
    for i in range(len(path) - 1):
        if is_flip(path[i], path[i + 1], z, mask):
            n += 1
    return n
