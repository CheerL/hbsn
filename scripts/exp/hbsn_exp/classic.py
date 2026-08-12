"""经典 HBS 计算（hbs 库 + 渲染-提取预处理）。

关键：hbs 库的 zipper 焊接对**几何精确边界**（圆/椭圆/三角形）数值退化
（Beltrami 饱和 0.9999、while 不收敛）。对照 MATLAB 仓库（ChanceAroundYou/hbs）
的图像形状（ell1/fish/tp，像素化边界）全部正常，确认像素化锯齿使焊接鲁棒。

因此经典 HBS 统一走：几何边界 → 渲染二值图 → cv2 提取像素化边界 → get_hbs。
"""

import numpy as np
from hbs import get_hbs

from . import grid, shapes
from .safe import HBSTimeout, timed_hbs

__all__ = ["classic_hbs"]


def classic_hbs(bound, disk=None, timeout=15.0, retries=4):
    """几何边界 → 经典 HBS 三角场。

    bound: (n,2) 几何边界点（单位圆盘内，逆时针）
    disk: 可选 DiskMesh（默认 GHBS 网格）
    timeout: 单次 get_hbs 超时（s）；retries: 失败/超时后扰动重试次数
    返回 (hbs_tri, mapping, cw)：hbs_tri 为 (face_num,) 复数 Beltrami 场。

    zipper/归一化对个别尖角/近对称形状数值崩溃（纯虚数中间值、Σhbs≈0 振荡）——
    失败或超时对边界做亚像素扰动重试（不改形状实质）。纯虚数输入防护已入 fork zipper。
    """
    img = shapes.shape_to_image(bound)[..., 0]
    bd = shapes.boundary_from_image(img)
    if disk is None:
        disk = grid.get_disk()
    for attempt in range(retries):
        try:
            hbs, mapping, cw, _ = timed_hbs(
                get_hbs,
                bd,
                circle_point_num=grid.DISK_CIRCLE_NUM,
                density=grid.DISK_DENSITY,
                disk=disk,
                timeout=timeout,
            )
            if np.isfinite(hbs).all():
                return hbs, mapping, cw
        except (ValueError, HBSTimeout, RuntimeError):
            pass
        # 扰动边界重试：随机平移（HBS 归一化吸收平移+旋转，不改输出代表）
        rng = np.random.default_rng(attempt)
        shift = rng.normal(0, 0.3, (1, 2))
        bd = bd + shift
    raise RuntimeError("get_hbs 连续失败（zipper 数值不稳定）")


def classic_pixel(bound, disk=None, timeout=15.0, retries=4):
    """几何边界 → 经典 HBS 像素场（128×128 复数场，GHBS 网格）。"""
    hbs, _, _ = classic_hbs(bound, disk, timeout=timeout, retries=retries)
    return grid.tri_to_pixel(hbs, disk if disk is not None else grid.get_disk())


def classic_pixel_from_img(img, disk=None, timeout=15.0):
    """已渲染图像 → 经典 HBS 像素场（图像→边界→get_hbs，不做居中重渲染）。"""
    bd = shapes.boundary_from_image(img)
    if disk is None:
        disk = grid.get_disk()
    hbs, *_ = timed_hbs(
        get_hbs,
        bd,
        circle_point_num=grid.DISK_CIRCLE_NUM,
        density=grid.DISK_DENSITY,
        disk=disk,
        timeout=timeout,
    )
    if not np.isfinite(hbs).all():
        raise RuntimeError("hbs 含 NaN")
    return grid.tri_to_pixel(hbs, disk)
