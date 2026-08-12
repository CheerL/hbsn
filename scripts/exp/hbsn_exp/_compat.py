"""hbs 库兼容 patch，import 本包时自动应用。

当前加载 hbs_python fork（已含 numpy 2.x 标量叉积、interp1d 外插、
zipper 纯虚数入口防护、归一化/居中循环上限）。此处的 #1/#2 幂等保留，
仅当 fork 加载失败回退 PyPI hbs==1.0.0 时才有实际作用。

1. `Mesh.get_area` numpy 2.x 标量叉积（fork 已修）
2. `ConformalWelding.linear_interp` interp1d 周期外插（fork 已修）
3. 边界点若含重复点（linspace endpoint=True）zipper 会折叠 —— shapes.py 保证 endpoint=False
4. 经典 HBS 对几何精确边界数值退化（Beltrami 饱和）—— classic.py 用渲染-提取预处理
"""

import numpy as np
from hbs.conformal_welding import ConformalWelding
from hbs.mesh import Mesh
from scipy.interpolate import interp1d


def _area(self):
    e1, e2, _ = self.edges
    return (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2


def _linear_interp(self, num):
    x_angle_regular = np.arange(0, 2 * np.pi, 2 * np.pi / num)
    self.x = np.exp(x_angle_regular * 1j)
    inter_func = interp1d(
        np.concatenate(
            [
                self.init_x_angle - 2 * np.pi,
                self.init_x_angle,
                self.init_x_angle + 2 * np.pi,
            ]
        ),
        np.concatenate([self.init_y, self.init_y, self.init_y]),
        kind="linear",
        fill_value="extrapolate",
    )
    y = inter_func(x_angle_regular)
    y_anlge = np.angle(y)
    self.y = np.exp((y_anlge - y_anlge[0]) * 1j)


def apply_patches():
    """幂等应用 hbs 兼容 patch（进程内只需一次）。"""
    if not getattr(apply_patches, "_done", False):
        Mesh.get_area = _area
        ConformalWelding.linear_interp = _linear_interp
        apply_patches._done = True
