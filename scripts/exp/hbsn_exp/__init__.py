"""scripts/exp 实验工具包：经典 HBS（hbs_python fork）+ HBSN 稳定性实验（E1-E4）。

import 本包即自动：
1. 优先加载 ~/projects/hbs_python 的 hbs fork（numpy 2.x 修复 + lsQC eps 正则），
   而非 PyPI hbs==1.0.0——fork 根目录可用环境变量 HBS_PYTHON_DIR 覆盖。
2. 应用 _compat 兼容 patch（fork 未覆盖的差异）。
"""

import os
import sys

_HBS_FORK = os.environ.get("HBS_PYTHON_DIR", "/home/nnb/projects/hbs_python")
if os.path.isdir(os.path.join(_HBS_FORK, "hbs")) and _HBS_FORK not in sys.path:
    sys.path.insert(0, _HBS_FORK)

from ._compat import apply_patches  # noqa: E402  # 须在 sys.path 覆盖后加载 hbs

apply_patches()

from . import classic, grid, ops, shapes  # noqa: E402  # 重导出供实验脚本使用

__all__ = ["classic", "grid", "ops", "shapes"]
