#!/usr/bin/env python3
"""E5 · 2 折（Z2）对称形状对 E3 对称分层的污染验证（内部 QA，latex internal spec）。

问题：E3 的 sym 度量 sym_geom 用 n∈{3,4,5,6,8,10,12} 旋转残差（**去掉 n=2**），
识别不了 2 折对称（矩形/菱形/中心对称多边形）。而理论 R_πB=B ⟹ (1-e^{-πi})I₂(B)
=2I₂(B)=0 ⟹ I₂(B)=0 被强制 → 这些形状贴近经典 HBS 退化点（实轴墙）。
若它们 sym 偏大（误分高箱）且经典翻率高，则污染 E3 的"翻率随 sym 单调降"分层。

形状族：
- rectangle(ar)：宽高比 ar∈[0.2,5] 对数扫过（ar=1 正方形 4 折，其余 2 折）
- rhombus(θ)：两邻边夹角 θ∈[60°,120°]（θ=90° 为正方形）
- centro(n_rng)：随机中心对称多边形（顶点对 v/-v，仅 180° 旋转对称）
- 对照：triangle-b07（E3 复用，n≥3 族，sym 度量可识别）

测量（复用 E3）：sym_geom / |I₂|(经典场与 HBSN 场) / 相邻对 180° 翻转率（经典/HBSN）。
产出 tables/tab_2fold.tex + 控制台结论。
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e3_flip_strat import sym_geom
from hbsn_exp import classic, grid, nets, ops, shapes

TAB_DIR = os.path.join(os.path.dirname(__file__), "../../tables")
CLASS_TIMEOUT, CLASS_RETRIES = 5.0, 1
RECT_ARS = np.logspace(np.log10(0.2), np.log10(5.0), 200)
RHOMBUS_THS = np.linspace(60, 120, 121)
CENTRO_N = 150
TRI_TS = np.linspace(0.2, 0.8, 200)


def rectangle(ar, n=300):
    """宽高比 ar 的矩形：半宽 0.85、半高 0.85/ar。ar=1 正方形（4 折）。"""
    w, h = 0.85, 0.85 / max(ar, 1e-3)
    verts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    edges = []
    for i in range(4):
        p0, p1 = verts[i], verts[(i + 1) % 4]
        seg = np.linspace(0, 1, n // 4 + 1, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def rhombus(th_deg, n=300):
    """两邻边夹角 θ 的菱形（等边平行四边形，中心对称 Z2）。θ=90° 为正方形。"""
    th = np.deg2rad(th_deg)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([np.cos(th), np.sin(th)])
    verts = np.array([v1, v2, -v1, -v2])
    edges = []
    for i in range(4):
        p0, p1 = verts[i], verts[(i + 1) % 4]
        seg = np.linspace(0, 1, n // 4 + 1, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def _polygon_from_polar(r, th, n=300):
    """极坐标顶点对 (±r e^{iθ}) → 中心对称凸多边形边界（角度排序）。"""
    z = r * np.exp(1j * th)
    z = np.concatenate([z, -z])
    idx = np.argsort(np.angle(z))
    z = z[idx]
    t = np.linspace(0, 1, n, endpoint=False)
    zz = []
    for i in range(len(z)):
        a, b = z[i], z[(i + 1) % len(z)]
        seg = a + (b - a) * t
        zz.append(seg[:-1])
    zz = np.concatenate(zz)
    zz = np.concatenate([zz, [z[0]]])
    bd = np.stack([zz.real, zz.imag], axis=1)
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def centro_sweep(seeds=(0, 1, 2), n_steps=50, n=300):
    """中心对称（Z2）多边形**连续**族：每 seed 固定基础顶点对，扫第 0 半顶点半径。

    关键：翻转率统计要求相邻形状沿参数路径连续（is_flip 才有几何意义）。
    随机独立形状序列相邻对无关，is_flip 退化为噪声（≈50%）。本族连续变形，
    保持 R_π 对称且无其他旋转对称（k≥3 半顶点）。
    """
    shapes = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        k = int(rng.integers(3, 7))
        th = np.sort(rng.uniform(0, np.pi, k))
        r = rng.uniform(0.4, 1.0, k)
        for v in np.linspace(0.4, 1.0, n_steps):
            rr = r.copy()
            rr[0] = v
            shapes.append(_polygon_from_polar(rr, th, n))
    return shapes


def gen_families():
    """[(name, [bound, ...])]：连续参数扫描族。"""
    fams = []
    fams.append(("rectangle", [rectangle(ar) for ar in RECT_ARS]))
    fams.append(("rhombus", [rhombus(th) for th in RHOMBUS_THS]))
    fams.append(("centro-Z2", centro_sweep()))
    fams.append(
        ("triangle-b07", [shapes.triangle(t=t, base=0.7) for t in TRI_TS])
    )
    return fams


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(TAB_DIR, exist_ok=True)
    z, mask = grid.get_ghbs_grid()
    net = nets.build_net(nets.BEST_CKPT)

    rows = []  # (name, n_pairs, sym_med, |I2|_c_med, |I2|_h_med, flip_c, flip_h)
    for fam_name, shapes_list in gen_families():
        if only is not None and fam_name != only:
            continue
        prev = None
        n_pairs = n_flip_c = n_flip_h = 0
        syms, i2c, i2h = [], [], []
        for bound in shapes_list:
            img = shapes.shape_to_image(bound)[..., 0]
            try:
                cf = classic.classic_pixel(
                    bound, timeout=CLASS_TIMEOUT, retries=CLASS_RETRIES
                )
            except RuntimeError:
                cf = None
            hf = nets.field_to_complex(nets.infer(net, img))
            syms.append(sym_geom(bound))
            i2c.append(abs(ops.i2(cf, z, mask)) if cf is not None else np.nan)
            i2h.append(abs(ops.i2(hf, z, mask)))
            if prev is not None:
                pcf, phf = prev
                if pcf is not None and cf is not None:
                    n_pairs += 1
                    n_flip_c += int(ops.is_flip(pcf, cf, z, mask))
                if True:
                    n_flip_h += int(ops.is_flip(phf, hf, z, mask))
            prev = (cf, hf)
        rows.append(
            (
                fam_name,
                n_pairs,
                np.nanmedian(syms),
                np.nanmedian(i2c),
                np.nanmedian(i2h),
                n_flip_c,
                n_flip_h,
            )
        )
        print(
            f"{fam_name:14s} pairs={n_pairs:4d} sym={np.nanmedian(syms):.4f} "
            f"|I2|c={np.nanmedian(i2c):.5f} |I2|h={np.nanmedian(i2h):.5f} "
            f"flip_c={n_flip_c:4d} ({100 * n_flip_c / n_pairs:.1f}%) "
            f"flip_h={n_flip_h:4d} ({100 * n_flip_h / n_pairs:.1f}%)"
        )

    out_name = f"tab_2fold_{only}.tex" if only else "tab_2fold.tex"
    with open(os.path.join(TAB_DIR, out_name), "w") as f:
        f.write(
            "%% E5 2 折对称形状：sym 误分箱 + |I2|≈0 + 经典翻率（内部 QA）\n"
        )
        f.write("\\begin{tabular}{|l|c|c|c|c|c|c|}\\hline\n")
        f.write(
            "family & pairs & sym(med) & $|I_2|_c$(med) & "
            "$|I_2|_h$(med) & flip-classic & flip-HBSN \\\\ \\hline\n"
        )
        for name, np_, sm, i2c_, i2h_, fc, fh in rows:
            f.write(
                f"{name} & {np_} & {sm:.3f} & {i2c_:.4f} & {i2h_:.4f} & "
                f"{fc}/{np_} ({100 * fc / np_:.0f}\\%) & "
                f"{fh}/{np_} ({100 * fh / np_:.0f}\\%) \\\\ \\hline\n"
            )
        f.write("\\end{tabular}\n")
    print(f"输出: {TAB_DIR}/{out_name}")


if __name__ == "__main__":
    main()
