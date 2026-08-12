#!/usr/bin/env python3
"""E4 · 变换不变性：经典 HBS 与 HBSN 对平移/缩放/旋转的轨道内 L² 方差。

20 测试形状，各取变换轨道（平移 ±10/20px、缩放 ×0.7/0.8/1.2/1.3、旋转 30/45/60/90°）。
轨道内方差 = mean_k ‖B_k − B̄‖²（L²）。经典按理论精确不变（相似变换），HBSN 近似不变。

产出 tables/tab_invariance.tex。
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, nets, shapes

TAB_DIR = os.path.join(os.path.dirname(__file__), "../../tables")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../figures/hbsn")
N_SHAPES = 20
TRANSLATIONS = [(0, 0), (10, 0), (-10, 0), (0, 10), (0, -10), (20, 0), (0, 20)]
SCALES = [1.0, 0.8, 1.2, 0.7, 1.3]
ROTATIONS = [0, 30, 45, 60, 90]


def gen_pool(n=N_SHAPES, seed=3):
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        k = i % 3
        if k == 0:
            pool.append(shapes.ellipse(b=rng.uniform(0.5, 0.9)))
        elif k == 1:
            pool.append(shapes.triangle(t=rng.uniform(0.3, 0.7), base=0.6))
        else:
            pool.append(
                shapes.random_polygon(verts=int(rng.integers(5, 8)), rng=rng)
            )
    return pool


def irregular_shape():
    """E4 代表形状：固定 seed 的随机 7 边形（明显不规则、可复现）。"""
    return shapes.random_polygon(verts=7, rng=np.random.default_rng(3))


def render_transformed(bound, offset_px=(0, 0), scale=1.0, rot_deg=0.0):
    """渲染变换后的形状图（变换可见，不做居中归一化）。

    offset_px 为像素偏移（图 256×256，数据坐标 [-1.5,1.5] ↔ 像素 [0,255]）。
    """
    z = bound[:, 0] + 1j * bound[:, 1]
    z = z - z.mean()
    z = z / np.abs(z).max() * 0.85
    z = z * np.exp(1j * np.deg2rad(rot_deg)) * scale
    z = z + offset_px[0] * (3.0 / 256) + 1j * offset_px[1] * (3.0 / 256)
    fig = plt.figure(figsize=(256, 256), facecolor="black", dpi=1)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axis("off")
    plt.fill(z.real, z.imag, color="white")
    plt.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def orbit_variance(fields, z, mask):
    """轨道内 L² 方差：mean_k ‖B_k − B̄‖²（论文口径，除以圆盘像素数 N）。

    fields 含 None 则跳过（经典失败）。与主文 d(B,B')=(1/N)Σ|B−B'|² 一致。
    """
    valid = [f for f in fields if f is not None]
    if len(valid) < 2:
        return np.nan
    mean = np.mean(valid, axis=0)
    n = mask.sum()
    return np.mean([np.sum(np.abs(f - mean)[mask] ** 2) / n for f in valid])


def draw_orbit_figure(bound, out_path, z, mask, net):
    """轨道场热图：3 变换（translation/scale/rotation）每行 3 个代表变体，
    每变体显示 [input | classical |HBS| | HBSN |HBS|] 三图——直观验证
    变换后 input 与两法场，以及轨道内场几乎不变。

    bound 用固定不规则七边形（irregular_shape，可复现）；
    classical 在个别变体可能失败（近墙翻转/数值），失败格显示空白。
    """
    trans_sets = {
        "translation": [(0, 0), (-20, 0), (20, 0)],
        "scale": [1.0, 0.7, 1.3],
        "rotation": [0, 45, 90],
    }

    def variant_label(tname, p):
        if isinstance(p, tuple):
            return f"({p[0]},{p[1]})px"
        return f"×{p}" if tname == "scale" else f"{p}°"

    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(3, 9, hspace=0.4, wspace=0.05)
    axs = np.array(
        [[fig.add_subplot(gs[r, c]) for c in range(9)] for r in range(3)]
    )

    for r, (tname, params) in enumerate(trans_sets.items()):
        for gi, p in enumerate(params):
            if tname == "translation":
                img = render_transformed(bound, offset_px=p)
            elif tname == "scale":
                img = render_transformed(bound, scale=p)
            else:
                img = render_transformed(bound, rot_deg=p)
            gray = img[..., 0]
            c0 = 3 * gi
            # input shape（变换后的）
            axs[r, c0].imshow(gray, cmap="gray", vmin=0, vmax=255)
            axs[r, c0].axis("off")
            axs[r, c0].set_title(variant_label(tname, p), fontsize=10)
            # classical |HBS|
            try:
                cf = classic.classic_pixel_from_img(gray, timeout=8.0)
                fc = np.abs(cf) * mask
            except (RuntimeError, ValueError):
                fc = np.zeros_like(mask)
            axs[r, c0 + 1].imshow(fc, cmap="jet", vmin=0, vmax=0.8)
            axs[r, c0 + 1].axis("off")
            # HBSN |HBS|
            hf = nets.field_to_complex(nets.infer(net, gray))
            axs[r, c0 + 2].imshow(
                np.abs(hf) * mask, cmap="jet", vmin=0, vmax=0.8
            )
            axs[r, c0 + 2].axis("off")
        # 行标签（竖排，贴左缘）
        axs[r, 0].text(
            -0.1,
            0.5,
            tname,
            transform=axs[r, 0].transAxes,
            va="center",
            ha="center",
            rotation=90,
            fontsize=12,
        )

    # 列含义标签（顶部，3 组 × input/classical/HBSN）
    col_labels = ["input", "classical", "HBSN"] * 3
    fig.canvas.draw()
    for c, label in enumerate(col_labels):
        pos = axs[0, c].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.965, label, ha="center", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    z, mask = grid.get_ghbs_grid()
    net = nets.build_net(nets.BEST_CKPT)
    pool = gen_pool()

    # (name, [(param, render_call)]) 的变换类型
    trans_types = [
        ("translation", TRANSLATIONS),
        ("scale", SCALES),
        ("rotation", ROTATIONS),
    ]

    rows = []  # (name, var_classic, var_hbsn)
    for name, params in trans_types:
        vc, vh = [], []
        for bound in pool:
            cf_orbit, hf_orbit = [], []
            for p in params:
                if name == "translation":
                    img = render_transformed(bound, offset_px=p)
                elif name == "scale":
                    img = render_transformed(bound, scale=p)
                else:
                    img = render_transformed(bound, rot_deg=p)
                gray = img[..., 0]  # HBSN/boundary 均需单通道灰度
                try:
                    cf_orbit.append(
                        classic.classic_pixel_from_img(gray, timeout=8.0)
                    )
                except (RuntimeError, ValueError):
                    cf_orbit.append(None)
                hf_orbit.append(nets.field_to_complex(nets.infer(net, gray)))
            vc.append(orbit_variance(cf_orbit, z, mask))
            vh.append(orbit_variance(hf_orbit, z, mask))

        # 中位数±IQR 稳健（经典在旋转时对个别近墙形状翻转，均值为离群值拉高）
        def iqr(x):
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            return (
                np.percentile(x, 75) - np.percentile(x, 25)
                if len(x)
                else np.nan
            )

        rows.append(
            (name, np.nanmedian(vc), iqr(vc), np.nanmedian(vh), iqr(vh))
        )
        print(
            f"{name:12s}: 经典方差(中位±IQR) {np.nanmedian(vc):.4f}±{iqr(vc):.4f} "
            f"HBSN 方差(中位±IQR) {np.nanmedian(vh):.4f}±{iqr(vh):.4f}"
        )

    # ---- 轨道场热图：3 变换 × [input + classical|HBS| + HBSN|HBS|] ----
    os.makedirs(OUT_DIR, exist_ok=True)
    draw_orbit_figure(
        irregular_shape(),
        os.path.join(OUT_DIR, "f_invariance.png"),
        z,
        mask,
        net,
    )
    print(f"输出: {OUT_DIR}/f_invariance.png")

    # ---- 表格 ----
    with open(os.path.join(TAB_DIR, "tab_invariance.tex"), "w") as f:
        f.write(
            "%% E4 变换不变性：轨道内 L2 方差（经典 vs HBSN，median±IQR over 20 shapes）\n"
        )
        f.write("\\begin{tabular}{|c|c|c|}\\hline\n")
        f.write("transformation & classical & HBSN \\\\ \\hline\n")
        for name, mc, sc, mh, sh in rows:
            f.write(
                f"{name} & {mc:.4f}$\\pm${sc:.4f} & {mh:.4f}$\\pm${sh:.4f} \\\\ \\hline\n"
            )
        f.write("\\end{tabular}\n")
    print(f"输出: {TAB_DIR}/tab_invariance.tex")


if __name__ == "__main__":
    main()
