#!/usr/bin/env python3
"""E5 大图合成：噪声鲁棒性折线图（A）与 5×5 可视化网格（B）拼成一张 AB 图。

两子图左右并排（A 左 B 右），同高对齐，宽度按各自宽高比自然分配；A/B 标题同字体。

用法：先跑 plot_noise_curves.py 与 vis_m1m2m3w.py 生成两张子图，再运行本脚本。
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVE = "figures/hbsn/f_noise_robust_cons.png"
VIS = "figures/hbsn/f_noise_vis_m1m2m3w.png"
OUT = "figures/hbsn/f_noise_ab.png"


def crop_white(arr):
    """裁剪近白边距，使两子图以内容框对齐（避免残留白边破坏同宽对齐）。"""
    arr = arr[:, :, :3]
    content = ~(arr > 0.98).all(axis=-1)
    rows = np.any(content, axis=1)
    cols = np.any(content, axis=0)
    if not rows.any():
        return arr
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return arr[r0:r1, c0:c1]


def main():
    curve = crop_white(plt.imread(CURVE))
    vis = crop_white(plt.imread(VIS))
    ac = curve.shape[1] / curve.shape[0]  # A 宽高比（高窄折线图）
    av = vis.shape[1] / vis.shape[0]  # B 宽高比（正方形网格）

    hb = 10.0  # 两子图同高（英寸）
    wb = hb * av  # B 宽（信息密度大，更宽）
    wa = hb * ac  # A 宽（高窄比例 → 比 B 窄）
    pad = 0.3  # 子图水平间距
    margin = 0.15  # 左右边距
    top = 0.6  # 顶部标题空间
    W = wa + wb + pad + 2 * margin
    H = hb + top

    fig = plt.figure(figsize=(W, H))
    panels = [
        (  # A：左，同高、顶对齐 → 标题与 B 对齐
            fig.add_axes([margin / W, top / H, wa / W, hb / H]),
            curve,
            "(a)",
        ),
        (  # B：右，同高
            fig.add_axes([(margin + wa + pad) / W, top / H, wb / W, hb / H]),
            vis,
            "(b)",
        ),
    ]
    for ax, img, tag in panels:
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(tag, fontsize=16)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"output: {OUT}  (A {wa:.1f}x{hb:.1f}in, B {wb:.1f}x{hb:.1f}in, 同高)")


if __name__ == "__main__":
    main()
