#!/usr/bin/env python3
"""拓扑实验终稿合成（Round B）— handoff H1。

用户定稿选择（2026-09-08）：
- (a) 多连通：细化扫掠步 1/4/7/12/14（H/6 方形；步 7 为洞位居中）；
- (b) 不连通：密集扫掠步 2/5/7/8/12（W/8 切口；步 7 为全宽切断）。

布局：两组上下合成（(a) 上 / (b) 下），每组 3 行 x 5 列
（Input / HBSN (ours) / Classic HBS），第 3 列畸形、其 Classic 行 N/A。
渲染口径与候选图完全一致（复用 topology_series）；渲染帧注意见其 docstring
——论文引用尺寸以渲染 px 为准（三角形 100x71px、方形 17.3px、切口 18.1px）。

用法：uv run python scripts/exp/topology_final.py
输出：figures/hbsn/topology_results.png（新文件）
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import topology_series as ts
from hbsn_exp import grid, nets, shapes

PICKS_MULTI = (1, 4, 7, 12, 14)  # fine sweep 步号
PICKS_DISC = (2, 5, 7, 8, 12)  # dense sweep 步号
OUT = os.path.join("figures", "hbsn", "topology_results.png")


def pick_cells(base, net, disk_mask, rects, steps, strict=True):
    """按步号取格并打印 RMS 质量证据与中心坐标（world + 256px）。"""
    cells = []
    for i, k in enumerate(steps, 1):
        rect = rects[k - 1]
        cell = ts.render_cell(base, net, disk_mask, rect, strict=strict)
        _, hf, cf, kind = cell
        if kind in ts.MALFORMED:
            msg = "N/A"
        elif cf is not None:
            msg = f"{ts.rms(cf, hf, disk_mask):.4f}"
        else:
            msg = "FAIL"
        cx, cy = (rect[0] + rect[1]) / 2, (rect[2] + rect[3]) / 2
        print(
            f"  col{i} (step {k}, {kind}): RMS={msg}"
            f"  center=({cx:+.3f},{cy:+.3f})w"
            f"=({(cx + 1.5) * ts.PX:.1f},{(1.5 - cy) * ts.PX:.1f})px"
        )
        cells.append(cell)
    return cells


def draw_final(blocks, titles, out):
    """3 行 x 11 列终稿（左右并排）：(a) 板占列 0-4，(b) 板占列 6-10，
    列 5 留空分隔，横排板标题 + 左侧行标签。"""
    fig, axs = plt.subplots(3, 11, figsize=(13.0, 4.3), squeeze=False)
    fig.subplots_adjust(
        left=0.055, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04
    )
    for r in range(3):
        for b in range(2):
            for c in range(5):
                ax = axs[r, c if b == 0 else c + 6]  # (a) 列 0-4，(b) 列 6-10
                ax.axis("off")
                gray, hf, cf, _ = blocks[b][c]
                if r == 0:
                    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
                elif r == 1:
                    ax.imshow(hf, cmap="jet", vmin=0, vmax=0.8)
                elif cf is not None:
                    ax.imshow(cf, cmap="jet", vmin=0, vmax=0.8)
                else:
                    ax.imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=1)
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=13,
                        color="w",
                    )
    for r in range(3):
        axs[r, 5].axis("off")  # 中间分隔列不画轴
    for b in range(2):
        c0 = 0 if b == 0 else 6  # (a) 板列 0-4，(b) 板列 6-10
        x0 = axs[0, c0].get_position().x0
        x1 = axs[0, c0 + 4].get_position().x1
        y = axs[0, 0].get_position().y1 + 0.025
        fig.text((x0 + x1) / 2, y, titles[b], ha="center", fontsize=13)
    for r in range(3):
        pos = axs[r, 0].get_position()
        fig.text(
            0.028,
            (pos.y0 + pos.y1) / 2,
            ts.ROW_LABELS[r],
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
        )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    bound = shapes.triangle(ts.T, base=ts.BASE)
    h, wbase, w = ts.tri_geometry(bound)
    base = shapes.shape_to_image(bound)[..., 0]
    net = nets.build_net(nets.BEST_CKPT)
    _, disk_mask = grid.get_ghbs_grid()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("== (a) multi-connected: fine steps", PICKS_MULTI, "==")
    fine = ts.fine_sweep(w, h / 6)
    cells_a = pick_cells(base, net, disk_mask, fine, PICKS_MULTI, strict=False)
    print("== (b) disconnected: dense steps", PICKS_DISC, "==")
    t = wbase / ts.FINAL_DISC["denom"]
    dense = ts.rect_sweep(w, t, ts.D_LO, ts.D_HI)
    cells_b = pick_cells(base, net, disk_mask, dense, PICKS_DISC)
    draw_final(
        [cells_a, cells_b],
        ["(a) Multi-connected shapes", "(b) Disconnected shapes"],
        OUT,
    )
    print(f"output: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
