#!/usr/bin/env python3
"""拓扑实验（多连通/不连通）候选图生成 — Round A（handoff H1）。

三角形 + 遮挡物序列，展示 HBSN 对违反单连通假设输入的稳定性：
- (a) 多连通组：轴对齐方形沿质心高度水平线对称扫入，第 3 位完全进入成洞；
- (b) 不连通组：水平细矩形固定高度从边缘切入，第 3 位全宽贯穿切断。

每组 3 档遮挡尺寸（small/smaller/smallest，图内标注参数值）x 5 个遮挡位置
（1->2 渐深 [25%->60%]，第 3 位畸形，4->5 右侧镜像），每档 3x5 网格
（行 = Input / HBSN / Classic HBS），纵向堆叠 3 档成一张长图，两组各一张。
放置比例跨档恒定、仅尺寸递减，便于用户纯比选遮挡尺寸。

畸形列（多连通/不连通）经典 HBS 无定义：经典管线输入契约为单一闭曲线
（boundary_from_image 取 RETR_EXTERNAL 最大连通域，会静默丢洞/丢小块——
显示的是被篡改形状的场而非输入的场），渲染 "N/A" 文字格。
正常列质量门槛：classic 可算且两法场一致（RMS 打印，HBS 论文式 7.1 度量）。

HBS 场渲染固定口径（禁改）：np.abs(field) * 圆盘 mask -> jet [0, 0.8]，无色条。

用法：uv run python scripts/exp/topology_series.py
输出：figures/hbsn/candidates/topology_candidates_{multiconn,disconnected}.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, nets, shapes
from hbsn_exp.safe import HBSTimeout

IMG = 256
PX = IMG / 3.0  # world [-1.5,1.5] -> 256px 缩放
OUT_DIR = "figures/hbsn/candidates"
T, BASE = 0.5, 0.7  # triangle(t, base)：t=0.5 等腰、顶点朝上
Y_SWEEP = 0.0  # (a) 方形扫掠高度（约质心高度）
Y_CUT = 0.1  # (b) 矩形切入高度
DEPTH = (0.25, 0.6)  # 部分遮挡深度比例（pos1、pos2 及镜像；>0.6 时 hbs 库
# 对右缘深切口数值饱和，见 classic.py 文档，已扫描验证 0.6 两侧全档稳定）
MARGIN = 0.05  # 遮挡物戳出边缘的余量（world 单位）
DENOMS = (4, 5, 6)  # (a) 方形边 = H/denom（旧图约 H/3，三档均更小）
DENOMS_CUT = (6, 7, 8)  # (b) 矩形厚 = W/denom（旧图约 W/5，三档均更小）
TIER_NAMES = ("small", "smaller", "smallest")
ROW_LABELS = ("Input", "HBSN (ours)", "Classic HBS")
TIMEOUT = 8.0

# pos3（hole/cut）不调用 classic 管线：输入契约是单一闭曲线，boundary_from_image
# 取 RETR_EXTERNAL 最大连通域会静默丢洞/丢小块（显示被篡改形状的场），直接 N/A。
# 正常位 classic 失败则该格标 FAIL。
MALFORMED = ("hole", "cut")


def tri_geometry(bound):
    """边界点 -> (H, W_base, w)；w(y) 为高度 y 处半宽（底边最低、顶点朝上）。"""
    y0, y1 = bound[:, 1].min(), bound[:, 1].max()
    half = bound[:, 0].max()
    return y1 - y0, 2 * half, lambda y: half * (y1 - y) / (y1 - y0)


def square_positions(w, s):
    """(a) 方形 5 位置 (x0,x1,y0,y1,kind)：1/2 浅深咬合、3 成洞、4/5 镜像。

    pos1 中心 x=-(e+s/4)（约 25% 边长在形内），pos2 中心 x=-(e-0.1*s)（约 60% 在
    形内），pos3 中心 (0, Y_SWEEP) 完全进入成洞；e = w(Y_SWEEP)；4/5 为 x 镜像。
    """
    e = w(Y_SWEEP)
    return [
        (
            -(e + s / 4) - s / 2,
            -(e + s / 4) + s / 2,
            Y_SWEEP - s / 2,
            Y_SWEEP + s / 2,
            "bite",
        ),
        (
            -(e - 0.1 * s) - s / 2,
            -(e - 0.1 * s) + s / 2,
            Y_SWEEP - s / 2,
            Y_SWEEP + s / 2,
            "bite",
        ),
        (-s / 2, s / 2, Y_SWEEP - s / 2, Y_SWEEP + s / 2, "hole"),
        (
            (e - 0.1 * s) - s / 2,
            (e - 0.1 * s) + s / 2,
            Y_SWEEP - s / 2,
            Y_SWEEP + s / 2,
            "bite",
        ),
        (
            (e + s / 4) - s / 2,
            (e + s / 4) + s / 2,
            Y_SWEEP - s / 2,
            Y_SWEEP + s / 2,
            "bite",
        ),
    ]


def rect_positions(w, t):
    """(b) 水平矩形 5 位置 (x0,x1,y0,y1,kind)：1/2 浅深切入、3 全宽切断、4/5 镜像。

    y 固定 [Y_CUT-t/2, Y_CUT+t/2]；部分位从左/右缘戳入（戳出 MARGIN），尖端
    深度 = Y_CUT 高度处全宽 x DEPTH 比例；pos3 全宽贯穿（两侧各戳出 MARGIN）。
    """
    e = w(Y_CUT)
    poke = w(Y_CUT - t / 2) + MARGIN
    tip1 = -e + DEPTH[0] * 2 * e
    tip2 = -e + DEPTH[1] * 2 * e
    return [
        (-poke, tip1, Y_CUT - t / 2, Y_CUT + t / 2, "notch"),
        (-poke, tip2, Y_CUT - t / 2, Y_CUT + t / 2, "notch"),
        (-poke, poke, Y_CUT - t / 2, Y_CUT + t / 2, "cut"),
        (-tip2, poke, Y_CUT - t / 2, Y_CUT + t / 2, "notch"),
        (-tip1, poke, Y_CUT - t / 2, Y_CUT + t / 2, "notch"),
    ]


def rect_mask(x0, x1, y0, y1):
    """像素中心世界坐标的轴对齐矩形 mask（256 网格，row=y/col=x）。"""
    v = (np.arange(IMG) + 0.5) / PX - 1.5
    xv = v[None, :]
    yv = -v[:, None]  # row 向下 -> y 减小
    return (yv >= y0) & (yv <= y1) & (xv >= x0) & (xv <= x1)


def topo_check(gray, kind):
    """拓扑 assert：连通域数 + 洞数符合遮挡类型（防静默错图）。

    bite/notch：1 连通域、0 洞；hole：1 连通域且 >=1 洞；cut：>=2 连通域。
    """
    import cv2

    binary = (gray > 127).astype(np.uint8)
    n_cc = cv2.connectedComponents(binary)[0] - 1
    _, hier = cv2.findContours(
        binary * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    holes = int((hier[0][:, 3] >= 0).sum())
    if kind == "cut":
        assert n_cc >= 2, f"cut 未切断（{n_cc} 域）"
    else:
        assert n_cc == 1, f"形状碎裂（{kind}）：{n_cc} 域"
        if kind == "hole":
            assert holes >= 1, "hole 未成洞"
        else:
            assert holes == 0, f"{kind} 意外成洞（{holes}）"
    return n_cc, holes


def rms(cf, hf, mask):
    """classic-HBSN 场 RMS（圆盘内，HBS 论文式 7.1 度量）。"""
    return float(np.sqrt(np.sum(np.abs(cf - hf)[mask] ** 2) / mask.sum()))


def render_cell(base, net, disk_mask, rect):
    """单格：遮挡 -> 拓扑自检 -> 推理。返回 (gray, hf, cf|None, kind)。

    cf=None 时：畸形位（hole/cut）-> N/A；正常位 classic 失败 -> FAIL。
    """
    x0, x1, y0, y1, kind = rect
    gray = base.copy()
    gray[rect_mask(x0, x1, y0, y1)] = 0
    topo_check(gray, kind)
    hf = np.abs(nets.field_to_complex(nets.infer(net, gray))) * disk_mask
    if kind in MALFORMED:
        cf = None
    else:
        try:
            cf = np.abs(classic.classic_pixel_from_img(gray, timeout=TIMEOUT))
            cf = cf * disk_mask
            if (cf[disk_mask] > 0.9).mean() > 0.5:
                # hbs 库数值退化：|μ|≈1 处处饱和（无异常/NaN，静默坏场）——
                # 扫描实测好形状 frac(|μ|>0.9)<=0.11，饱和格 0.997，阈值 0.5 干净分离。
                raise RuntimeError("classic 场饱和")
        except (RuntimeError, ValueError, HBSTimeout):
            cf = None
    return gray, hf, cf, kind


def draw_figure(cells, tier_texts, title, out):
    """9x5 长图：3 档 x 3 行（Input/HBSN/Classic）x 5 遮挡位。"""
    fig, axs = plt.subplots(9, 5, figsize=(6.8, 11.6))
    fig.subplots_adjust(
        left=0.135,
        right=0.99,
        top=0.955,
        bottom=0.005,
        wspace=0.03,
        hspace=0.06,
    )
    for r in range(9):
        tier, row = divmod(r, 3)
        for c in range(5):
            ax = axs[r, c]
            ax.axis("off")
            gray, hf, cf, kind = cells[tier][c]
            if row == 0:
                ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            elif row == 1:
                ax.imshow(hf, cmap="jet", vmin=0, vmax=0.8)
            elif cf is not None:
                ax.imshow(cf, cmap="jet", vmin=0, vmax=0.8)
            else:
                ax.imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=1)
                text = "N/A" if kind in MALFORMED else "FAIL"
                ax.text(
                    0.5,
                    0.5,
                    text,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=13,
                    color="w",
                )
    for r in range(9):
        pos = axs[r, 0].get_position()
        fig.text(
            0.055,
            (pos.y0 + pos.y1) / 2,
            ROW_LABELS[r % 3],
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
        )
    for tier in range(3):
        y0 = axs[3 * tier, 0].get_position().y0
        y1 = axs[3 * tier + 2, 0].get_position().y1
        fig.text(
            0.012,
            (y0 + y1) / 2,
            tier_texts[tier],
            ha="center",
            va="center",
            rotation=90,
            fontsize=10.5,
        )
    fig.text(0.5, 0.985, title, ha="center", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    bound = shapes.triangle(T, base=BASE)
    h, wbase, w = tri_geometry(bound)
    base = shapes.shape_to_image(bound)[..., 0]
    net = nets.build_net(nets.BEST_CKPT)
    _, disk_mask = grid.get_ghbs_grid()
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = (
        ("multiconn", DENOMS, "square", "Multi-connected (square occluder)"),
        ("disconnected", DENOMS_CUT, "rect", "Disconnected (rect occluder)"),
    )
    for name, denoms, occ, title in groups:
        print(f"\n== {name} ==")
        cells, tier_texts = [], []
        for tier, denom in zip(TIER_NAMES, denoms, strict=True):
            if occ == "square":
                s = h / denom
                rects = square_positions(w, s)
                tier_texts.append(f"{tier}: side {s * PX:.0f} px (H/{denom})")
                print(f"  [{tier}] side {s:.4f} world / {s * PX:.1f} px")
            else:
                t = wbase / denom
                rects = rect_positions(w, t)
                length = 2 * (w(Y_CUT - t / 2) + MARGIN)
                tier_texts.append(
                    f"{tier}: cut {t * PX:.0f}x{length * PX:.0f} px (W/{denom})"
                )
                print(
                    f"  [{tier}] thick {t:.4f} world / {t * PX:.1f} px,"
                    f" len {length:.4f} world / {length * PX:.1f} px"
                )
            row_cells = []
            for k, rect in enumerate(rects, 1):
                cell = render_cell(base, net, disk_mask, rect)
                _, hf, cf, kind = cell
                if kind in MALFORMED:
                    msg = "N/A"
                elif cf is not None:
                    msg = f"{rms(cf, hf, disk_mask):.4f}"
                else:
                    msg = "FAIL"
                print(f"    pos{k} {kind}: classic-HBSN RMS = {msg}")
                row_cells.append(cell)
            cells.append(row_cells)
        out = os.path.join(OUT_DIR, f"topology_candidates_{name}.png")
        draw_figure(cells, tier_texts, title, out)
        print(f"  output: {out}")
        for tier_i, denom in enumerate(denoms):
            if occ == "square":
                s = h / denom
                info = f"side={s:.4f} world / {s * PX:.1f} px"
                centers = [
                    (x0 + x1) / 2 for x0, x1, *_ in square_positions(w, s)
                ]
            else:
                t = wbase / denom
                info = (
                    f"thick={t:.4f} world / {t * PX:.1f} px,"
                    f" len={2 * (w(Y_CUT - t / 2) + MARGIN):.4f} world"
                )
                centers = [(x0 + x1) / 2 for x0, x1, *_ in rect_positions(w, t)]
            ys = Y_SWEEP if occ == "square" else Y_CUT
            px = [
                (round((cx + 1.5) * PX, 1), round((1.5 - ys) * PX, 1))
                for cx in centers
            ]
            print(f"  centers[{TIER_NAMES[tier_i]}] {info}")
            print(f"    world x: {[round(c, 3) for c in centers]}, y={ys}")
            print(f"    px (x,y): {px}")


if __name__ == "__main__":
    main()
