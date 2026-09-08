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

渲染帧注意（重要）：shape_to_image 对边界做 max-模长重归一（k=0.85/max|z|，
本三角形 k=0.9029）+ mpl 默认轴盒 [0.125,0.9]x[0.11,0.88]（66.1/65.7 px/world，
中心 (131,129)）。本脚本的遮挡 mask 用全画布 85.33px/world 帧——mask 帧与
渲染帧差 ~1.43x，脚本内 H/6、W/8 等比例是 mask 帧设计量；论文引用尺寸一律
以渲染 px 为准（实测：三角形 100x71px、方形 mask 17.3px、切口厚 18.1px）。

用法：uv run python scripts/exp/topology_series.py [--dense | --fine | --final]
--fine：H/6 下 15 均匀细化位穿过洞过渡区（实测拓扑接管，洞态 N/A）。
--final：不连通组定稿面板（用户选 W/8 + 密集步 2/5/7/8/12）。
输出：figures/hbsn/candidates/topology_candidates_{name}[_dense].png
"""

import argparse
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


N_SIDE = 6  # 密集扫掠单侧档数（6 深档 + 畸形 + 6 镜像 = 13 位，畸形居中）
# 密集模式参数（用户定：档位集中到真正切进本体的深档，浅档近乎完整无意义）
DENOMS_CUT_DENSE = (8, 10, 12)  # (b) 密集矩形厚 = W/denom（18/14/12px，更窄）
D_LO, D_HI = 0.3, 0.65  # (b) 切入深度梯子（0.7+ 意外切断/饱和，扫描定）
F_LO, F_HI = 0.3, 0.9  # (a) 咬合深度梯子（0.9 两侧三档全部稳定，扫描定）


def square_sweep(w, s, f_lo, f_hi):
    """(a) 密集扫掠 2*N_SIDE+1 位：方形咬合 f_lo->f_hi 六档、成洞、右侧镜像。

    深度比例 f = 占方形边长比例（中心高度处）。梯子集中深档（用户定：浅档
    近乎完整三角形无意义）。f_hi=0.9 已扫描验证两侧三档全部稳定。
    """
    e = w(Y_SWEEP)
    out = []
    for k in range(N_SIDE):
        cx = -e + s * (f_lo + (f_hi - f_lo) * k / (N_SIDE - 1) - 0.5)
        out.append(
            (cx - s / 2, cx + s / 2, Y_SWEEP - s / 2, Y_SWEEP + s / 2, "bite")
        )
    out.append((-s / 2, s / 2, Y_SWEEP - s / 2, Y_SWEEP + s / 2, "hole"))
    for k in reversed(range(N_SIDE)):
        cx = e - s * (f_lo + (f_hi - f_lo) * k / (N_SIDE - 1) - 0.5)
        out.append(
            (cx - s / 2, cx + s / 2, Y_SWEEP - s / 2, Y_SWEEP + s / 2, "bite")
        )
    return out


def rect_sweep(w, t, d_lo, d_hi):
    """(b) 密集扫掠 2*N_SIDE+1 位：切入 d_lo->d_hi 六档、全宽切断、右侧镜像。

    深度比例 d = 占切入高度处全宽比例。d_hi=0.65 为扫描稳定上限（0.7+ 意外
    切断或饱和）。畸形位（cut）居中第 N_SIDE+1 列。
    """
    e = w(Y_CUT)
    poke = w(Y_CUT - t / 2) + MARGIN
    y0, y1 = Y_CUT - t / 2, Y_CUT + t / 2
    out = []
    for k in range(N_SIDE):
        tip = -e + (d_lo + (d_hi - d_lo) * k / (N_SIDE - 1)) * 2 * e
        out.append((-poke, tip, y0, y1, "notch"))
    out.append((-poke, poke, y0, y1, "cut"))
    for k in reversed(range(N_SIDE)):
        tip = e - (d_lo + (d_hi - d_lo) * k / (N_SIDE - 1)) * 2 * e
        out.append((tip, poke, y0, y1, "notch"))
    return out


# 细化模式（用户定 2026-09-08）：H/6 尺寸，6-9 列过渡区完全均匀 15 档，
# 端点咬合占比 f=0.78（原密集图第 5/9 列状态），实测拓扑接管分类（洞态 N/A）。
N_FINE = 15
F_FINE = 0.78
# 用户定稿（2026-09-08）：不连通组 W/8，密集步 2/5/7/8/12。
FINAL_DISC = {"denom": 8, "steps": (2, 5, 7, 8, 12)}


def fine_sweep(w, s):
    """(a) 细化扫掠 N_FINE 个均匀位：cx 从 -x_end 到 +x_end 完全均匀。

    x_end = e - s*(F_FINE-0.5)，即端点为咬合占比 F_FINE 的深咬合位；
    中间穿过洞态（H/6 时洞区约 cx∈[-0.40,+0.40]，多数列为实测多连通）。
    """
    e = w(Y_SWEEP)
    x_end = e - s * (F_FINE - 0.5)
    out = []
    for k in range(N_FINE):
        cx = -x_end + 2 * x_end * k / (N_FINE - 1)
        out.append(
            (cx - s / 2, cx + s / 2, Y_SWEEP - s / 2, Y_SWEEP + s / 2, "bite")
        )
    return out


def rect_mask(x0, x1, y0, y1):
    """像素中心世界坐标的轴对齐矩形 mask（256 网格，row=y/col=x）。"""
    v = (np.arange(IMG) + 0.5) / PX - 1.5
    xv = v[None, :]
    yv = -v[:, None]  # row 向下 -> y 减小
    return (yv >= y0) & (yv <= y1) & (xv >= x0) & (xv <= x1)


def topo_measure(gray):
    """二值图 -> (连通域数, 洞数)（RETR_CCOMP 层级计洞）。"""
    import cv2

    binary = (gray > 127).astype(np.uint8)
    n_cc = cv2.connectedComponents(binary)[0] - 1
    _, hier = cv2.findContours(
        binary * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return n_cc, int((hier[0][:, 3] >= 0).sum())


def topo_check(gray, kind):
    """拓扑 assert：连通域/洞数符合预期类型（防静默错图）。"""
    n_cc, holes = topo_measure(gray)
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


def render_cell(base, net, disk_mask, rect, strict=True):
    """单格：遮挡 -> 拓扑检查 -> 推理。返回 (gray, hf, cf|None, kind)。

    strict=True（默认）按意图类型 assert；strict=False（细化模式）实测拓扑
    接管分类——实测成洞/碎裂即按畸形处理（classic -> N/A），不 raise。
    cf=None 时：畸形位 -> N/A；正常位 classic 失败/饱和 -> FAIL。
    """
    x0, x1, y0, y1, kind = rect
    gray = base.copy()
    gray[rect_mask(x0, x1, y0, y1)] = 0
    if strict:
        topo_check(gray, kind)
    else:
        n_cc, holes = topo_measure(gray)
        if holes >= 1 or n_cc >= 2:
            kind = "hole"
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


def draw_sheet(blocks, block_texts, title, out, col_numbers=False):
    """长图：n_blocks 组 x 3 行（Input/HBSN/Classic）x n_cols 遮挡位。"""
    n_blocks, n_cols = len(blocks), len(blocks[0])
    margin = 1.05  # 左侧标签区（英寸）
    fig_w = 0.95 * n_cols + margin + 0.1
    fig_h = 1.18 * 3 * n_blocks + 0.6
    fig, axs = plt.subplots(
        3 * n_blocks, n_cols, figsize=(fig_w, fig_h), squeeze=False
    )
    left = margin / fig_w
    fig.subplots_adjust(
        left=left, right=0.99, top=0.955, bottom=0.005, wspace=0.03, hspace=0.06
    )
    for r in range(3 * n_blocks):
        block, row = divmod(r, 3)
        for c in range(n_cols):
            ax = axs[r, c]
            ax.axis("off")
            gray, hf, cf, kind = blocks[block][c]
            if row == 0:
                if col_numbers:
                    ax.set_title(str(c + 1), fontsize=8)
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
    for r in range(3 * n_blocks):
        pos = axs[r, 0].get_position()
        fig.text(
            0.62 * left,
            (pos.y0 + pos.y1) / 2,
            ROW_LABELS[r % 3],
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
        )
    for b in range(n_blocks):
        y0 = axs[3 * b, 0].get_position().y0
        y1 = axs[3 * b + 2, 0].get_position().y1
        fig.text(
            0.2 * left,
            (y0 + y1) / 2,
            block_texts[b],
            ha="center",
            va="center",
            rotation=90,
            fontsize=10.5,
        )
    fig.text(0.5, 0.985, title, ha="center", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(dense=False, fine=False, final=False):
    bound = shapes.triangle(T, base=BASE)
    h, wbase, w = tri_geometry(bound)
    base = shapes.shape_to_image(bound)[..., 0]
    net = nets.build_net(nets.BEST_CKPT)
    _, disk_mask = grid.get_ghbs_grid()
    os.makedirs(OUT_DIR, exist_ok=True)
    if fine:
        run_fine(base, net, disk_mask, w, h)
    elif final:
        run_final_panel(base, net, disk_mask, w, wbase)
    else:
        run_candidates(base, net, disk_mask, w, h, wbase, dense)


def run_candidates(base, net, disk_mask, w, h, wbase, dense):
    groups = (
        (
            "multiconn",
            DENOMS,
            DENOMS,
            "square",
            "Multi-connected (square occluder)",
        ),
        (
            "disconnected",
            DENOMS_CUT,
            DENOMS_CUT_DENSE,
            "rect",
            "Disconnected (rect occluder)",
        ),
    )
    for name, denoms, denoms_dense, occ, title in groups:
        print(f"\n== {name}{' [dense]' if dense else ''} ==")
        blocks, block_texts = [], []
        used = denoms_dense if dense else denoms
        for tier, denom in zip(TIER_NAMES, used, strict=True):
            if occ == "square":
                s = h / denom
                rects = (
                    square_sweep(w, s, F_LO, F_HI)
                    if dense
                    else square_positions(w, s)
                )
                block_texts.append(f"{tier}: side {s * PX:.0f} px (H/{denom})")
                print(f"  [{tier}] side {s:.4f} world / {s * PX:.1f} px")
            else:
                t = wbase / denom
                rects = (
                    rect_sweep(w, t, D_LO, D_HI)
                    if dense
                    else rect_positions(w, t)
                )
                length = 2 * (w(Y_CUT - t / 2) + MARGIN)
                block_texts.append(
                    f"{tier}: cut {t * PX:.0f}x{length * PX:.0f} px (W/{denom})"
                )
                print(
                    f"  [{tier}] thick {t:.4f} world / {t * PX:.1f} px,"
                    f" len {length:.4f} world / {length * PX:.1f} px"
                )
            block = []
            for k, (x0, x1, y0, y1, kind) in enumerate(rects, 1):
                cell = render_cell(base, net, disk_mask, (x0, x1, y0, y1, kind))
                _, hf, cf, kind = cell
                if kind in MALFORMED:
                    msg = "N/A"
                elif cf is not None:
                    msg = f"{rms(cf, hf, disk_mask):.4f}"
                else:
                    msg = "FAIL"
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                print(
                    f"    step{k:02d} {kind}: RMS={msg}"
                    f"  center=({cx:+.3f},{cy:+.3f})w"
                    f"=({(cx + 1.5) * PX:.1f},{(1.5 - cy) * PX:.1f})px"
                )
                block.append(cell)
            blocks.append(block)
        suffix = "_dense" if dense else ""
        out = os.path.join(OUT_DIR, f"topology_candidates_{name}{suffix}.png")
        draw_sheet(blocks, block_texts, title, out, col_numbers=dense)
        print(f"  output: {out}")


def run_fine(base, net, disk_mask, w, h):
    """细化扫掠：H/6、15 均匀位穿过洞过渡区，实测拓扑接管分类。"""
    s = h / 6
    rects = fine_sweep(w, s)
    block = []
    for k, (x0, x1, y0, y1, kind) in enumerate(rects, 1):
        cell = render_cell(
            base, net, disk_mask, (x0, x1, y0, y1, kind), strict=False
        )
        gray, hf, cf, kind = cell
        n_cc, holes = topo_measure(gray)
        msg = (
            "N/A"
            if kind in MALFORMED
            else (f"{rms(cf, hf, disk_mask):.4f}" if cf is not None else "FAIL")
        )
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        print(
            f"    step{k:02d} {kind} [{n_cc}cc/{holes}h]: RMS={msg}"
            f"  center=({cx:+.3f},{cy:+.3f})w"
            f"=({(cx + 1.5) * PX:.1f},{(1.5 - cy) * PX:.1f})px"
        )
        block.append(cell)
    out = os.path.join(OUT_DIR, "topology_candidates_multiconn_fine.png")
    draw_sheet(
        [block],
        [f"H/6: {N_FINE} uniform steps (s={s * PX:.0f} px)"],
        "Multi-connected (fine sweep across hole transition)",
        out,
        col_numbers=True,
    )
    print(f"  output: {out}")


def run_final_panel(base, net, disk_mask, w, wbase):
    """不连通组定稿面板（用户选：W/8，密集步 2/5/7/8/12）。"""
    denom = FINAL_DISC["denom"]
    t = wbase / denom
    sweep = rect_sweep(w, t, D_LO, D_HI)
    block = []
    for k in FINAL_DISC["steps"]:
        cell = render_cell(base, net, disk_mask, sweep[k - 1])
        _, hf, cf, kind = cell
        msg = (
            "N/A"
            if kind in MALFORMED
            else (f"{rms(cf, hf, disk_mask):.4f}" if cf is not None else "FAIL")
        )
        print(f"    final col {k}: RMS={msg}")
        block.append(cell)
    out = os.path.join(OUT_DIR, "topology_results_disc_preview.png")
    draw_sheet(
        [block],
        [f"W/{denom}: cut {t * PX:.0f} px, dense steps {FINAL_DISC['steps']}"],
        "Disconnected (rect occluder)",
        out,
    )
    print(f"  output: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true", help="13 档密集扫掠候选图")
    ap.add_argument(
        "--fine", action="store_true", help="15 均匀细化扫掠（H/6）"
    )
    ap.add_argument("--final", action="store_true", help="不连通组定稿面板")
    a = ap.parse_args()
    main(dense=a.dense, fine=a.fine, final=a.final)
