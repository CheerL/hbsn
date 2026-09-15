#!/usr/bin/env python3
"""【已弃用】E1 · 连续形变（半圆扫掠族）——论文改用 osc 族，见 e1_deform_osc.py。

半圆扫掠族：经典 HBS 沿路径翻 180°，HBSN 连续。

半圆扫掠族：底边 A=(-base,0), B=(base,0) 固定，上顶点 C(t)=(cos πt, sin πt)
沿上半单位圆扫过。参数用顶点极角 θ=πt（度数显示，纯 θ 不出现 t）。
- 窗口 t∈[0.616,0.624] → θ∈[110.88°,112.32°]，翻转 θ*=111.10°（Im I₂ 穿越实轴）。
- 经典归一化（N2: Im I₂>0）在 I₂ 穿越实轴处对域旋转 π（R_πB=B(-z)，非全局负号）
  ——检测用 ops.is_flip（已修 R_π）。

单 figure + GridSpec：左 (a) d(B_θ, B_θ*)-vs-θ 曲线（经典阶跃/HBSN 直线，xlim 聚焦翻转）；
右 (b) 3 行×4 列快照（行=输入三角形（黑底白三角形 + C 顶点坐标）/经典/HBSN，
列=θ*-2ε/θ*-ε/θ*+ε/θ*+2ε），每行开头竖排文字标签（无 bbox）。
产出 figures/hbsn/f_deform_semicircle.png（不再被论文引用）。
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, nets, ops, shapes

OUT_DIR = os.path.join(os.path.dirname(__file__), "../../figures/hbsn")
BASE = 0.7
T_LO, T_HI = 0.616, 0.624
N_T = 41
EPS = 0.0006  # 快照偏移（4 快照点 θ*±ε/±2ε 全部落在 HBSN 稳定区）
FONTSIZE = 13
DEG = 180.0  # θ(rad) = πt → θ(deg) = 180·t


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z, mask = grid.get_ghbs_grid()
    net = nets.build_net(nets.BEST_CKPT)

    ts = np.linspace(T_LO, T_HI, N_T)
    ths = ts * DEG  # 度数
    classic_fields, hbsn_fields, i2s = [], [], []
    for t in ts:
        bound = shapes.triangle(t=t, base=BASE)
        try:
            cf = classic.classic_pixel(bound)
        except RuntimeError:
            cf = None
        img = shapes.shape_to_image(bound)[..., 0]
        hf = nets.field_to_complex(nets.infer(net, img))
        classic_fields.append(cf)
        hbsn_fields.append(hf)
        i2s.append(ops.i2(cf, z, mask) if cf is not None else np.nan)

    # θ* = Im I₂ 变号（限有限值，排除失败段 NaN 伪穿越）
    i2s = np.array(i2s)
    im = np.sign(i2s.imag)
    valid = ~np.isnan(im)
    crossings = np.where(valid[:-1] & valid[1:] & (np.diff(im) != 0))[0]
    t_star = (
        ts[crossings[0]]
        if len(crossings)
        else ts[np.nanargmin(np.abs(i2s.imag))]
    )
    th_star = t_star * DEG
    i_star = int(np.argmin(np.abs(ts - t_star)))

    # d(B_θ, B_θ*)：每个场与翻转点场的 L2 距离
    # （经典：θ<θ* 侧差 180° 翻转 → 大值阶跃，θ>θ* 侧接近 → ~0；HBSN：全场一致 → 直线）
    d_classic = [
        np.nan
        if classic_fields[i] is None or classic_fields[i_star] is None
        else ops.d2(classic_fields[i], classic_fields[i_star], z, mask)
        for i in range(N_T)
    ]
    d_hbsn = [
        ops.d2(hbsn_fields[i], hbsn_fields[i_star], z, mask) for i in range(N_T)
    ]

    # R_π 翻转计数（经典 vs HBSN）
    def count_flips(fields):
        n = 0
        for i in range(len(fields) - 1):
            if fields[i] is None or fields[i + 1] is None:
                continue
            if ops.is_flip(fields[i], fields[i + 1], z, mask):
                n += 1
        return n

    flip_classic = count_flips(classic_fields)
    flip_hbsn = count_flips(hbsn_fields)

    # ---- 单 figure：GridSpec(1,2)，左 a 曲线 / 右 b 快照 ----
    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.2)
    ax_a = fig.add_subplot(gs[0])
    gs_b = gs[1].subgridspec(3, 4, hspace=0.15, wspace=0.08)
    axs = np.array(
        [[fig.add_subplot(gs_b[r, c]) for c in range(4)] for r in range(3)]
    )

    # ---- (a) d(B_θ, B_θ*)-vs-θ 曲线（xlim 聚焦翻转） ----
    ax_a.plot(
        ths,
        d_classic,
        "r-o",
        ms=3,
        label=f"classical (flips={flip_classic})",
    )
    ax_a.plot(ths, d_hbsn, "b-s", ms=3, label=f"HBSN (flips={flip_hbsn})")
    ax_a.axvline(th_star, color="k", ls="--", lw=1)
    # θ* 标注：紧贴虚线右侧、红色折线上方（图例已缩小避开虚线）
    ax_a.text(
        th_star + 0.01,
        np.nanmax(d_classic) * 1.18,
        rf"$\theta^*$={th_star:.2f}$^\circ$",
        ha="left",
        va="center",
        fontsize=FONTSIZE,
    )
    # x 轴：刻度直接带 °，xlabel 不带 [°]
    ax_a.set_xlabel(
        r"polar angle $\theta$ of apex $C=(\cos\theta,\sin\theta)$",
        fontsize=FONTSIZE,
    )
    ax_a.set_ylabel(r"$d(B_\theta, B_{\theta^*})$", fontsize=FONTSIZE)
    ax_a.tick_params(labelsize=FONTSIZE - 2)
    ax_a.set_xlim(110.95, 111.45)
    ax_a.set_xticks([111.0, 111.2, 111.4])
    ax_a.set_xticklabels(["111.0°", "111.2°", "111.4°"])
    ax_a.set_ylim(0, np.nanmax(d_classic) * 1.3)  # 顶部留空给 θ* 标注
    # 图例放数据区上方（ncol=2 横排），不占数据区、不遮虚线/θ* 标注
    ax_a.legend(
        fontsize=FONTSIZE - 2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )

    # ---- (b) 3 行×4 列快照 ----
    snap_ts = [
        t_star - 2 * EPS,
        t_star - EPS,
        t_star + EPS,
        t_star + 2 * EPS,
    ]
    row_labels = ["input shape", "classical HBS", "HBSN"]

    for c, tv in enumerate(snap_ts):
        thv = tv * DEG
        bound = shapes.triangle(t=tv, base=BASE)
        i = int(np.argmin(np.abs(ts - tv)))
        f = classic_fields[i]

        # 行 1：黑底白三角形 + 仅 C 顶点坐标（4 位小数）
        zz = bound[:, 0] + 1j * bound[:, 1]
        zz = zz - zz.mean()
        zz = zz / np.abs(zz).max() * 0.85
        ax0 = axs[0, c]
        ax0.set_facecolor("black")
        ax0.fill(zz.real, zz.imag, color="white", edgecolor="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\theta={thv:.2f}^\circ$", fontsize=FONTSIZE + 1)
        # 手动隐藏轴元素但保留 patch 背景（axis("off") 会隐藏 patch）
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_visible(False)
        # C 顶点标注（构造坐标，4 位小数；放 C 正上方，无箭头）
        cx, cy = np.cos(np.pi * tv), np.sin(np.pi * tv)
        ax0.text(
            zz[202].real,
            zz[202].imag + 0.12,
            f"C({cx:.4f},{cy:.4f})",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 4,
            color="white",
            fontweight="bold",
            clip_on=False,
        )

        # 行 2：经典 |HBS|
        axs[1, c].imshow(
            np.abs(f) * mask if f is not None else np.zeros_like(mask),
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        axs[1, c].axis("off")

        # 行 3：HBSN |HBS|
        axs[2, c].imshow(
            np.abs(hbsn_fields[i]) * mask,
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        axs[2, c].axis("off")

    # 每行开头竖排文字标签（紧贴图片左缘，无 bbox）
    for r, label in enumerate(row_labels):
        axs[r, 0].text(
            -0.08,
            0.5,
            label,
            transform=axs[r, 0].transAxes,
            va="center",
            ha="center",
            rotation=90,
            fontsize=FONTSIZE + 1,
        )

    # (a)/(b) 标题统一用 fig.text：各自居中于所在面板，同一 y 高度对齐
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_b0 = axs[0, 0].get_position()
    pos_b3 = axs[0, 3].get_position()
    title_y = 0.965
    fig.text(
        (pos_a.x0 + pos_a.x1) / 2,
        title_y,
        r"(a) Distance from flip-point $d(B_\theta, B_{\theta^*})$",
        ha="center",
        fontsize=FONTSIZE + 1,
    )
    fig.text(
        (pos_b0.x0 + pos_b3.x1) / 2,
        title_y,
        rf"(b) HBS around flip at $\theta^*={th_star:.2f}^\circ$",
        ha="center",
        fontsize=FONTSIZE + 1,
    )

    fig.savefig(
        os.path.join(OUT_DIR, "f_deform_semicircle.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"θ* = {th_star:.2f}°  (t* = {t_star:.4f})  "
        f"Im I₂ 穿越 {len(crossings)} 次（窗口内）"
    )
    print(f"经典 R_π 翻转 {flip_classic} 次, HBSN 翻转 {flip_hbsn} 次")
    print(
        f"d(B_θ,B_θ*) classic: 左侧阶跃 {np.nanmax(d_classic):.4f} → 右侧 ~0, "
        f"hbsn max={np.nanmax(d_hbsn):.4f}"
    )
    print(f"输出: {OUT_DIR}/f_deform_semicircle.png (a+b GridSpec)")


if __name__ == "__main__":
    main()
