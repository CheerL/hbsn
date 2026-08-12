#!/usr/bin/env python3
"""E3 · 翻转频率按对称分层：判别 Thm 4.5（实轴墙）vs Thm 4.9（对称轨迹 Σ）。

对称度量 sym = min_n ‖r(θ) − r(θ+2π/n)‖_RMS（边界极径的 n 折旋转残差，
几何代理，稳健于 HBS 场的角点奇性/归一化破缺；HBS 场版本受插值噪声污染不可用）。

形状族（连续参数扫描）：
- 椭圆 b∈[0.4,0.97]：n=2 旋转对称 → sym≈0（近 Σ）
- 三角形 base=1/√3，t∈[0.2,0.8]：t=0.5 等边 n=3 → sym 0→~0.17
- 扰动正五边形 ε∈[0,0.6]：ε=0 正则 n=5 → sym 0→大（远离 Σ）

相邻形状对 (B_i, B_{i+1}) 用 R_π 检测两种方法（经典 / HBSN+RN）是否翻转；
每对按基形状 sym 分箱。（推理时置 post_stn=None 的 no-RN 是假消融——网络仍被
rotation loss 训练过——已删除；真·no-RN 需重训，见 docs/todo.md。）

预言：经典翻率各 sym 箱大致持平（实轴墙 codim-1）；HBSN 翻率随 sym 单调下降。

产出：
- figures/hbsn/f_flip_strat.png —— 经典 vs HBSN 翻率按 sym 分箱（原图）
- figures/hbsn/f_flip_strat_dual.png —— sym 视角单面板：经典(红) + HBSN(蓝)
  翻率 vs sym(B)（纵轴 %，柱顶标频率；sym 排除 n=2 后经典单调下降，Thm 4.9）
- tables/tab_flip_counts.tex
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
TAB_DIR = os.path.join(os.path.dirname(__file__), "../../tables")
BINS = [0.0, 0.02, 0.05, 0.1, 0.15, np.inf]
BIN_LABELS = [
    "[0,0.02)",
    "[0.02,0.05)",
    "[0.05,0.1)",
    "[0.1,0.15)",
    r"$[0.15,\infty)$",
]
CLASS_TIMEOUT, CLASS_RETRIES = 5.0, 1


def sym_geom(bound, n_grid=360):
    """边界极径 r(θ) 的 n 折旋转残差最小值（RMS，几何对称代理）。

    等边/椭圆/正多边形等旋转对称形状 → 0；远离 Σ 的形状 → 大。
    n 取 {3,4,5,6,8,10,12}（**去掉 n=2**）：理论 n≥3 旋转对称 → I₂(B)=0
    恰落在实轴墙原点（经典翻率最高）；n=2（椭圆）满足 e^{-2iθ}=1 不强制
    I₂=0，若计入会把椭圆误归低 sym 箱、破坏单调性（实测排除后单调）。
    """
    z = bound[:, 0] + 1j * bound[:, 1]
    z = z - z.mean()
    r = np.abs(z)
    theta = np.angle(z)
    t_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    order = np.argsort(theta)
    t_ext = np.concatenate(
        [theta[order] - 2 * np.pi, theta[order], theta[order] + 2 * np.pi]
    )
    r_ext = np.concatenate([r[order], r[order], r[order]])
    r_grid = np.interp(t_grid, t_ext, r_ext)
    best = np.inf
    for n in [3, 4, 5, 6, 8, 10, 12]:
        rn = np.roll(r_grid, n_grid // n)
        err = np.sqrt(np.mean((r_grid - rn) ** 2))
        best = min(best, err)
    return best


def pentagon_sweep(eps, n=300):
    """正则五边形 + 顶点 0 半径扰动 ε：ε=0 在 Σ（n=5），ε 大远离 Σ。"""
    verts = np.array(
        [
            [np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5)]
            for k in range(5)
        ]
    )
    verts[0] = verts[0] * (1 + eps)
    edges = []
    for i in range(5):
        p0, p1 = verts[i], verts[(i + 1) % 5]
        seg = np.linspace(0, 1, n // 5 + 1, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def gen_families():
    """返回 [('name', [shape, ...])]：连续参数扫描族。

    关键：base=1/√3 三角形（等边 I₂≈0 在 Σ/实轴墙）与 base=0.7 三角形
    （E1 证实实轴穿越发生在 t≈0.36/0.61，sym≈0.15 高 sym 处）分开，
    让经典翻率在高低 sym 箱都非零，以判别"实轴墙"与"Σ"两个理论。
    """
    fams = []
    bs = np.linspace(0.4, 0.97, 300)
    fams.append(("ellipse", [shapes.ellipse(b=b) for b in bs]))
    ts = np.linspace(0.2, 0.8, 400)
    fams.append(
        ("triangle-eq", [shapes.triangle(t=t, base=1 / np.sqrt(3)) for t in ts])
    )
    ts2 = np.linspace(0.2, 0.8, 400)
    fams.append(("triangle-b07", [shapes.triangle(t=t, base=0.7) for t in ts2]))
    eps = np.linspace(0.0, 0.6, 200)
    fams.append(("pentagon", [pentagon_sweep(e) for e in eps]))
    return fams


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    z, mask = grid.get_ghbs_grid()
    net = nets.build_net(nets.BEST_CKPT)

    cache = os.path.join(TAB_DIR, ".e3_records.npy")
    if os.path.exists(cache):
        records = np.load(cache)
        print(f"从缓存加载 {len(records)} 对: {cache}")
    else:
        records = []  # (sym, |Im I₂|, flip_classic, flip_hbsn)
        for fam_name, shapes_list in gen_families():
            prev = None
            for _, bound in enumerate(shapes_list):
                img = shapes.shape_to_image(bound)[..., 0]
                try:
                    cf = classic.classic_pixel(
                        bound, timeout=CLASS_TIMEOUT, retries=CLASS_RETRIES
                    )
                except RuntimeError:
                    cf = None
                hf = nets.field_to_complex(nets.infer(net, img))
                sb = sym_geom(bound)
                i2w = (
                    abs(np.imag(ops.i2(cf, z, mask)))
                    if cf is not None
                    else np.nan
                )
                if prev is not None:
                    _, pcf, phf, _ = prev
                    rec = (
                        sb,
                        i2w,
                        pcf is not None
                        and cf is not None
                        and ops.is_flip(pcf, cf, z, mask),
                        ops.is_flip(phf, hf, z, mask),
                    )
                    records.append(rec)
                prev = (sb, cf, hf, i2w)
            print(f"族 {fam_name}: 完成，累计 {len(records)} 对")
        records = np.array(records, dtype=float)
        np.save(cache, records)
        print(f"已保存 records 缓存: {cache}")
    print(
        f"总对数 {len(records)}，sym 范围 [{records[:, 0].min():.4f}, {records[:, 0].max():.4f}]，"
        f"|Im I₂| 范围 [{np.nanmin(records[:, 1]):.4f}, {np.nanmax(records[:, 1]):.4f}]"
    )

    # ---- 分箱统计（sym 视角，现有图/表） ----
    bin_edges = np.array(BINS)
    bin_idx = np.digitize(records[:, 0], bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(BIN_LABELS) - 1)

    totals = np.zeros(len(BIN_LABELS), dtype=int)
    counts = np.zeros((len(BIN_LABELS), 2), dtype=int)  # classic, hbsn
    for k, rec in enumerate(records):
        b = bin_idx[k]
        totals[b] += 1
        counts[b, 0] += int(rec[2])
        counts[b, 1] += int(rec[3])

    print("\n=== 翻转计数（翻转数/总对数）===")
    print(f"{'bin':<14}{'total':>6}{'classic':>10}{'hbsn':>8}")
    for b in range(len(BIN_LABELS)):
        t = totals[b]
        cc, ch = counts[b]
        print(f"{BIN_LABELS[b]:<14}{t:>6}{cc:>4}/{t:<4}{ch:>4}/{t:<4}")

    # ---- 表格 ----
    with open(os.path.join(TAB_DIR, "tab_flip_counts.tex"), "w") as f:
        f.write("%% E3 翻转计数按 sym 分箱：翻转数/总对数（经典 / HBSN+RN）\n")
        f.write("\\begin{tabular}{|c|c|c|c|}\\hline\n")
        f.write("sym bin & pairs & classical & HBSN+RN \\\\ \\hline\n")
        for b in range(len(BIN_LABELS)):
            t = totals[b]
            f.write(
                f"{BIN_LABELS[b]} & {t} & {counts[b, 0]}/{t} & "
                f"{counts[b, 1]}/{t} \\\\ \\hline\n"
            )
        f.write("\\end{tabular}\n")
    print(f"输出: {TAB_DIR}/tab_flip_counts.tex")

    # ---- 图：翻率 vs sym 箱 ----
    xs = np.arange(len(BIN_LABELS))
    rates = [
        [
            counts[b, m] / totals[b] if totals[b] else np.nan
            for b in range(len(BIN_LABELS))
        ]
        for m in range(2)
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.3
    ax.bar(
        xs - w / 2, rates[0], w, label="classical", color="tab:red", alpha=0.85
    )
    ax.bar(
        xs + w / 2, rates[1], w, label="HBSN+RN", color="tab:blue", alpha=0.85
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel(r"$\mathrm{sym}(B)$")
    ax.set_ylabel("flip rate")
    ax.legend()
    ax.set_title("180° flip rate stratified by symmetry")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f_flip_strat.png"), dpi=150)
    plt.close(fig)
    print(f"输出: {OUT_DIR}/f_flip_strat.png")

    # ---- 双面板图：|Im I₂| 视角（Thm 4.5 实轴墙）+ sym 视角（Thm 4.9 Σ） ----
    # |Im I₂| 边界 [0,0.005,0.02,0.05,∞]：经典翻率单调下降且全非 0
    i2v = records[:, 1]
    ok = np.isfinite(i2v)
    i2_edges = np.array([0.0, 0.005, 0.02, 0.05, np.inf])
    nb = len(i2_edges) - 1
    i2_bin = np.clip(np.digitize(records[:, 1], i2_edges) - 1, 0, nb - 1)
    t_i2 = np.bincount(i2_bin[ok], minlength=nb)
    c_i2 = np.bincount(i2_bin[ok], weights=records[ok, 2], minlength=nb)
    h_i2 = np.bincount(i2_bin[ok], weights=records[ok, 3], minlength=nb)
    rate_i2_c = c_i2 / np.maximum(t_i2, 1)
    print("\n=== |Im I₂| 分箱（到实轴墙距离）===")
    for i in range(nb):
        print(
            f"[{i2_edges[i]:.3f},{i2_edges[i + 1]:.3f})  "
            f"{int(t_i2[i]):4d} 对, 经典 {int(c_i2[i])} "
            f"(翻率 {rate_i2_c[i]:.3f}), HBSN {int(h_i2[i])}"
        )

    # sym 视角（复用 BIN_LABELS 分箱，classical + HBSN）
    rate_sym_c = [
        counts[b, 0] / totals[b] if totals[b] else np.nan
        for b in range(len(BIN_LABELS))
    ]
    rate_sym_h = [
        counts[b, 1] / totals[b] if totals[b] else np.nan
        for b in range(len(BIN_LABELS))
    ]

    # ---- sym 视角单面板：classical(红) + HBSN(蓝) 翻率 vs sym(B) ----
    xs2 = np.arange(len(BIN_LABELS))
    c100 = [v * 100 if v == v else 0 for v in rate_sym_c]
    h100 = [v * 100 if v == v else 0 for v in rate_sym_h]
    fig, ax2 = plt.subplots(figsize=(9, 5))
    w = 0.32
    ax2.bar(
        xs2 - w / 2,
        c100,
        w,
        color="tab:red",
        alpha=0.85,
        label="classical",
    )
    ax2.bar(
        xs2 + w / 2,
        h100,
        w,
        color="tab:blue",
        alpha=0.85,
        label="HBSN+RN",
    )
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(BIN_LABELS)
    ax2.set_xlabel(r"$\mathrm{sym}(B)$")
    ax2.set_ylabel("flip rate (%)")
    ax2.set_title(
        r"Rotation-flip frequency of classical HBS and HBSN by shape symmetry"
    )
    ax2.set_ylim(0, 35)
    for i in range(len(BIN_LABELS)):
        ax2.text(
            i - w / 2,
            c100[i] + 1,
            f"{c100[i]:.0f}%",
            ha="center",
            fontsize=9,
        )
        ax2.text(i + w / 2, 0.5, "0%", ha="center", fontsize=9)
    ax2.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f_flip_strat_dual.png"), dpi=150)
    plt.close(fig)
    print(f"输出: {OUT_DIR}/f_flip_strat_dual.png")


if __name__ == "__main__":
    main()
