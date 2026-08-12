#!/usr/bin/env python3
"""旧架构 checkpoint 综合性能评估：场质量（对比经典）/ E3 翻转率 / E1 连续性。

对 runs/remote_no_rn/ 下已下载的 4 个旧架构 checkpoint（legacy 兼容加载）：
- 场质量：一组代表性形状上 HBSN 场与经典 HBS 场的 |场| 相关系数 + 相对误差
- E3 翻转率：E3 形状族（gen_families）相邻对 R_π 翻转，按 sym 分箱
- E1 连续性：半圆扫掠族（e1_deform 的族）相邻 d(B_t,B_{t+δ}) 的 d_max

用法：uv run python scripts/exp/eval_legacy_perf.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e3_flip_strat import BIN_LABELS, BINS, gen_families, sym_geom
from hbsn_exp import classic, grid, legacy, ops, shapes

MODELS = {
    "no_rn_stn1": "runs/remote_no_rn/no_rn_stn1_best993.pth",
    "no_rn_nostn": "runs/remote_no_rn/no_rn_nostn_best943.pth",
    "no_rn_nostn_stnloss": "runs/remote_no_rn/no_rn_nostn_stnloss_best975.pth",
    "stn3_big_aug": "runs/remote_no_rn/stn3_big_aug_best886.pth",
}
# E1 半圆扫掠族（e1_deform 参数）
E1_T = np.linspace(0.616, 0.624, 41)
E1_BASE = 0.7


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    z, mask = grid.get_ghbs_grid()

    nets_old = {
        name: legacy.build_old(path, device=device)[0]
        for name, path in MODELS.items()
    }
    results = {name: {} for name in MODELS}

    # ---- 场质量：经典 HBS vs 各模型（代表性形状） ----
    print("\n===== 场质量（|场| 相关系数 / 相对误差 vs 经典 HBS）=====")
    t0 = time.time()
    # 代表性形状：椭圆/等边三角/普通三角/正五边形
    sample = [
        shapes.ellipse(b=0.6),
        shapes.ellipse(b=0.9),
        shapes.triangle(t=0.5, base=1 / np.sqrt(3)),
        shapes.triangle(t=0.6, base=0.7),
        shapes.triangle(t=0.3, base=0.7),
    ]
    corr_sum = {n: [] for n in MODELS}
    err_sum = {n: [] for n in MODELS}
    for bi, bound in enumerate(sample):
        try:
            cf = classic.classic_pixel(bound, timeout=5, retries=1)
            fc = cf * mask  # 经典场
        except RuntimeError:
            print(f"  形状{bi} 经典失败，跳过")
            continue
        img = shapes.shape_to_image(bound)[..., 0]
        for name, net in nets_old.items():
            hf = legacy.infer_to_ghbs(net, img, z, mask)
            ac, bc = abs(fc), abs(hf)
            r = np.corrcoef(ac.flatten(), bc.flatten())[0, 1]
            err = np.sqrt(np.mean((ac - bc) ** 2)) / (
                np.sqrt(np.mean(ac**2)) + 1e-8
            )
            corr_sum[name].append(r)
            err_sum[name].append(err)
    for name in MODELS:
        results[name]["field_corr"] = float(np.mean(corr_sum[name]))
        results[name]["field_rel_err"] = float(np.mean(err_sum[name]))
        print(
            f"  {name:22s} corr={results[name]['field_corr']:.3f} "
            f"rel_err={results[name]['field_rel_err']:.3f}"
        )
    print(f"  场质量耗时 {time.time() - t0:.0f}s")

    # ---- E3 翻转率（sym 分箱） ----
    print("\n===== E3 翻转率（sym 分箱，flips/pairs）=====")
    t0 = time.time()
    recs = {name: [] for name in MODELS}
    prev = dict.fromkeys(MODELS)
    for _fam_name, shapes_list in gen_families():
        for bound in shapes_list:
            img = shapes.shape_to_image(bound)[..., 0]
            sb = sym_geom(bound)
            for name, net in nets_old.items():
                hf = legacy.infer_to_ghbs(net, img, z, mask)
                if prev[name] is not None:
                    recs[name].append(
                        (sb, ops.is_flip(prev[name], hf, z, mask))
                    )
                prev[name] = hf
    edges = np.array(BINS)
    for name in MODELS:
        arr = np.array(recs[name], dtype=float)
        rates = []
        for b in range(len(BIN_LABELS)):
            m = (np.digitize(arr[:, 0], edges) - 1) == b
            t = m.sum()
            rates.append(arr[m, 1].sum() / t if t else 0.0)
        results[name]["e3_flip_rates"] = rates
        print(
            f"  {name:22s} rates(%)="
            + " ".join(f"{r * 100:.1f}" for r in rates)
        )
    print(f"  E3 耗时 {time.time() - t0:.0f}s")

    # ---- E1 连续性（半圆扫掠族 d_max） ----
    print("\n===== E1 连续性（半圆扫掠族 d(B_t,B_{t+δ}) max）=====")
    t0 = time.time()
    for name, net in nets_old.items():
        dmax = 0.0
        prev_hf = None
        for t in E1_T:
            bound = shapes.triangle(t=t, base=E1_BASE)
            img = shapes.shape_to_image(bound)[..., 0]
            hf = legacy.infer_to_ghbs(net, img, z, mask)
            if prev_hf is not None:
                d = ops.d2(prev_hf, hf, z, mask)
                dmax = max(dmax, d)
            prev_hf = hf
        results[name]["e1_dmax"] = float(dmax)
        print(f"  {name:22s} d_max={dmax:.3f}")
    print(f"  E1 耗时 {time.time() - t0:.0f}s")

    # ---- 汇总表 ----
    print("\n===== 汇总 =====")
    print(
        f"{'model':<22}{'corr':>7}{'rel_err':>9}{'E3_rate[0]':>12}{'E1_dmax':>10}"
    )
    for name in MODELS:
        r = results[name]
        e3 = r["e3_flip_rates"][0] * 100
        print(
            f"{name:<22}{r['field_corr']:>7.3f}{r['field_rel_err']:>9.3f}"
            f"{e3:>11.1f}%{r['e1_dmax']:>10.3f}"
        )


if __name__ == "__main__":
    main()
