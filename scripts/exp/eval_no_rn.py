#!/usr/bin/env python3
"""E3 · no-RN 对比：旧架构 no-RN checkpoint 的翻转频率（sym 分箱）。

用 legacy.build_old 加载服务器迁移来的旧架构 checkpoint（stn_mode 0/1/3），
对 E3 形状族（gen_families）相邻对检测 R_π 翻转，按 sym(B) 分箱，
与当前 HBSN+RN（BEST_CKPT）对比。

产出 stdout 对比表（不写图/表文件——是否回填 paper 由用户决定）。

用法：uv run python scripts/exp/eval_no_rn.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e3_flip_strat import BIN_LABELS, BINS, gen_families, sym_geom
from hbsn_exp import grid, legacy, nets, ops, shapes

NO_RN_CKPTS = {
    "no_rn_stn1": "runs/remote_no_rn/no_rn_stn1_best993.pth",
    "no_rn_nostn": "runs/remote_no_rn/no_rn_nostn_best943.pth",
    "no_rn_nostn_stnloss": "runs/remote_no_rn/no_rn_nostn_stnloss_best975.pth",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    z, mask = grid.get_ghbs_grid()  # numpy，保持 CPU（ops/legacy 用 numpy）

    # 加载模型：当前 HBSN+RN + 3 个旧 no-RN（推理可在 GPU，场转 numpy）
    nets_cur = nets.build_net(nets.BEST_CKPT, device=device)
    nets_old = {
        name: legacy.build_old(path, device=device)[0]
        for name, path in NO_RN_CKPTS.items()
    }
    model_names = ["HBSN+RN", *list(nets_old.keys())]

    # 每模型: 相邻对 (sym, flip)
    recs = {name: [] for name in model_names}
    prev = dict.fromkeys(model_names)

    t0 = time.time()
    done = 0
    for fam_name, shapes_list in gen_families():
        for bound in shapes_list:
            img = shapes.shape_to_image(bound)[..., 0]
            sb = sym_geom(bound)
            # 当前 HBSN+RN
            hf_cur = nets.field_to_complex(nets.infer(nets_cur, img, device=device))
            fields = {"HBSN+RN": hf_cur}
            for name, net in nets_old.items():
                fields[name] = legacy.infer_to_ghbs(net, img, z, mask)
            for name, hf in fields.items():
                if prev[name] is not None:
                    recs[name].append((sb, ops.is_flip(prev[name], hf, z, mask)))
                prev[name] = hf
            done += 1
        print(f"族 {fam_name}: 累计 {done} 形状, {time.time()-t0:.0f}s")

    # ---- 分箱统计 ----
    print("\n=== 翻转计数（翻转数/总对数）===")
    header = f"{'bin':<14}{'total':>6}"
    for name in model_names:
        header += f"{name:>14}"
    print(header)
    edges = np.array(BINS)
    for b in range(len(BIN_LABELS)):
        row = f"{BIN_LABELS[b]:<14}"
        m_total = None
        for name in model_names:
            arr = np.array(recs[name], dtype=float)
            m = (np.digitize(arr[:, 0], edges) - 1) == b
            t = m.sum()
            c = int(arr[m, 1].sum())
            if m_total is None:
                m_total = t
            row += f"{c}/{t:>6}".rjust(14)
        print(row)

    # 翻率表
    print("\n=== 翻率（%）===")
    print(f"{'bin':<14}" + "".join(f"{name:>14}" for name in model_names))
    for b in range(len(BIN_LABELS)):
        row = f"{BIN_LABELS[b]:<14}"
        for name in model_names:
            arr = np.array(recs[name], dtype=float)
            m = (np.digitize(arr[:, 0], edges) - 1) == b
            t = m.sum()
            r = arr[m, 1].sum() / t * 100 if t else np.nan
            row += f"{r:>13.1f}%"
        print(row)

    print(f"\n总形状 {done}, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
