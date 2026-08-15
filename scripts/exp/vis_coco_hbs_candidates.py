#!/usr/bin/env python3
"""Coco 候选人工审核图：unet/deeplab 的 row2(big) 与 row3(mod) 各 10 个候选。

用法：uv run python scripts/exp/vis_coco_hbs_candidates.py
输出：figures/hbsn/cand_{unet,deeplab}_{big,mod}.png

候选池与最终图同一 scan 口径（relax=True 同 seed/offset），审核选定后把 idx
写进 vis_coco_hbs.FORCE_ROWS 对应行即可复现。

每行左起：编号+IoU/coh/dH 统计列，然后 Input / GT mask / GT HBS / base mask /
base HBS / +HBSN mask / +HBSN HBS。HBS 场统一 |HBS| jet、[0,1] 截断。
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vis_coco_hbs import COLUMNS, PAIRS, render_hbs, scan

# 已定稿/已弃用：行1/行4 定稿，行2/行3 现候选待换（全部排除，池里只留新选择）
EXCLUDE = {
    "unet": {477, 447, 456, 486},
    "deeplab": {551, 262, 511, 10},
}
CATS = ("big", "mod")  # 行2=big, 行3=mod
N_SHOW = 10
CELL = 1.8  # 列名（DeepLabV3+HBSN…）在 1.7in 下会贴边，稍放宽
# unet 的 hbsn HBS 须贴近 GT：审核池按 dh_h 升序排，并硬性卡 dh_h ≤ 上限
DH_CAP = {"unet": 0.16, "deeplab": None}


def _render_grid(key, cat, entries, out):
    rows = [e for e in entries if e[2][0] not in EXCLUDE[key]]
    cap = DH_CAP[key]
    if cap is not None:
        rows = [e for e in rows if e[2][13] <= cap]
    if key == "unet":
        rows.sort(key=lambda e: e[2][13])  # dh_h 升序：hbsn HBS 贴近 GT 的优先
    # 去重：同 idx 且 IoU 相同 → 同一裁剪内容，只留一个（unet 大提升池在多 offset 下会重复）
    seen, uniq = set(), []
    for e in rows:
        ent = e[2]
        k = (ent[0], round(ent[8], 3), round(ent[9], 3))
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    rows = uniq
    rows = rows[:N_SHOW]
    if not rows:
        print(f"  [warn] {key}/{cat}: 无候选")
        return
    cols = COLUMNS[key]
    nrows = len(rows)
    ncols = len(cols) + 1  # +1 统计列
    fig = plt.figure(figsize=(ncols * CELL, (nrows + 0.5) * CELL))
    gs = fig.add_gridspec(
        nrows + 1, ncols,
        width_ratios=[0.9] + [1] * len(cols),
        height_ratios=[0.15] + [1] * nrows,
        left=0.01, right=0.99, top=0.99, bottom=0.01,
        wspace=0.02, hspace=0.01,
    )
    axs = np.empty((nrows + 1, ncols), dtype=object)
    for r in range(nrows + 1):
        for c in range(ncols):
            axs[r, c] = fig.add_subplot(gs[r, c])
    axs[0, 0].axis("off")
    for c, lab in enumerate(cols):
        axs[0, c + 1].text(
            0.5, 0.0, lab, ha="center", va="bottom",
            transform=axs[0, c + 1].transAxes, fontsize=11,
        )
        axs[0, c + 1].axis("off")
    print(f"\n=== {key} / {cat}（显示 {nrows} 个）===")
    for r, (off, _, e) in enumerate(rows):
        (idx, img, mask_hw, m_base, m_hbsn,
         f_gt, f_base, f_hbsn,
         iou_b, iou_h, coh_b, coh_h, dh_b, dh_h) = e
        cells = [
            np.clip(img.transpose(1, 2, 0), 0, 1),
            mask_hw,
            render_hbs(f_gt),
            m_base,
            render_hbs(f_base),
            m_hbsn,
            render_hbs(f_hbsn),
        ]
        for c, cell_img in enumerate(cells):
            ax = axs[r + 1, c + 1]
            if c in (0, 2, 4, 6):
                ax.imshow(cell_img)
            else:
                ax.imshow(cell_img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axs[r + 1, 0].text(
            0.5, 0.5,
            f"{r + 1}. #{idx}@{off}\nIoU {iou_b:.2f}→{iou_h:.2f}\n"
            f"coh {coh_b:.2f}→{coh_h:.2f}\ndH {dh_b:.3f}→{dh_h:.3f}",
            ha="center", va="center", transform=axs[r + 1, 0].transAxes,
            fontsize=10,
        )
        axs[r + 1, 0].axis("off")
        print(f"  [{r + 1}] off={off} #{idx}: IoU {iou_b:.3f}→{iou_h:.3f} "
              f"coh {coh_b:.2f}→{coh_h:.2f} dH {dh_b:.3f}→{dh_h:.3f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"输出: {out}")


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for key in PAIRS:
        cands, n = scan(key, 561, device, relax=True)
        print(f"===== {key}（扫 {n} 张）池: "
              f"big {len(cands['big'])}, mod {len(cands['mod'])}")
        for cat in CATS:
            _render_grid(key, cat, cands[cat],
                         f"figures/hbsn/cand_{key}_{cat}.png")


if __name__ == "__main__":
    main()
