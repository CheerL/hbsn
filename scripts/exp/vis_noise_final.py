#!/usr/bin/env python3
"""E5 (b)-panel final render (Round B): 8 columns from the user-picked list.

PICKED = (σ, image_id) in display order (user pick, 2026-09-08). Replays the
exact vis_m1m2m3w noise/inference pipeline via vis_noise_candidates helpers,
cross-checks IoU against the Round-A CSV, then renders 5 rows × 8 cols →
figures/hbsn/f_noise_vis_final.png (input to compose_noise_ab.py).

Usage: uv run python scripts/exp/vis_noise_final.py
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vis_m1m2m3w import ROW_LABELS
from vis_noise_candidates import _replay, _setup

OUT = "figures/hbsn/f_noise_vis_final.png"
CSV = "figures/hbsn/candidates/noise_candidates.csv"
# 展示顺序 = 用户挑选顺序（σ 升序分组，组内按用户给出顺序）
PICKED = [
    (0.0, 15272),
    (0.0, 331569),
    (0.1, 263474),
    (0.1, 289586),
    (0.2, 85478),
    (0.2, 550797),
    (0.3, 466085),
    (0.5, 288391),
]


def _check_against_csv(entries):
    """Cross-check replay IoU vs the Round-A candidate CSV (same seed/replay)."""
    if not os.path.exists(CSV):
        print("[warn] Round-A CSV missing; cross-check skipped")
        return
    ref = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            ref[(float(r["sigma"]), int(r["image_id"]))] = tuple(
                float(r[k]) for k in ("iou_m1", "iou_m2", "iou_m3")
            )
    for e in entries:
        got = tuple(round(v, 4) for v in e["iou"])
        want = ref.get((e["sigma"], e["img_id"]))
        if want is None:
            print(f"[warn] σ={e['sigma']:g} #{e['img_id']}: not in Round-A CSV")
        elif max(abs(a - b) for a, b in zip(got, want, strict=True)) > 0.005:
            print(
                f"[warn] σ={e['sigma']:g} #{e['img_id']}: IoU mismatch vs CSV "
                f"final={got} roundA={want}"
            )
        else:
            print(
                f"  σ={e['sigma']:g} #{e['img_id']}: IoU={got} matches Round-A CSV"
            )


def _render_panel(entries, out):
    """5 rows × 8 cols, old (b)-panel typography (σ title + IoU under preds)."""
    ncols = len(entries)
    nrows = len(ROW_LABELS)
    fig, axs = plt.subplots(nrows, ncols, figsize=(24, 15), squeeze=False)
    left, right, top, bottom = 0.05, 0.99, 0.97, 0.02
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=0.04, hspace=0.15
    )
    for ci, e in enumerate(entries):
        s = e["sigma"]
        axs[0, ci].set_title(f"σ={s:.1f}" if s else "σ=0", fontsize=18)
        axs[0, ci].imshow(np.clip(e["noisy"].transpose(1, 2, 0), 0, 1))
        axs[0, ci].axis("off")
        rows_img = [e["mask_hw"]] + e["preds"]
        for ri, im in enumerate(rows_img):
            ax = axs[1 + ri, ci]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if ri > 0:  # IoU under the three prediction rows (3 dp, as before)
                ax.text(
                    0.5,
                    -0.05,
                    f"IoU={e['iou'][ri - 1]:.3f}",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=13,
                )
    for ri, lab in enumerate(ROW_LABELS):
        y = top - (ri + 0.5) / nrows * (top - bottom)
        fig.text(
            left / 2, y, lab, ha="center", va="center", rotation=90, fontsize=16
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ds, _, nets, dtype, dev = _setup()
    pairs = []
    for s, img_id in PICKED:
        idx = ds.img_ids.index(img_id)
        pairs.append((idx, s))
    out = _replay(ds, nets, dtype, dev, pairs)
    entries = []
    for (s, img_id), (idx, s2) in zip(PICKED, pairs, strict=True):
        e = out[(idx, s2)]
        e["sigma"], e["img_id"] = s, img_id
        entries.append(e)
    _check_against_csv(entries)
    _render_panel(entries, OUT)
    print(f"\noutput: {OUT}  (8 columns, order = PICKED)")
    for e in entries:
        i1, i2, i3 = e["iou"]
        print(
            f"  σ={e['sigma']:g} #{e['img_id']}: "
            f"IoU M1/M2/M3' = {i1:.3f}/{i2:.3f}/{i3:.3f}"
        )


if __name__ == "__main__":
    main()
