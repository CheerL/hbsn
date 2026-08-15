#!/usr/bin/env python3
"""分割网络 w/wo HBSN 结果图（unet/deeplab 分开，connected 单连通子集）。

在 connected（单连通，segmentation 单多边形 + min_area）coco val 子集上，
对比 hbs0（无 HBSN）vs hbs0.05（带 HBSN）的分割 mask IoU，生成 top-N
改进样本结果图（原图 + GT + 各模型预测）。unet 和 deeplab 各出一张图
（分开目录），不混在一起。

用法：uv run python scripts/exp/eval_connected_seg.py [max_samples]
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # scripts/（eval_coco 所在）

from eval_coco import compare_evaluate

OUT = "figures/hbsn/seg"
# connected: 单连通形状（segmentation 单多边形 + min_area 过滤）
COMMON_SETS = ["connected=true", "single_instance=true"]

FAMILIES = {
    "unetpp": [
        (
            "unetpp",
            "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth",
            "hbs0 (wo HBSN)",
        ),
        (
            "unetpp",
            "runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/best.pth",
            "hbs0.05 (w HBSN)",
        ),
    ],
    "deeplab": [
        (
            "deeplab",
            "runs/migrated/deeplab/May15_11-02-15_hbs0_all_c/checkpoints/best.pth",
            "hbs0 (wo HBSN)",
        ),
        (
            "deeplab",
            "runs/migrated/deeplab/May23_21-02-42_hbs0.05_all/checkpoints/best.pth",
            "hbs0.05 (w HBSN)",
        ),
    ],
}


def main():
    max_samples = (
        int(sys.argv[1]) if len(sys.argv) > 1 else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, max_samples: {max_samples}")

    for fam, entries in FAMILIES.items():
        out_dir = os.path.join(OUT, fam)
        print(f"\n===== {fam} =====")
        compare_evaluate(
            entries=entries,
            top_n=6,
            out_dir=out_dir,
            data_dir="coco/val2017",
            annotation_path="coco/annotations/instances_val2017.json",
            sets=COMMON_SETS,
            device=device,
            max_samples=max_samples,
        )
        print(f"结果图: {out_dir}/top_n_improved.png")


if __name__ == "__main__":
    main()
