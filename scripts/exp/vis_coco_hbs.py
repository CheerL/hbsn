#!/usr/bin/env python3
"""Coco 分割对比图：UNet 与 DeepLabV3 各一张，4 组 × 7 列。

每行（左→右）：Input / GT mask / GT HBS / w/o-HBSN mask / w/o-HBSN HBS /
w/-HBSN mask / w/-HBSN HBS。HBS 场统一 |HBS| jet、[0,1] 截断（见项目约定）。

选图标准（每张图 4 组，全部来自 connected 单连通子集 561 张）：
- 大提升 ×2：base 掩码碎（低 coherence / 多碎片）、+hbsn 完整清晰、IoU 提升大
- 中等提升 ×1：IoU 提升适中、+hbsn 掩码完整
- HBS 差距 ×1：两者 IoU 都好（掩码完整），但 base pred HBS 与 GT HBS 差距明显

用法：uv run python scripts/exp/vis_coco_hbs.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label as cc_label
from scipy.ndimage import rotate as nd_rotate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_coco import _build_configs
from finetune_noise import build_bce_hbsn

from hbsn.data.dataset import RANDOM_SEED
from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

DATA = {
    "data_dir": "coco/val2017",
    "annotation_path": "coco/annotations/instances_val2017.json",
}
SETS = ["single_instance=true", "connected=true"]

# 迁移 ckpt 的 hbsn 子网默认架构加载下为近常数（strict=False 部分加载损坏），
# HBS 统一用远程预训练 HBSN（build_bce_hbsn，与 finetune 同款，输入敏感）。
PAIRS = {
    "unet": {
        "spec": "unetpp",
        "base": "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth",
        "hbsn": "runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth",
        "out": "figures/hbsn/coco_unet.png",
        "hbsn_ckpt": "runs/remote_hbsn/hbsn_stn3_loog3_best1481.pth",
        "cd": [8, 16, 32, 64, 128, 256],
        "cu": [8, 16, 32, 64, 128, 256],
    },
    "deeplab": {
        "spec": "deeplab",
        "base": "runs/migrated/deeplab/May15_11-02-15_hbs0_all_c/checkpoints/best.pth",
        "hbsn": "runs/migrated/deeplab/May23_21-02-42_hbs0.05_all/checkpoints/epoch_195.pth",
        "out": "figures/hbsn/coco_deeplab.png",
        "hbsn_ckpt": "runs/remote_hbsn/hbsn_stn3_soft_all2_best.pth",
        "cd": [8, 8, 16, 32, 64, 128],
        "cu": [8, 16, 32, 64, 128],
    },
}

COLUMNS = {
    "unet": [
        "Input", "GT mask", "GT HBS", "UNet mask", "UNet HBS",
        "UNet+HBSN mask", "UNet+HBSN HBS",
    ],
    "deeplab": [
        "Input", "GT mask", "GT HBS", "DeepLabV3 mask", "DeepLabV3 HBS",
        "DeepLabV3+HBSN mask", "DeepLabV3+HBSN HBS",
    ],
}
KEEP = 20  # 每类候选池大小（含人工审核候选池）
# BoundedRandomCrop 是随机裁剪：num_workers=0 下主进程 manual_seed 完全控制 → 跨进程可复现。
# 每 offset 一套裁剪位置；候选池跨多个 offset 收集供人工审核。unet offset 0 即有理想
# big 候选（#477: 0.33→0.89, dh_h 0.070）。
SEED_OFFSETS = {"unet": [0, 1, 2, 3, 4, 5, 6, 7], "deeplab": [8, 9, 10, 11]}
# 选图阈值（可调）
BIG_MIN_DELTA = 0.07  # 大提升：IoU 差下限
BIG_IOU_BASE = (0.10, 0.55)  # base 掩码有检测但差（排除空检测与已很好）
BIG_COH_BASE = 0.80  # base 掩码碎：coherence 上限
BIG_COH_HBSN = 0.90  # +hbsn 完整：coherence 下限
BIG_IOU_H = {"unet": 0.70, "deeplab": 0.75}  # 大提升：+hbsn 掩码须明显好
MOD_COH_HBSN = 0.85  # 中等提升：+hbsn 完整下限
GAP_BOTH_IOU = 0.75  # HBS 差距组：两者 IoU 下限（都"基本很好"）
GAP_COH = 0.90  # HBS 差距组：两者掩码都完整
GAP_DH_MIN = 0.12  # HBS 差距组：base pred HBS 与 GT HBS 距离下限
GAP_MIN_ROT = 0.05  # gap 排除"只差旋转"：最小旋转对齐距离下限
DH_H_MAX = {"unet": 0.15, "deeplab": 0.12}  # +hbsn HBS 与 GT 距离上限（裁剪后量纲）
RELAX = True  # 候选池口径：放宽阈值供人工审核（最终图同口径才能复现候选）


def _iou(pred, mask):
    inter = np.logical_and(pred > 0.5, mask > 0.5).sum()
    union = np.logical_or(pred > 0.5, mask > 0.5).sum()
    return inter / (union + 1e-8)


def _coherence(pred):
    lab, n = cc_label(pred > 0.5)
    if n == 0:
        return 0.0
    return np.bincount(lab.ravel())[1:].max() / (lab > 0).sum()


def _hbs_dist(f1, f2):
    """HBS 场 RMS 距离（论文式 7.1）：sqrt(mean(|Δ|²))。"""
    return float((f1 - f2).abs().pow(2).mean().sqrt().item())


def _min_rot_dist(f1, f2):
    """f1 相对 f2 的最小旋转对齐距离：排除"HBS 只是差一个旋转"的伪差距。

    复场分别旋转实部/虚部（近似 HBS 的旋转等变性），取 12 个角度的最小 RMS 距离。
    """
    a = f1.float().cpu().numpy()
    b = f2.float().cpu().numpy()
    best = float("inf")
    for ang in range(0, 360, 30):
        r = nd_rotate(a[0, 0], ang, reshape=False, order=1)
        i = nd_rotate(a[0, 1], ang, reshape=False, order=1)
        d = np.sqrt(((r - b[0, 0]) ** 2 + (i - b[0, 1]) ** 2).mean())
        best = min(best, d)
    return float(best)


def render_hbs(field):
    """|HBS| jet、[0,1] 截断（项目约定，不加色条）。复场 (B,2,H,W) 取模长。"""
    f = field.float().cpu().numpy()
    mag = np.sqrt(f[..., 0, :, :] ** 2 + f[..., 1, :, :] ** 2)
    mag = np.squeeze(mag)
    return plt.get_cmap("jet")(np.clip(mag, 0, 1))[:, :, :3]


def load_pair(key, device):
    """加载 base 与 +hbsn 两个分割模型，返回 (net_base, net_hbsn)。"""
    p = PAIRS[key]
    spec = get_spec(p["spec"])
    nets = []
    for ckpt_path in (p["base"], p["hbsn"]):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net_cfg, _ = _build_configs(
            spec, ckpt.get("config", {}), {"device": device, **DATA}, SETS
        )
        net = spec.net.factory(net_cfg)
        net.load_state_dict(ckpt["state_dict"], strict=False)
        net.eval()
        nets.append(net)
    return nets[0], nets[1]


def _thr(pair_key, relax):
    """筛选阈值：relax=True 为人工审核候选池（更宽；最终图同口径才可复现候选）。"""
    dh = DH_H_MAX[pair_key]
    if not relax:
        return {
            "big": (BIG_MIN_DELTA, BIG_IOU_BASE, BIG_COH_BASE, BIG_COH_HBSN,
                    BIG_IOU_H[pair_key], dh),
            "mod": ((0.03, 0.10), (0.40, 0.75), 0.90, MOD_COH_HBSN, dh),
        }
    return {
        "big": (0.05, (0.08, 0.60), 0.85, 0.82,
                (0.60 if pair_key == "unet" else 0.65), dh + 0.03),
        "mod": ((0.02, 0.15), (0.30, 0.80), 0.85, 0.78, dh + 0.03),
    }


def _scan_offset(pair_key, off, max_samples, device, relax):
    """单个裁剪 offset 的扫描。返回 (cands dict, n)，每类为 [(off, score, entry), ...]。"""
    p = PAIRS[pair_key]
    spec = get_spec(p["spec"])
    ckpt = torch.load(p["base"], map_location="cpu", weights_only=False)
    _, dataset_cfg = _build_configs(
        spec, ckpt.get("config", {}), {"device": device, **DATA}, SETS
    )
    dataset_cfg.num_workers = 0  # BoundedRandomCrop 在主进程跑 → manual_seed 完全控制
    ds = spec.dataset(dataset_cfg)
    # 固定裁剪位置 → 跨进程可复现（每 offset 一套裁剪）
    torch.manual_seed(RANDOM_SEED + off)
    _, dl = ds.get_dataloader(batch_size=1, split_rate=0, drop_last=True)

    net_base, net_hbsn = load_pair(pair_key, device)
    dtype = torch_dtype(net_base.config)
    dev = net_base.config.device
    # 冻结预训练 HBSN：远程 ckpt，bce 旧架构。unet(loog3) 原生 256、deeplab(soft_all2) 原生 128。
    hbsn = build_bce_hbsn(p["hbsn_ckpt"], p["cd"], p["cu"], dev).to(dev)

    def hbs_of(mask):
        f = hbsn(mask)
        if f.shape[-1] == 256:  # unet：取中央 128（磁盘填满，与 deeplab 128 对齐）
            f = f[..., 64:192, 64:192]
        return f

    cands = {"big": [], "mod": [], "gap": []}
    t = _thr(pair_key, relax)
    b_min_delta, b_iou_b, b_coh_b, b_coh_h, b_iou_h, b_dh = t["big"]
    m_delta, m_iou_b, m_coh_b, m_coh_h, m_dh = t["mod"]
    n = 0
    for i, (img, mask) in enumerate(dl):
        if max_samples and i >= max_samples:
            break
        n += 1
        img = img.to(dev, dtype=dtype)
        mask_t = mask.to(dev, dtype=dtype)
        mask_hw = mask[0, 0].float().cpu().numpy()

        with torch.no_grad():
            hard_base = net_base.get_hard_mask(net_base(img)[0])[0, 0]
            hard_hbsn = net_hbsn.get_hard_mask(net_hbsn(img)[0])[0, 0]
            f_gt = hbs_of(mask_t)
            f_base = hbs_of(hard_base.unsqueeze(0).unsqueeze(0))
            f_hbsn = hbs_of(hard_hbsn.unsqueeze(0).unsqueeze(0))
        iou_b = _iou(hard_base.float().cpu().numpy(), mask_hw)
        iou_h = _iou(hard_hbsn.float().cpu().numpy(), mask_hw)
        coh_b = _coherence(hard_base.float().cpu().numpy())
        coh_h = _coherence(hard_hbsn.float().cpu().numpy())
        dh_b = _hbs_dist(f_base, f_gt)
        dh_h = _hbs_dist(f_hbsn, f_gt)

        entry = (
            i,
            img[0].float().cpu().numpy(),
            mask_hw,
            hard_base.float().cpu().numpy(),
            hard_hbsn.float().cpu().numpy(),
            f_gt,
            f_base,
            f_hbsn,
            iou_b,
            iou_h,
            coh_b,
            coh_h,
            dh_b,
            dh_h,
        )
        delta = iou_h - iou_b
        dh_max = b_dh
        if (
            delta >= b_min_delta
            and b_iou_b[0] <= iou_b <= b_iou_b[1]
            and coh_b <= b_coh_b
            and coh_h >= b_coh_h
            and iou_h >= b_iou_h
            and dh_h <= dh_max
        ):
            cands["big"].append((off, delta + (coh_h - coh_b) * 0.2, entry))
        if (
            m_delta[0] <= delta <= m_delta[1]
            and m_iou_b[0] <= iou_b <= m_iou_b[1]
            and coh_b >= m_coh_b  # base 完整但有小瑕疵（"碎"留给 big 组）
            and coh_h >= m_coh_h
            and dh_h <= m_dh
        ):
            cands["mod"].append((off, delta, entry))
        if (
            iou_b >= GAP_BOTH_IOU
            and iou_h >= GAP_BOTH_IOU
            and coh_b >= GAP_COH
            and coh_h >= GAP_COH
            and dh_b >= GAP_DH_MIN
            and dh_h < dh_b
            and dh_h <= dh_max
            and iou_h >= iou_b  # w/ HBSN 不劣于 base
            and _min_rot_dist(f_base, f_gt) > GAP_MIN_ROT  # 排除"只差旋转"
        ):
            cands["gap"].append((off, (dh_b - dh_h), entry))
    return cands, n


def scan(pair_key, max_samples, device, offsets=None, relax=RELAX):
    """跨多个裁剪 offset 扫描，合并候选池。返回 (cands dict, n)。

    cands 每类为 [(offset, score, entry), ...]，按 score 降序截断到 KEEP。
    offsets 默认 SEED_OFFSETS[model]；最终图按需只扫 FORCE_ROWS 用到的 offset。
    """
    if offsets is None:
        offsets = SEED_OFFSETS[pair_key]
    cands = {"big": [], "mod": [], "gap": []}
    n = 0
    for off in offsets:
        part, nn = _scan_offset(pair_key, off, max_samples, device, relax)
        n += nn
        for cat in cands:
            cands[cat].extend(part[cat])
    for cat in cands:
        cands[cat].sort(key=lambda e: e[1], reverse=True)
        del cands[cat][KEEP:]
    return cands, n


# 人工审核定稿：每行指定 (offset, idx)（None=自动取池内最优）。
# 行序: row1 big / row2 big / row3 mod / row4 gap；(offset, idx) 须在对应类别池中。
FORCE_ROWS = {
    "unet": [(0, 477), (7, 168), (0, 121), (0, 447)],
    "deeplab": [(11, 169), (9, 548), (8, 219), (8, 262)],
}


def _pick(cands, forced):
    """按定稿行取候选：forced[i]=(offset, idx) 指定该图，否则池内最优；4 组互不相同。"""
    chosen = []
    used = set()
    for i, cat in enumerate(("big", "big", "mod", "gap")):
        f = forced[i]
        if f is not None:
            off, fidx = f
            for e in cands[cat]:
                if e[0] == off and e[2][0] == fidx:
                    chosen.append(e[2])
                    used.add(fidx)
                    break
            else:
                raise RuntimeError(f"定稿行 {i} ({off},{fidx}) 不在 {cat} 池中")
            continue
        for e in cands[cat]:
            if e[2][0] not in used:
                chosen.append(e[2])
                used.add(e[2][0])
                break
        else:
            raise RuntimeError(f"类别 {cat} 候选不足")
    return chosen


def _header_label(lab):
    """长列名折行：'DeepLabV3+HBSN mask' → 两行，避免横向溢出。"""
    for word in ("mask", "HBS"):
        suffix = " " + word
        if lab.endswith(suffix) and len(lab) > 16:
            return lab[: -len(suffix)] + "\n" + word
    return lab


def render(key, chosen, out):
    cols = COLUMNS[key]
    nrows = len(chosen)
    ncols = len(cols)
    cell = 2.1
    hdr = 0.28 if key == "deeplab" else 0.15  # deeplab 长列名两行，表头略高
    # 表头行很薄 + 文本贴底（va=bottom）：列首紧贴数据行
    fig = plt.figure(figsize=(ncols * cell, (nrows + 0.45) * cell))
    gs = fig.add_gridspec(
        nrows + 1, ncols, height_ratios=[hdr] + [1] * nrows,
        left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.01,
    )
    axs = np.empty((nrows + 1, ncols), dtype=object)
    for r in range(nrows + 1):
        for c in range(ncols):
            axs[r, c] = fig.add_subplot(gs[r, c])
    header_fs = 15 if key == "unet" else 13  # deeplab 列名长（DeepLabV3+HBSN…）降号+折行
    for c, lab in enumerate(cols):
        axs[0, c].text(
            0.5, 0.0, _header_label(lab), ha="center", va="bottom",
            transform=axs[0, c].transAxes, fontsize=header_fs,
        )
        axs[0, c].axis("off")
    for r, e in enumerate(chosen):
        (
            idx, img, mask_hw, m_base, m_hbsn,
            f_gt, f_base, f_hbsn,
            iou_b, iou_h, _, _, dh_b, dh_h,
        ) = e
        cells = [
            np.clip(img.transpose(1, 2, 0), 0, 1),  # 原图 CHW→HWC
            mask_hw,
            render_hbs(f_gt),
            m_base,
            render_hbs(f_base),
            m_hbsn,
            render_hbs(f_hbsn),
        ]
        for c, cell_img in enumerate(cells):
            ax = axs[r + 1, c]
            if c in (0, 2, 4, 6):  # 彩色：原图 + HBS 场
                ax.imshow(cell_img)
            else:  # 灰度掩码
                ax.imshow(cell_img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        print(
            f"  #{idx}: IoU {iou_b:.3f}→{iou_h:.3f} (Δ{iou_h - iou_b:+.3f}), "
            f"coh {_coherence(m_base):.2f}→{_coherence(m_hbsn):.2f}, "
            f"dH {dh_b:.3f}→{dh_h:.3f}"
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"输出: {out}")


def main():
    torch.manual_seed(0)  # 扫描必须可复现（GPU conv 默认非确定）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for key in PAIRS:
        print(f"\n===== {key} =====")
        # 只扫定稿行用到的 offset（None 行需全池 → 扫全部）
        needed = set()
        for row in FORCE_ROWS[key]:
            if row is None:
                needed |= set(SEED_OFFSETS[key])
            else:
                needed.add(row[0])
        cands, n = scan(key, 561, device, offsets=sorted(needed))
        print(f"  候选: big {len(cands['big'])}, mod {len(cands['mod'])}, "
              f"gap {len(cands['gap'])}  (扫 {n} 张)")
        chosen = _pick(cands, FORCE_ROWS[key])
        render(key, chosen, PAIRS[key]["out"])


if __name__ == "__main__":
    main()
