#!/usr/bin/env python3
"""Stock-pipeline continuation with explicit rotation-planned targets.

Per-step loss = min_theta masked-MSE(pred, R_theta gt): the argmin is
searched on the CURRENT prediction each step (coarse 5-deg circle +
0.25-deg refine), then net.loss backprops against the aligned teacher
(keeps the stock hbs_loss + stn_loss structure).  Data: img/osc_merged
(original corpus + osc family, stock soft-label/augment transforms).

Starts from the osc_ft2 best checkpoint (val 0.00489).
Pinned hbs==1.0.1 (HBS_PYTHON_DIR) before importing hbsn_exp.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_osc import _align_theta, _rot_torch
from hbsn_exp import grid, nets

from hbsn.registry import get_spec

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _epoch(net, dl, device, opt, mask_t, is_train, fine_step=1.0,
           branch=False, batch_cap=None):
    tot, n = 0.0, 0
    for it, (x, gt) in enumerate(dl):
        if batch_cap is not None and it >= batch_cap:
            break
        x = x.to(device)
        gt = gt.to(device)
        pred = net(x)
        if is_train:
            # loss = min_theta masked-MSE(pred, R_theta gt) + 0.1 stn;
            # bypass net.loss: its internal gt->RN step re-rotates the
            # aligned teacher (measured d 0.06-0.27) and defeats the min
            with torch.no_grad():
                th = _align_theta(pred.detach(), gt, mask_t,
                                  branch=branch, fine_step=fine_step)
            th_rad = torch.tensor(np.deg2rad(th), device=device)
            gt_alg = _rot_torch(gt, th_rad)
            m = mask_t.float()
            hbs = ((((pred - gt_alg) ** 2) * m).sum((1, 2, 3)).sum()
                   / m.sum())
            stn = F.mse_loss(net.post_stn(pred)[0], pred)
            loss = hbs + 0.1 * stn
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_loss = float(loss.item())
        else:
            ldict, _ = net.loss(pred, gt)
            step_loss = float(ldict["loss"].item())
        bs = x.shape[0]
        tot += step_loss * bs
        n += bs
    return tot / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="stock continuation with min-rotation targets")
    p.add_argument("--ckpt", default=os.path.join(
        REPO, "runs/hbsn/Sep10_02-37-17_osc_ft2/checkpoints/best.pth"))
    p.add_argument("--data-dir",
                   default="/home/nnb/projects/HBSN/python/img/osc_merged")
    p.add_argument("--test-dir",
                   default="/home/nnb/projects/HBSN/python/img/gen2")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=6.25e-6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--align-fine", type=float, default=0.25)
    p.add_argument("--align", choices=["full", "branch"], default="full")
    p.add_argument("--out", default=os.path.join(
        REPO, "runs/finetuned/osc/v7/best.pth"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = get_spec("hbsn")
    ds_cfg = OmegaConf.merge(
        OmegaConf.structured(spec.dataset_schema),
        OmegaConf.create({"data_dir": args.data_dir,
                          "test_data_dir": args.test_dir,
                          "is_augment": True}))
    train_ds = spec.dataset(ds_cfg)
    train_dl, _ = train_ds.get_dataloader(batch_size=args.batch_size,
                                          split_rate=1)
    test_ds = spec.dataset(ds_cfg, is_test=True)
    _, test_dl = test_ds.get_dataloader(batch_size=args.batch_size,
                                        split_rate=0)
    print(f"train {len(train_ds)} / test {len(test_ds)}", flush=True)
    net = nets.build_net(args.ckpt, device="cuda").to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=1e-5, betas=(0.9, 0.999))
    _, mask = grid.get_ghbs_grid()
    mask_t = torch.tensor(mask)[None, None].to(device)
    best = np.inf
    for epoch in range(args.epochs):
        net.train()
        tr = _epoch(net, train_dl, device, opt, mask_t, True,
                    args.align_fine, args.align == "branch")
        net.eval()
        with torch.no_grad():
            te = _epoch(net, test_dl, device, opt, mask_t, False)
        print(f"epoch {epoch + 1}/{args.epochs}: train {tr:.6f} "
              f"test {te:.6f}", flush=True)
        if te < best:
            best = te
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": net.cpu().state_dict(),
                        "config": {"net": dict(net.config),
                                   "dataset": dict(ds_cfg)}},
                       args.out)
            net.to(device)
            print(f"  saved best ({best:.6f})", flush=True)
    print(f"done, best test {best:.6f} -> {args.out}")


if __name__ == "__main__":
    main()
