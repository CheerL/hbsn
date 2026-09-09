#!/usr/bin/env python3
"""Osc-family finetune of the standalone HBSNet.

Lightly finetunes the best checkpoint on triangle_slide(x=0.55 sin psi,
h=0.933) so the predicted |field| images track the classical output per
column.  Flip windows (classical seam jumps) and NaN dead bands are
excluded from supervision so HBSN keeps morphing through the seam
instead of learning the jump.

Pinned hbs==1.0.1 (HBS_PYTHON_DIR) before importing hbsn_exp.
"""

import argparse
import os
import sys

import numpy as np
import torch

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, nets, ops, shapes

H_SLIDE = 0.933
AMP = 0.55
FLIPS = [40.38, 139.62, 220.89, 319.11]
FIELD_CACHE = "/tmp/h3scan/fields_osc.npz"
IMG_CACHE = "/tmp/h3scan/imgs_osc.npz"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(REPO, "runs", "finetuned", "osc")
COLS = [40.3, 40.5, 90.0, 139.5, 139.7, 180.0, 220.8, 221.0, 289.0, 319.0, 319.2]
HELD_OUT = [
    ("circle", lambda: shapes.circle(n=300, r=0.85)),
    ("ellipse", lambda: shapes.ellipse(n=300, a=1.0, b=0.7)),
    ("star_k3", lambda: shapes.star(n=300, k=3)),
    ("star_k5", lambda: shapes.star(n=300, k=5)),
    ("quad", lambda: shapes.quadrilateral(n=300)),
    ("semi_t0.3", lambda: shapes.triangle(t=0.3, base=0.7)),
    ("semi_t0.5", lambda: shapes.triangle(t=0.5, base=0.7)),
]
HELD_N = 7
# per-prefix lr multiplier (0 = frozen); user asked for backbone finetune
VARIANTS = {
    "V1": {"backbone": 1.0, "pre_stn": 1.0, "post_stn": 1.0},
    "V2": {"backbone": 1.0, "pre_stn": 0.2, "post_stn": 0.0},
    "V3": {"backbone": 1.0, "pre_stn": 0.0, "post_stn": 0.0},
}


def _osc_bound(psi):
    return shapes.triangle_slide(x=AMP * np.sin(np.deg2rad(psi)), h=H_SLIDE)


def _valid(ths, cf, window):
    """Samples with a finite classical teacher, outside the flip windows."""
    ok = np.array([not np.all(np.isnan(c)) for c in cf])
    for f in FLIPS:
        ok &= np.abs(ths - f) > window
    return ok


def _images(ths, force=False):
    """Pre-rendered input images, cached in /tmp (matplotlib render)."""
    if not force and os.path.exists(IMG_CACHE):
        return np.load(IMG_CACHE)["imgs"]
    out = np.zeros((len(ths), 256, 256), dtype=np.uint8)
    for i, psi in enumerate(ths):
        out[i] = shapes.shape_to_image(_osc_bound(psi))[..., 0]
    np.savez_compressed(IMG_CACHE, imgs=out)
    return out


def _param_groups(net, variant, lr):
    """Adam groups per prefix; rate 0 freezes the group."""
    groups = []
    for prefix, rate in VARIANTS[variant].items():
        params = [p for n, p in net.named_parameters()
                  if n.startswith(prefix)]
        if rate == 0:
            for p in params:
                p.requires_grad_(False)
            continue
        groups.append({"params": params, "lr": lr * rate})
    return groups


def _distill_set(net, ths, imgs, device):
    """Behavior anchors: base-net outputs at flip neighborhoods (pins the
    morph zone / continuity story) and on held-out shapes (anti-forgetting).
    """
    idxs = np.unique(np.concatenate(
        [np.where(np.abs(ths - f) <= 2.0)[0][::4] for f in FLIPS]))
    x_fam = imgs[idxs].astype(np.float32)
    gt_fam = _infer_all(net, imgs[idxs], device)
    gt_fam = np.stack([np.stack([g.real, g.imag]) for g in gt_fam])
    ho = np.stack([shapes.shape_to_image(mk())[..., 0] for _, mk in HELD_OUT])
    gt_ho = _infer_all(net, ho, device)
    gt_ho = np.stack([np.stack([g.real, g.imag]) for g in gt_ho])
    x = np.concatenate([x_fam, ho])
    gt = np.concatenate([gt_fam, gt_ho])
    return x, gt


def _train(net, args, imgs, gts, sel, distill=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device).train()
    # freeze BN running stats: a 1-parametric family would drag the
    # pretrained statistics toward triangles (seen as forgetting elsewhere)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    groups = _param_groups(net, args.variant, args.lr)
    opt = torch.optim.Adam(groups, lr=args.lr)
    anchor0 = ([[q.detach().clone() for q in g["params"]]
                for g in groups] if args.anchor > 0 else None)
    x_all = (torch.tensor(imgs[sel].astype(np.float32))[:, None].to(device)
             / 255.0)
    gt_all = torch.tensor(gts[sel], dtype=torch.float32).to(device)
    if distill is not None:
        x_d = (torch.tensor(distill[0].astype(np.float32))[:, None]
               .to(device) / 255.0)
        gt_d = torch.tensor(distill[1], dtype=torch.float32).to(device)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(sel), device=device)
        losses = []
        for lo in range(0, len(sel), args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            pred = net(x_all[idx])
            ldict, _ = net.loss(pred, gt_all[idx])
            loss = ldict["loss"]
            if x_d is not None:
                j = torch.randint(
                    0, len(x_d), (min(args.batch_size, len(x_d)),),
                    device=device)
                loss = loss + args.distill_rate * torch.mean(
                    (net(x_d[j]) - gt_d[j]) ** 2)
            if anchor0 is not None:
                a = torch.zeros((), device=device)
                for g, g0 in zip(groups, anchor0, strict=True):
                    for q, q0 in zip(g["params"], g0, strict=True):
                        a = a + torch.sum((q - q0) ** 2)
                loss = loss + args.anchor * a
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(ldict["loss"].item())
        mean = float(np.mean(losses))
        print(f"  epoch {epoch + 1}/{args.epochs}: loss {mean:.6f}",
              flush=True)
        if mean < best:
            best, wait = mean, 0
            best_state = {k: v.cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop at epoch {epoch + 1}")
                break
    net.load_state_dict(best_state)
    net.cpu().eval()
    return float(best)


def _infer_all(net, imgs, device):
    """Complex fields over the full dense grid (batched, no grad)."""
    net = net.to(device).eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(imgs), 32):
            x = (torch.tensor(imgs[lo:lo + 32].astype(np.float32))[:, None]
                 .to(device) / 255.0)
            pred = net(x)
            out += [p[0] + 1j * p[1] for p in pred.cpu().numpy()]
    return out


def _d_mod(fa, fb, mask):
    return float(np.sqrt(np.mean((np.abs(fa)[mask] - np.abs(fb)[mask]) ** 2)))


def _reg_mod(fa, fb, z, mask):
    """min-rotation RMS of |field| (5-deg coarse + 1-deg fine)."""
    best = (0.0, np.inf)
    for st in (5.0, 1.0):
        lo, hi = (0, 359) if st == 5.0 else (best[0] - 6, best[0] + 6)
        for phi in np.arange(lo, hi + st / 2, st):
            rot = np.abs(ops.rotate_mu(fa, np.deg2rad(phi), z, mask))
            dd = _d_mod(rot, fb, mask)
            if dd < best[1]:
                best = (float(phi % 360), float(dd))
    return best


def _report(net, base, ths, cf, imgs, hf_old, device):
    """11-col before/after table + morph peaks + held-out sanity."""
    z, mask = grid.get_ghbs_grid()
    base = base.to(device)
    hf_new = _infer_all(net, imgs, device)
    print(f"{'psi':>6} {'dmod_old':>8} {'dmod_new':>8} "
          f"{'scale':>6} {'reg_d':>6} {'reg_phi':>7}")
    for ps in COLS:
        k = int(np.argmin(np.abs(ths - ps)))
        if np.all(np.isnan(cf[k])):
            print(f"{ps:6.1f}  classic NaN")
            continue
        c = cf[k].astype(complex)
        o, n = hf_old[k].astype(complex), hf_new[k]
        phi, rd = _reg_mod(np.abs(c), np.abs(n), z, mask)
        print(f"{ps:6.1f} {_d_mod(np.abs(c), np.abs(o), mask):8.4f} "
              f"{_d_mod(np.abs(c), np.abs(n), mask):8.4f} "
              f"{np.abs(n).max() / np.abs(c).max():6.3f} "
              f"{rd:6.4f} {phi:7.1f}")
    # morph-peak guard: adjacent-d peaks near flips must stay bounded
    print("morph peaks (adjacent-d near flips):")
    for f in FLIPS:
        vals = [(ops.d2(hf_new[i], hf_new[i + 1], z, mask),
                 0.5 * (ths[i] + ths[i + 1]))
                for i in range(len(ths) - 1)
                if abs(0.5 * (ths[i] + ths[i + 1]) - f) < 5
                and np.isfinite(np.abs(hf_new[i]).sum())
                and np.isfinite(np.abs(hf_new[i + 1]).sum())]
        if vals:
            peak = max(vals)
            print(f"  flip {f:6.2f}: peak d {peak[0]:.4f} at psi {peak[1]:.2f}")
    # held-out sanity (teacher computed live, NaN teacher -> SKIP)
    print("held-out shapes (d_modulus base -> finetuned):")
    for name, mk in HELD_OUT:
        try:
            c = classic.classic_pixel(mk())
        except RuntimeError:
            print(f"  {name}: teacher failed, SKIP")
            continue
        img = shapes.shape_to_image(mk())[..., 0]
        a = _infer_all(base, img[None], device)[0]
        b = _infer_all(net, img[None], device)[0]
        print(f"  {name}: {_d_mod(np.abs(c), np.abs(a), mask):.4f} -> "
              f"{_d_mod(np.abs(c), np.abs(b), mask):.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="osc-family HBSN finetune")
    p.add_argument("--variant", default="V1", choices=["V1", "V2", "V3"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--flip-window", type=float, default=1.5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor", type=float, default=1e-3,
                   help="L2-SP weight-anchor coefficient (0 = off)")
    p.add_argument("--distill-rate", type=float, default=1.0,
                   help="behavior-anchoring distillation weight (0 = off)")
    p.add_argument("--precache", action="store_true")
    p.add_argument("--eval-only")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = np.load(FIELD_CACHE)
    ths, cf = d["ths"], d["cf"]
    imgs = _images(ths, force=args.precache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = nets.build_net(nets.BEST_CKPT)
    if args.eval_only:
        net = nets.build_net(args.eval_only)
        _report(net, base, ths, cf, imgs, d["hf"], device)
        return
    sel = np.where(_valid(ths, cf, args.flip_window))[0][::args.stride]
    gts = np.stack([np.stack([c.real, c.imag]) for c in cf])
    print(f"training samples: {len(sel)} (stride {args.stride}, "
          f"flip-window {args.flip_window})")
    print(f"variant {args.variant}: " + ", ".join(
        f"{k} x{v}" for k, v in VARIANTS[args.variant].items()))
    net = nets.build_net(nets.BEST_CKPT)
    distill = None
    if args.distill_rate > 0:
        distill = _distill_set(net, ths, imgs, device)
        print(f"distill anchors: {len(distill[0])} samples")
    best = _train(net, args, imgs, gts, sel, distill=distill)
    os.makedirs(os.path.join(OUT_DIR, args.variant), exist_ok=True)
    path = os.path.join(OUT_DIR, args.variant, "best.pth")
    torch.save({"state_dict": net.state_dict(),
                "config": {"net": dict(net.config), "dataset": {}}}, path)
    print(f"saved: {path} (best train loss {best:.6f})")
    _report(net, base, ths, cf, imgs, d["hf"], device)


if __name__ == "__main__":
    main()
