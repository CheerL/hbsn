#!/usr/bin/env python3
"""Osc-family finetune of the standalone HBSNet (v2: rotation-planned).

v2 redesign after the v1 attempt degraded field structure (central
low-amplitude core scattered, orientation drift, morph peaks up):

- teacher = classical field rotated by the per-sample optimal angle
  (rotation planning: theta* = argmin_theta d(R_theta cf, hf)), refreshed
  each epoch -- the classical seam is absorbed by the alignment, so the
  whole finite family (no flip-window exclusion) can be supervised;
- three-band weighted loss: complex MSE on the high-amplitude band,
  MODULUS MSE on the low-amplitude core (no phase noise there), complex
  MSE on the relaxed middle band;
- post_STN/pre_STN frozen (the RN's canonical frame is coupled to the
  backbone it was calibrated with), BN stats frozen, L2-SP weight anchor,
  behavior distillation on flip neighborhoods (+-5 deg) and held-out
  shapes (anti-forgetting + morph-zone pinning);
- eval-gated early stop: best state tracked by the mean REGISTERED dmod
  of the 11 preview columns, patience 2.

Pinned hbs==1.0.1 (HBS_PYTHON_DIR) before importing hbsn_exp.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, nets, ops, shapes

H_SLIDE = 0.933
AMP = 0.55
FLIPS = [40.38, 139.62, 220.89, 319.11]
FIELD_CACHE = "/tmp/h3scan/fields_osc.npz"
IMG_CACHE = "/tmp/h3scan/imgs_osc.npz"
V1_CACHE = "/tmp/h3scan/fields_osc_ftv1.npz"
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(REPO, "runs", "finetuned", "osc")
COLS = [40.3, 40.5, 90.0, 139.5, 139.7, 180.0, 220.8, 221.0, 289.0,
        319.0, 319.2]
HELD_OUT = [
    ("circle", lambda: shapes.circle(n=300, r=0.85)),
    ("ellipse", lambda: shapes.ellipse(n=300, a=1.0, b=0.7)),
    ("star_k3", lambda: shapes.star(n=300, k=3)),
    ("star_k5", lambda: shapes.star(n=300, k=5)),
    ("quad", lambda: shapes.quadrilateral(n=300)),
    ("semi_t0.3", lambda: shapes.triangle(t=0.3, base=0.7)),
    ("semi_t0.5", lambda: shapes.triangle(t=0.5, base=0.7)),
]


def _osc_bound(psi):
    return shapes.triangle_slide(x=AMP * np.sin(np.deg2rad(psi)), h=H_SLIDE)


def _images(ths, force=False):
    """Pre-rendered input images, cached in /tmp (matplotlib render)."""
    if not force and os.path.exists(IMG_CACHE):
        return np.load(IMG_CACHE)["imgs"]
    out = np.zeros((len(ths), 256, 256), dtype=np.uint8)
    for i, psi in enumerate(ths):
        out[i] = shapes.shape_to_image(_osc_bound(psi))[..., 0]
    np.savez_compressed(IMG_CACHE, imgs=out)
    return out


def _rot_torch(x, th):
    """HBS rotation e^{-2i th} B(e^{i th} z) on a (B,2,H,W) tensor.

    th: per-sample rotation in radians, shape (B,).  Matrix convention
    verified against ops.rotate_mu (worst d 0.077 over 10-100 deg,
    interpolation-level agreement; both sides used consistently).
    """
    th = th.view(-1)
    cos, sin = torch.cos(th), torch.sin(th)
    zero = torch.zeros_like(cos)
    p = torch.stack([torch.stack([cos, sin, zero], 1),
                     torch.stack([-sin, cos, zero], 1)], 1)
    g = F.affine_grid(p, x.size(), align_corners=True)
    r = F.grid_sample(x, g, align_corners=True)
    c2 = torch.cos(2 * th).view(-1, 1, 1)
    s2 = torch.sin(2 * th).view(-1, 1, 1)
    return torch.stack([r[:, 0] * c2 + r[:, 1] * s2,
                        -r[:, 0] * s2 + r[:, 1] * c2], 1)


def _align_theta(pred, gt, mask_t):
    """Per-sample optimal rotation (deg) aligning gt to pred.

    Coarse 5-deg full circle + fine 1-deg within +-6 deg of the coarse
    optimum.  pred/gt: (N,2,128,128) on GPU; mask_t: (1,1,128,128) bool.
    """
    n = pred.shape[0]
    mask_n = mask_t.float()
    denom = mask_n.sum()
    best_v = torch.full((n,), np.inf, device=pred.device)
    best_t = torch.zeros(n, device=pred.device)
    for angs, fine in ((np.arange(0.0, 360.0, 5.0), False),
                       (np.arange(-6.0, 6.01, 1.0), True)):
        for th_deg in angs:
            th = torch.full((n,), np.deg2rad(th_deg),
                            device=pred.device)
            rt = _rot_torch(gt, th)
            mse = (((rt - pred) ** 2) * mask_n).sum((1, 2, 3)) / denom
            if fine:
                near = (th_deg - best_t).abs() <= 6.0
                better = (mse < best_v) & near
            else:
                better = mse < best_v
            best_v = torch.where(better, mse, best_v)
            best_t = torch.where(better, torch.full_like(best_t, th_deg),
                                 best_t)
    return best_t.cpu().numpy()


def _band_loss(pred, gt, mask_t, tau_lo, tau_hi, w_mid):
    """Complex MSE on hi band + MODULUS MSE on lo band + mid (relaxed)."""
    amp_g = (gt ** 2).sum(1).sqrt()
    amp_p = (pred ** 2).sum(1).sqrt()
    mx = amp_g.amax(dim=(1, 2), keepdim=True)
    m = mask_t.squeeze()
    hi = (amp_g > tau_hi * mx) & m
    lo = ((amp_g < tau_lo * mx) & m)
    mid = m & ~(hi | lo)
    dc = ((pred - gt) ** 2).sum(1)
    dm = (amp_p - amp_g) ** 2
    eps = 1e-8
    l_hi = (dc * hi).sum() / (hi.sum() + eps)
    l_lo = (dm * lo).sum() / (lo.sum() + eps)
    l_mid = (dc * mid).sum() / (mid.sum() + eps)
    return l_hi + l_lo + w_mid * l_mid


def _param_groups(net, lr):
    """Backbone-only finetune; STNs frozen (canonical frame coupled)."""
    for n_, p in net.named_parameters():
        p.requires_grad_(n_.startswith("backbone"))
    return [{"params": [p for n_, p in net.named_parameters()
                        if p.requires_grad], "lr": lr}]


def _distill_set(net, ths, imgs, window, device):
    """Behavior anchors: base outputs on flip neighborhoods (+-window,
    pins the morph zone) and on held-out shapes (anti-forgetting)."""
    idxs = np.unique(np.concatenate(
        [np.where(np.abs(ths - f) <= window)[0][::4] for f in FLIPS]))
    gt_fam = _infer_all(net, imgs[idxs], device)
    gt_fam = np.stack([np.stack([g.real, g.imag]) for g in gt_fam])
    x_fam = imgs[idxs].astype(np.float32)
    ho = np.stack([shapes.shape_to_image(mk())[..., 0] for _, mk in HELD_OUT])
    gt_ho = _infer_all(net, ho, device)
    gt_ho = np.stack([np.stack([g.real, g.imag]) for g in gt_ho])
    x = np.concatenate([x_fam, ho]).astype(np.float32)
    gt = np.concatenate([gt_fam, gt_ho])
    return x, gt


def _infer_all(net, imgs, device, batch=32):
    """Complex fields for a stack of uint8 images (batched, no grad)."""
    net = net.to(device).eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(imgs), batch):
            x = (torch.tensor(imgs[lo:lo + batch].astype(np.float32))[:, None]
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


def _band_masks(amp, mask):
    lo = mask & (amp < TAU_LO * amp.max())
    hi = mask & (amp > TAU_HI * amp.max())
    return lo, hi


TAU_LO, TAU_HI = 0.25, 0.55


def _report(net, base, ths, cf, imgs, hf_v1, device):
    """3-way table + band table + morph peaks + held-out + gate verdicts."""
    z, mask = grid.get_ghbs_grid()
    hf_v2 = _infer_all(net, imgs, device)
    hf_base = _infer_all(base, imgs, device)
    rows = []
    print(f"{'psi':>6} | {'dmod':>16} | {'dmod_reg':>14} | "
          f"{'phi_v2':>6} | {'dmod_lo':>13} | {'dmod_hi':>13}")
    print("       |   base    v1   v2 |  base   v2   | "
          "base -> v2   base -> v2")
    for ps in COLS:
        k = int(np.argmin(np.abs(ths - ps)))
        if np.all(np.isnan(cf[k])):
            print(f"{ps:6.1f} | classic NaN")
            continue
        c = cf[k].astype(complex)
        ampc = np.abs(c)
        lob, hib = _band_masks(ampc, mask)
        fields = {"base": hf_base[k], "v2": hf_v2[k]}
        if hf_v1 is not None:
            fields["v1"] = hf_v1[k].astype(complex)
        dm = {t: _d_mod(ampc, np.abs(f), mask) for t, f in fields.items()}
        _pb, rb = _reg_mod(np.abs(fields["base"]), ampc, z, mask)
        p2, r2 = _reg_mod(np.abs(fields["v2"]), ampc, z, mask)
        rows.append({"psi": ps, "dm": dm, "rb": rb, "r2": r2,
                     "lo": (_d_mod(np.abs(fields["base"])[lob], ampc[lob],
                                   np.ones(int(lob.sum()), dtype=bool)),
                           _d_mod(np.abs(fields["v2"])[lob], ampc[lob],
                                  np.ones(int(lob.sum()), dtype=bool))),
                     "hi": (_d_mod(np.abs(fields["base"])[hib], ampc[hib],
                                   np.ones(int(hib.sum()), dtype=bool)),
                            _d_mod(np.abs(fields["v2"])[hib], ampc[hib],
                                   np.ones(int(hib.sum()), dtype=bool))),
                     "p2": p2})
        v1s = f"{dm['v1']:.3f}" if "v1" in dm else "  -"
        print(f"{ps:6.1f} | {dm['base']:.3f} {v1s} {dm['v2']:.3f} | "
              f"{rb:.3f} {r2:.3f} | {p2:6.0f} | "
              f"{rows[-1]['lo'][0]:.3f} -> {rows[-1]['lo'][1]:.3f}   "
              f"{rows[-1]['hi'][0]:.3f} --gt {rows[-1]['hi'][1]:.3f}")
    # morph peaks: adjacent d2 within +-5 deg of each flip (v2 fields)
    print("morph peaks (v2 adjacent-d near flips; gate <= 0.088):")
    peaks = {}
    for f in FLIPS:
        near = np.where(np.abs(ths - f) <= 5.0)[0]
        vals = [ops.d2(hf_v2[i], hf_v2[i + 1], z, mask)
                for i in near[:-1]]
        peaks[f] = max(vals) if vals else np.nan
        print(f"  flip {f:6.2f}: peak {peaks[f]:.4f}")
    # held-out sanity
    print("held-out shapes (d_modulus base -> v2):")
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
    # gate verdicts
    rb_mean = np.mean([r["rb"] for r in rows])
    r2_mean = np.mean([r["r2"] for r in rows])
    lo_imp = np.mean([r["lo"][1] - r["lo"][0] for r in rows])
    hi_imp = np.mean([r["hi"][1] - r["hi"][0] for r in rows])
    worsen = [r["psi"] for r in rows if r["r2"] > r["rb"] + 0.02]
    gates = {
        "morph<=0.088": bool(np.nanmax(list(peaks.values())) <= 0.088),
        "held-out<=1.25x": None,  # filled by caller if needed
        "reg improved>=15%": bool(r2_mean <= 0.85 * rb_mean),
        "no column worsens": not worsen,
        "lo band improves": bool(lo_imp <= 0),
        "hi band improves": bool(hi_imp <= 0),
        "rotation bounded": all(min(r["p2"], 360 - r["p2"],
                                    abs(r["p2"] - 180)) <= 45
                                for r in rows),
    }
    print("gates:", {k: v for k, v in gates.items() if v is not None})
    print(f"mean dmod_reg base {rb_mean:.4f} -> v2 {r2_mean:.4f} "
          f"({(1 - r2_mean / rb_mean) * 100:.0f}% down)")
    return gates, rows, peaks


def _quick_eval(net, ths, cf, imgs, device, z, mask):
    """Mean registered dmod over the 11 columns (per-epoch gate metric)."""
    hf = _infer_all(net, imgs, device)
    vals = []
    for ps in COLS:
        k = int(np.argmin(np.abs(ths - ps)))
        if np.all(np.isnan(cf[k])):
            continue
        vals.append(_reg_mod(np.abs(hf[k]), np.abs(cf[k].astype(complex)),
                             z, mask)[1])
    return float(np.mean(vals))


def _train(net, args, imgs, gts, sel, distill, mask_t, ths, cf, z,
           mask, device):
    net = net.to(device)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    groups = _param_groups(net, args.lr)
    opt = torch.optim.Adam(groups, lr=args.lr)
    anchor0 = ([q.detach().clone() for q in groups[0]["params"]]
               if args.anchor > 0 else None)
    x_all = (torch.tensor(imgs[sel].astype(np.float32))[:, None].to(device)
             / 255.0)
    gt_all = torch.tensor(gts[sel], dtype=torch.float32).to(device)
    if distill is not None:
        x_d = (torch.tensor(distill[0].astype(np.float32))[:, None]
               .to(device) / 255.0)
        gt_d = torch.tensor(distill[1], dtype=torch.float32).to(device)
    best_score, best_state, wait = np.inf, None, 0
    for epoch in range(args.epochs):
        net.eval()
        with torch.no_grad():
            pred_all = torch.cat([
                net(x_all[lo:lo + 64])
                for lo in range(0, len(sel), 64)])
            theta = _align_theta(pred_all, gt_all, mask_t)
            th_rad = torch.tensor(np.deg2rad(theta), device=device)
            gt_alg = torch.cat([
                _rot_torch(gt_all[lo:lo + 64], th_rad[lo:lo + 64])
                for lo in range(0, len(sel), 64)])
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        net.train()
        perm = torch.randperm(len(sel), device=device)
        losses = []
        for lo in range(0, len(sel), args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            pred = net(x_all[idx])
            loss = _band_loss(pred, gt_alg[idx], mask_t,
                              args.tau_lo, args.tau_hi, args.w_mid)
            if x_d is not None:
                j = torch.randint(
                    0, len(x_d), (min(args.batch_size, len(x_d)),),
                    device=device)
                loss = loss + args.distill_rate * torch.mean(
                    (net(x_d[j]) - gt_d[j]) ** 2)
            if anchor0 is not None:
                a = torch.zeros((), device=device)
                for q, q0 in zip(groups[0]["params"], anchor0, strict=True):
                    a = a + torch.sum((q - q0) ** 2)
                loss = loss + args.anchor * a
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        score = _quick_eval(net, ths, cf, imgs, device, z, mask)
        print(f"  epoch {epoch + 1}/{args.epochs}: band-loss "
              f"{np.mean(losses):.6f}  reg-dmod {score:.4f}", flush=True)
        if score < best_score:
            best_score, wait = score, 0
            best_state = {k: v.cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop at epoch {epoch + 1}")
                break
    net.load_state_dict(best_state)
    net.cpu().eval()
    return float(best_score)


def main() -> None:
    p = argparse.ArgumentParser(description="osc-family HBSN finetune v2")
    p.add_argument("--tag", default="v2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--distill-window", type=float, default=5.0)
    p.add_argument("--tau-lo", type=float, default=0.25)
    p.add_argument("--tau-hi", type=float, default=0.55)
    p.add_argument("--anchor", type=float, default=1e-3)
    p.add_argument("--distill-rate", type=float, default=1.0)
    p.add_argument("--w-mid", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precache", action="store_true")
    p.add_argument("--eval-only")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = np.load(FIELD_CACHE)
    ths, cf = d["ths"], d["cf"]
    imgs = _images(ths, force=args.precache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z, mask = grid.get_ghbs_grid()
    mask_t = torch.tensor(mask)[None, None].to(device)
    base = nets.build_net(nets.BEST_CKPT)
    if args.eval_only:
        net = nets.build_net(args.eval_only)
        hf_v1 = None
        if os.path.exists(V1_CACHE):
            hf_v1 = np.load(V1_CACHE)["hf"]
        _report(net, base, ths, cf, imgs, hf_v1, device)
        return
    finite = np.array([not np.all(np.isnan(c)) for c in cf])
    sel_fit = np.where(finite & ~np.logical_or.reduce(
        [np.abs(ths - f) <= args.distill_window for f in FLIPS]))[0]
    sel_fit = sel_fit[::args.stride]
    sel_d = np.unique(np.concatenate(
        [np.where((np.abs(ths - f) <= args.distill_window) & finite)[0][::4]
         for f in FLIPS]))
    gts = np.stack([np.stack([c.real, c.imag]) for c in cf])
    print(f"fit samples: {len(sel_fit)}, distill family rows: {len(sel_d)}",
          flush=True)
    net = nets.build_net(nets.BEST_CKPT)
    distill = _distill_set(net, ths, imgs, args.distill_window, device)
    best = _train(net, args, imgs, gts, sel_fit, distill, mask_t, ths, cf,
                  z, mask, device)
    out = os.path.join(OUT_DIR, args.tag, "best.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "config": {"net": dict(net.config), "dataset": {}}}, out)
    print(f"saved: {out} (best reg-dmod {best:.4f})")
    hf_v1 = None
    if os.path.exists(V1_CACHE):
        hf_v1 = np.load(V1_CACHE)["hf"]
    _report(net, base, ths, cf, imgs, hf_v1, device)


if __name__ == "__main__":
    main()
