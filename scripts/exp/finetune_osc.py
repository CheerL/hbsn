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

import osc_family as OF
from hbsn_exp import classic, grid, nets, ops, shapes

H_SLIDE, AMP = OF.H_SLIDE, OF.AMP
# measured seam midpoints for THIS family (see build_gt_arcs)
FLIPS = [30.88, 149.12, 246.88, 293.12]
_CACHE = os.path.join(REPO, "runs", "cache_osc")
FIELD_CACHE = os.path.join(_CACHE, f"fields_osc_{OF.FAMILY}.npz")
IMG_CACHE = os.path.join(_CACHE, f"imgs_osc_{OF.FAMILY}.npz")
V1_CACHE = os.path.join(_CACHE, "fields_osc_ftv1.npz")
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
OUT_DIR = os.path.join(REPO, "runs", "finetuned", "osc")
COLS = sorted(
    {90.0, 180.0, 270.0} | {f - 0.1 for f in FLIPS} | {f + 0.1 for f in FLIPS}
)
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
    return OF.bound(psi)


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
    p = torch.stack(
        [torch.stack([cos, sin, zero], 1), torch.stack([-sin, cos, zero], 1)], 1
    )
    g = F.affine_grid(p, x.size(), align_corners=True)
    r = F.grid_sample(x, g, align_corners=True)
    c2 = torch.cos(2 * th).view(-1, 1, 1)
    s2 = torch.sin(2 * th).view(-1, 1, 1)
    return torch.stack(
        [r[:, 0] * c2 + r[:, 1] * s2, -r[:, 0] * s2 + r[:, 1] * c2], 1
    )


def _align_theta(pred, gt, mask_t, branch=False, fine_step=1.0):
    """Per-sample optimal rotation (deg) aligning gt to pred.

    Full mode: coarse 5-deg circle + fine search within +-6 deg.  Branch
    mode: theta restricted to {0, 180} -- the seam is absorbed while the
    orientation stays anchored to the classical frame.
    """
    n = pred.shape[0]
    mask_n = mask_t.float()
    denom = mask_n.sum()
    best_v = torch.full((n,), np.inf, device=pred.device)
    best_t = torch.zeros(n, device=pred.device)
    if branch:
        stages = [(np.array([0.0, 180.0]), False)]
    else:
        stages = [
            (np.arange(0.0, 360.0, 5.0), False),
            (np.arange(-6.0, 6.01, fine_step), True),
        ]
    for angs, fine in stages:
        for th_deg in angs:
            th = torch.full((n,), np.deg2rad(th_deg), device=pred.device)
            rt = _rot_torch(gt, th)
            mse = (((rt - pred) ** 2) * mask_n).sum((1, 2, 3)) / denom
            if fine:
                near = (th_deg - best_t).abs() <= 6.0
                better = (mse < best_v) & near
            else:
                better = mse < best_v
            best_v = torch.where(better, mse, best_v)
            best_t = torch.where(
                better, torch.full_like(best_t, th_deg), best_t
            )
    return best_t.cpu().numpy()


def _band_loss(pred, gt, mask_t, tau_lo, tau_hi, w_mid):
    """Complex MSE on hi band + MODULUS MSE on lo band + mid (relaxed)."""
    amp_g = (gt**2).sum(1).sqrt()
    amp_p = (pred**2).sum(1).sqrt()
    mx = amp_g.amax(dim=(1, 2), keepdim=True)
    m = mask_t.squeeze()
    hi = (amp_g > tau_hi * mx) & m
    lo = (amp_g < tau_lo * mx) & m
    mid = m & ~(hi | lo)
    dc = ((pred - gt) ** 2).sum(1)
    dm = (amp_p - amp_g) ** 2
    eps = 1e-8
    l_hi = (dc * hi).sum() / (hi.sum() + eps)
    l_lo = (dm * lo).sum() / (lo.sum() + eps)
    l_mid = (dc * mid).sum() / (mid.sum() + eps)
    return l_hi + l_lo + w_mid * l_mid


def _param_groups(net, lr, rn_rate=0.0, pre_rate=0.0, bb_rate=1.0):
    """backbone at lr*bb_rate; post_STN/pre_STN at lr*rate (0 = frozen)."""
    groups = []
    for prefix, rate in (
        ("backbone", bb_rate),
        ("post_stn", rn_rate),
        ("pre_stn", pre_rate),
    ):
        params = [
            p for n_, p in net.named_parameters() if n_.startswith(prefix)
        ]
        if rate == 0:
            for p in params:
                p.requires_grad_(False)
            continue
        groups.append({"params": params, "lr": lr * rate})
    if not groups:
        raise ValueError(
            "no trainable parameters: backbone/post_STN/pre_STN are all "
            "frozen (bb_rate, rn_rate, pre_rate <= 0)"
        )
    return groups


def _distill_set(net, ths, imgs, window, device):
    """Behavior anchors: base outputs on flip neighborhoods (+-window,
    pins the morph zone) and on held-out shapes (anti-forgetting).

    `window = 0` keeps ONLY anti-forgetting, which is the right setting
    for the ideal-continuous-gt route: pinning the model to base AT the
    seams pins it to the very jump the gt exists to remove.  (An empty
    flip set used to crash np.stack, hence the guard.)
    """
    xs, gts = [], []
    if window > 0:
        idxs = np.unique(
            np.concatenate(
                [np.where(np.abs(ths - f) <= window)[0][::4] for f in FLIPS]
            )
        )
        gt_fam = _infer_all(net, imgs[idxs], device)
        xs.append(imgs[idxs].astype(np.float32))
        gts.append(np.stack([np.stack([g.real, g.imag]) for g in gt_fam]))
    ho = np.stack([shapes.shape_to_image(mk())[..., 0] for _, mk in HELD_OUT])
    gt_ho = _infer_all(net, ho, device)
    xs.append(ho)
    gts.append(np.stack([np.stack([g.real, g.imag]) for g in gt_ho]))
    return np.concatenate(xs).astype(np.float32), np.concatenate(gts)


def _infer_all(net, imgs, device, batch=32):
    """Complex fields for a stack of uint8 images (batched, no grad)."""
    net = net.to(device).eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(imgs), batch):
            x = (
                torch.tensor(imgs[lo : lo + batch].astype(np.float32))[
                    :, None
                ].to(device)
                / 255.0
            )
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
    print(
        f"{'psi':>6} | {'dmod':>16} | {'dmod_reg':>14} | "
        f"{'phi_v2':>6} | {'dmod_lo':>13} | {'dmod_hi':>13}"
    )
    print("       |   base    v1   v2 |  base   v2   | base -> v2   base -> v2")
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
        rows.append(
            {
                "psi": ps,
                "dm": dm,
                "rb": rb,
                "r2": r2,
                "lo": (
                    _d_mod(
                        np.abs(fields["base"])[lob],
                        ampc[lob],
                        np.ones(int(lob.sum()), dtype=bool),
                    ),
                    _d_mod(
                        np.abs(fields["v2"])[lob],
                        ampc[lob],
                        np.ones(int(lob.sum()), dtype=bool),
                    ),
                ),
                "hi": (
                    _d_mod(
                        np.abs(fields["base"])[hib],
                        ampc[hib],
                        np.ones(int(hib.sum()), dtype=bool),
                    ),
                    _d_mod(
                        np.abs(fields["v2"])[hib],
                        ampc[hib],
                        np.ones(int(hib.sum()), dtype=bool),
                    ),
                ),
                "p2": p2,
            }
        )
        v1s = f"{dm['v1']:.3f}" if "v1" in dm else "  -"
        print(
            f"{ps:6.1f} | {dm['base']:.3f} {v1s} {dm['v2']:.3f} | "
            f"{rb:.3f} {r2:.3f} | {p2:6.0f} | "
            f"{rows[-1]['lo'][0]:.3f} -> {rows[-1]['lo'][1]:.3f}   "
            f"{rows[-1]['hi'][0]:.3f} --gt {rows[-1]['hi'][1]:.3f}"
        )
    # morph peaks: adjacent d2 within +-5 deg of each flip (v2 fields)
    print("morph peaks (v2 adjacent-d near flips; gate <= 0.088):")
    peaks = {}
    for f in FLIPS:
        near = np.where(np.abs(ths - f) <= 5.0)[0]
        vals = [ops.d2(hf_v2[i], hf_v2[i + 1], z, mask) for i in near[:-1]]
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
        print(
            f"  {name}: {_d_mod(np.abs(c), np.abs(a), mask):.4f} -> "
            f"{_d_mod(np.abs(c), np.abs(b), mask):.4f}"
        )
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
        "rotation bounded": all(
            min(r["p2"], 360 - r["p2"], abs(r["p2"] - 180)) <= 45 for r in rows
        ),
    }
    print("gates:", {k: v for k, v in gates.items() if v is not None})
    print(
        f"mean dmod_reg base {rb_mean:.4f} -> v2 {r2_mean:.4f} "
        f"({(1 - r2_mean / rb_mean) * 100:.0f}% down)"
    )
    return gates, rows, peaks


def _quick_eval(net, ths, ref, imgs, device, z, mask, ref_ok=None):
    """Mean F on the radially weighted unit disk over the 11 columns.

    Per-epoch early-stop score: fidelity to THE TARGET THE NET IS BEING
    TRAINED ON, measured with the eval_osc_standard operator (same unit-disk
    mask, fixed origin/radius, same radial weight).  Keep these in sync with
    the audit or the checkpoint we pick is not the one the audit would pick.

    `ref` is that target, NOT necessarily the raw classical field.  Under
    --gt-cache the target is the ideal continuous gt, whose seams no longer
    jump; scoring against raw cf there would select the "best" checkpoint by
    a standard the run is not optimising.
    """
    from eval_osc_standard import _best_theta, _radial_weight

    w = _radial_weight(mask, z)
    hf = _infer_all(net, imgs, device)
    vals = []
    for ps in COLS:
        k = int(np.argmin(np.abs(ths - ps)))
        if ref_ok is not None and not ref_ok[k]:
            continue
        if np.all(np.isnan(ref[k])):
            continue
        amp = np.abs(hf[k]).astype(np.float32)
        vals.append(_best_theta(ref[k].astype(complex), amp, w, z, mask)[1])
    return float(np.mean(vals))


def _smooth_penalty(pred, gt, mask_t, tau_lo):
    """Scale-free high-frequency penalty on the modulus, blue core only.

    lap(|pred|)^2 summed over the low-amplitude band, normalized by the
    band's own modulus energy -- dimensionless, so one weight transfers
    across runs.  Targets the "core looks more scattered" failure, which
    a plain RMS-to-classical loss does not see.
    """
    pm = (pred**2).sum(1).sqrt()
    gm = (gt**2).sum(1).sqrt()
    mx = gm.amax(dim=(1, 2), keepdim=True)
    band = (gm < tau_lo * mx) & mask_t[:, 0]
    lap = (
        pm[:, :-2, 1:-1]
        + pm[:, 2:, 1:-1]
        + pm[:, 1:-1, :-2]
        + pm[:, 1:-1, 2:]
        - 4 * pm[:, 1:-1, 1:-1]
    )
    core = band[:, 1:-1, 1:-1]
    return (lap**2 * core).sum() / ((pm**2 * band).sum() + 1e-8)


def _train_mode_frozen_bn(net):
    """train() everywhere EXCEPT BatchNorm, whose running stats must stay
    frozen.  Do NOT inline this as two separate statements: net.train()
    after the BN loop silently un-freezes it and the running stats get
    clobbered by batch-size-8 statistics, which destroys a converged net
    at any lr (measured: circle 0.035 -> 0.11 in one epoch).
    """
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def _train(
    net,
    args,
    imgs,
    gts,
    sel,
    distill,
    mask_t,
    ths,
    cf,
    z,
    mask,
    device,
    ref=None,
    ref_ok=None,
):
    """ref/ref_ok: what _quick_eval scores against (default: the raw
    classical field, i.e. the pre---gt-cache behaviour)."""
    net = net.to(device)
    groups = _param_groups(
        net, args.lr, args.rn_rate, args.pre_rate, args.bb_rate
    )
    opt = torch.optim.Adam(groups, lr=args.lr)
    anchor0 = (
        [[q.detach().clone() for q in g["params"]] for g in groups]
        if args.anchor > 0
        else None
    )
    x_all = (
        torch.tensor(imgs[sel].astype(np.float32))[:, None].to(device) / 255.0
    )
    gt_all = torch.tensor(gts[sel], dtype=torch.float32).to(device)
    if distill is not None:
        x_d = (
            torch.tensor(distill[0].astype(np.float32))[:, None].to(device)
            / 255.0
        )
        gt_d = torch.tensor(distill[1], dtype=torch.float32).to(device)
    best_score, best_state, wait = np.inf, None, 0
    for epoch in range(args.epochs):
        net.eval()
        with torch.no_grad():
            if args.align == "none":
                gt_alg = gt_all
            else:
                pred_all = torch.cat(
                    [net(x_all[lo : lo + 64]) for lo in range(0, len(sel), 64)]
                )
                theta = _align_theta(
                    pred_all, gt_all, mask_t, branch=args.align == "branch"
                )
                th_rad = torch.tensor(np.deg2rad(theta), device=device)
                gt_alg = torch.cat(
                    [
                        _rot_torch(gt_all[lo : lo + 64], th_rad[lo : lo + 64])
                        for lo in range(0, len(sel), 64)
                    ]
                )
        # net.train() first, THEN freeze BN -- see _train_mode_frozen_bn
        _train_mode_frozen_bn(net)
        perm = torch.randperm(len(sel), device=device)
        losses, sms = [], []
        for lo in range(0, len(sel), args.batch_size):
            idx = perm[lo : lo + args.batch_size]
            pred = net(x_all[idx])
            if args.loss_mode == "stock":
                # the network's own objective: gt first goes through
                # post_STN, plus stn_rate/grad_rate terms.  Kept as the
                # reference "original recipe" comparison.
                ld, _ = net.loss(pred, gt_alg[idx])
                loss = ld["loss"]
            elif args.direct or args.align != "none":
                # bypass net.loss: its internal gt->RN re-rotates a
                # pre-aligned teacher (measured d 0.06-0.27), and the
                # explicit stn term is what we want to control here
                m = mask_t.float()
                loss = (((pred - gt_alg[idx]) ** 2) * m).sum(
                    (1, 2, 3)
                ).sum() / m.sum()
                loss = loss + 0.1 * F.mse_loss(
                    net.post_stn(pred)[0] if net.post_stn is not None else pred,
                    pred,
                )
            else:
                loss = _band_loss(
                    pred,
                    gt_alg[idx],
                    mask_t,
                    args.tau_lo,
                    args.tau_hi,
                    args.w_mid,
                )
            if args.smooth > 0:
                sm = _smooth_penalty(pred, gt_alg[idx], mask_t, args.tau_lo)
                loss = loss + args.smooth * sm
                sms.append(float(sm.item()))
            if x_d is not None:
                j = torch.randint(
                    0,
                    len(x_d),
                    (min(args.batch_size, len(x_d)),),
                    device=device,
                )
                loss = loss + args.distill_rate * torch.mean(
                    (net(x_d[j]) - gt_d[j]) ** 2
                )
            if anchor0 is not None:
                a = torch.zeros((), device=device)
                for g, g0 in zip(groups, anchor0, strict=True):
                    for q, q0 in zip(g["params"], g0, strict=True):
                        a = a + torch.sum((q - q0) ** 2)
                loss = loss + args.anchor * a
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        score = _quick_eval(
            net,
            ths,
            ref if ref is not None else cf,
            imgs,
            device,
            z,
            mask,
            ref_ok,
        )
        extra = f"  smooth {np.mean(sms):.4f}" if sms else ""
        print(
            f"  epoch {epoch + 1}/{args.epochs}: loss "
            f"{np.mean(losses):.6f}  D_free {score:.4f}{extra}",
            flush=True,
        )
        # Save EVERY epoch, not just the D_free-best one: D_free and the
        # flip-window morph peak are NOT co-monotone (measured on v10/v11:
        # D_free keeps falling while morph regresses past base's 0.0622).
        # Without these we cannot tell whether some intermediate epoch
        # dominates the final one -- early stop never fired on v10/v11,
        # so their "best" is simply the last epoch.
        if args.save_epochs:
            ep = os.path.join(OUT_DIR, args.tag, f"ep{epoch + 1}.pth")
            os.makedirs(os.path.dirname(ep), exist_ok=True)
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "config": {"net": dict(net.config), "dataset": {}},
                },
                ep,
            )
        if score < best_score:
            best_score, wait = score, 0
            best_state = {
                k: v.cpu().clone() for k, v in net.state_dict().items()
            }
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
    p.add_argument(
        "--init",
        default=nets.BEST_CKPT,
        help="starting checkpoint (default: the paper model)",
    )
    p.add_argument(
        "--direct",
        action="store_true",
        help="use direct masked MSE + 0.1 stn instead of net.loss",
    )
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
    p.add_argument(
        "--rn-rate",
        type=float,
        default=0.0,
        help="post_STN lr multiplier (co-adaptation)",
    )
    p.add_argument(
        "--align",
        choices=["full", "branch", "none"],
        default="full",
        help="teacher rotation planning; none = raw classical "
        "target (v1 route, net.loss's internal gt->RN is "
        "then correct)",
    )
    p.add_argument(
        "--loss",
        choices=["band", "plain"],
        default="band",
        help="fit loss: 3-band weighted or plain masked MSE",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="weight of the scale-free blue-core Laplacian penalty (0 = off)",
    )
    p.add_argument(
        "--pre-rate",
        type=float,
        default=0.0,
        help="pre_STN lr multiplier (0 = frozen)",
    )
    p.add_argument(
        "--bb-rate",
        type=float,
        default=1.0,
        help="backbone lr multiplier (0 = freeze backbone entirely and "
        "train only the STN heads; the from-base divergence is a "
        "backbone phenomenon, so this tests whether the 6900 backbone "
        "weights moving at all is what destroys it)",
    )
    p.add_argument(
        "--gt-cache",
        default=None,
        help="npz from ideal_gt_osc.py: train on the ideal continuous gt "
        "with NO flip-window exclusion (the target is already smooth)",
    )
    p.add_argument(
        "--loss-mode",
        choices=["mse", "stock"],
        default="mse",
        help="mse = plain masked MSE on net(x) vs gt (no RN, no extras); "
        "stock = the network's own net.loss, which re-rotates gt through "
        "post_STN and adds gradient-matching terms",
    )
    p.add_argument(
        "--fit-window",
        type=float,
        default=None,
        help="deg around each flip excluded from the FIT target "
        "(default: same as --distill-window).  Set smaller than "
        "--distill-window to keep the anchor wide while still giving "
        "the fit data just outside the flip",
    )
    p.add_argument(
        "--anchor-ckpt",
        default=None,
        help="teacher for the flip-window behavior anchor (default: the "
        "BASE model, NOT --init -- see the note in main())",
    )
    p.add_argument("--patience", type=int, default=2)
    p.add_argument(
        "--save-epochs",
        action="store_true",
        help="also save ep<N>.pth each epoch (D_free and morph peak "
        "are not co-monotone, so the D_free-best ckpt may not be "
        "the morph-best one)",
    )
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
    # what _quick_eval scores against; --gt-cache swaps it for the target
    ref, ref_ok = cf, None
    if args.gt_cache:
        # Ideal-continuous-gt mode: the target is already continuous across
        # psi, so the flip windows need no exclusion -- that exclusion
        # existed only because the raw classical field jumps ~0.30 at each
        # seam, which would ask the net to reproduce a discontinuity.
        gd = np.load(args.gt_cache)
        gt_fields = gd["gt"].astype(complex)
        gvalid = gd["valid"]
        finite = finite & gvalid
        ref, ref_ok = gt_fields, gvalid
        sel_fit = np.where(finite)[0][:: args.stride]
        fit_win = 0.0
        gts = np.stack([np.stack([c.real, c.imag]) for c in gt_fields])
        print(f"ideal gt: {args.gt_cache} (width={gd['width']})", flush=True)
    else:
        # Two windows, deliberately decoupled.  `--distill-window` is the
        # ANCHOR region (kept wide, pinned to base).  `--fit-window` is how
        # much of that region is also withheld from the fit target.  They
        # were the same number before, which forced a bad trade: a wide
        # exclusion leaves the flip window with no fit signal at all (F
        # regressions cluster at 0-4 deg from each flip), while a narrow
        # one lets the fit pull the window away from base.  Letting fit
        # data in close to the flip while the anchor still covers it is
        # what breaks the tie.
        fit_win = (
            args.fit_window
            if args.fit_window is not None
            else args.distill_window
        )
        sel_fit = np.where(
            finite
            & ~np.logical_or.reduce([np.abs(ths - f) <= fit_win for f in FLIPS])
        )[0]
        sel_fit = sel_fit[:: args.stride]
        gts = np.stack([np.stack([c.real, c.imag]) for c in cf])
    sel_d = np.unique(
        np.concatenate(
            [
                np.where((np.abs(ths - f) <= args.distill_window) & finite)[0][
                    ::4
                ]
                for f in FLIPS
            ]
        )
    )
    print(
        f"fit samples: {len(sel_fit)}, distill family rows: {len(sel_d)}",
        flush=True,
    )
    net = nets.build_net(args.init)
    # The flip-window teacher must be a model that HOLDS still there.
    # `sel_fit` excludes every psi within +-distill_window of a flip, so
    # the flip window has NO fit data at all -- the anchor is its only
    # constraint.  Passing `net` (the thing being optimised) makes that
    # "anchor" chase its own moving output, which is how v10/v11 ended up
    # 1.2x worse than base on the flip peak: base's peak is the classical
    # seam edge sitting at +-1.26 deg, v10's has slid to +-2.24 deg with
    # excess spread over |off| 1-4 deg.  Anchor to `base` unless a teacher
    # is pinned explicitly.
    teacher = nets.build_net(args.anchor_ckpt) if args.anchor_ckpt else base
    distill = _distill_set(teacher, ths, imgs, args.distill_window, device)
    if teacher is not base:
        print(f"flip-window anchor teacher: {args.anchor_ckpt}", flush=True)
    best = _train(
        net,
        args,
        imgs,
        gts,
        sel_fit,
        distill,
        mask_t,
        ths,
        cf,
        z,
        mask,
        device,
        ref=ref,
        ref_ok=ref_ok,
    )
    out = os.path.join(OUT_DIR, args.tag, "best.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": {"net": dict(net.config), "dataset": {}},
        },
        out,
    )
    print(f"saved: {out} (best reg-dmod {best:.4f})")
    hf_v1 = None
    if os.path.exists(V1_CACHE):
        hf_v1 = np.load(V1_CACHE)["hf"]
    _report(net, base, ths, cf, imgs, hf_v1, device)


if __name__ == "__main__":
    main()
