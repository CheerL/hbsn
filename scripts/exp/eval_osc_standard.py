#!/usr/bin/env python3
"""Osc-family evaluation standard: F (fidelity) + C (continuity) + O (frame).

F and C measure the field's shape and its continuity.  Both are exactly
invariant to a GLOBAL 180 deg rotation of the model, so neither can see a
frame error: F removes each psi's best rotation, and R_180 is an isometry
so continuity is untouched.  O closes that hole by pinning the candidate
to base's absolute frame.

Two axes only.  Smoothness is deliberately NOT scored: a field can be
locally smooth and still be "crinkled silk" (many shallow folds), and the
earlier Laplacian metric measured exactly that useless notion.  What the
model should reproduce is a simple surface, which is what F and C test.

O -- absolute frame (see ORIENT_TOL for why base, not classical, is the
    reference, and for the measured case that motivated adding it).

Domain and radial weighting.
    HBS lives on the unit disk (the same fixed mask the training loss
    uses, `hbsn.nets.hbsn.HBSNet.create_mask` = x^2+y^2<=1).  Outside it
    the render is flat `masked_fill` blue and rotation-invariant, so it is
    excluded first -- that is the "outer ring" to drop, NOT an amplitude
    threshold.

    Inside, weight is NOT uniform: full weight for radius <= R_CORE, then
    a linear taper down to W_EDGE at radius 1.  Reason (measured): the
    classical HBS puts ALL of its roughness in the thin rim
    (|z| in [0.95,0.99] has Laplacian RMS 0.81, |z|>=0.99 reaches 1.17
    with mean |B| = 0, versus 0.003-0.021 for |z| <= 0.9).  A uniform
    disk average therefore scores the rim artifact, not the model.

F -- fidelity to the classical field, rotation removed.
    theta* = argmin_theta wRMS_w(|h|, |R_theta cf|);  F = wRMS at theta*.
    Removing each psi's single best rotation makes F a pure SHAPE
    distance, so a model is never penalised for being in a different
    orientation frame, and never rewarded for one either.  theta*(psi) is
    reported for information only -- it is NOT a gate: theta* is
    degenerate wherever the field is near-symmetric, so an absolute
    orientation test fails even on the base model.
    The rotation operator is the project reference `ops.rotate_mu` math
    (bilinear inverse-map + phase, modulus only) -- NOT the training-time
    `_rot_torch`, whose align_corners convention differs by half a pixel
    (measured F shift ~0.02, the size of the effects under test).

C -- continuity across psi.
    d_i = wRMS(|h(psi_i)|, |h(psi_i+1)|) on the cache's native 0.25 deg
    grid.  NO rotation is removed: continuity is a property of the raw
    sequence, and removing a rotation would hide a drift.

    There are NO exceptions.  In particular the classical seam's ~180 deg
    branch relabel is the pathology, not a licence: measured, the
    classical field jumps 0.3059 across a flip but only 0.0092 after
    R_180, i.e. the shape is continuous and only the branch label flips.
    HBSN is supposed to morph through instead (verified: both base and
    its finetunes take theta* = 0 at every flip, i.e. they do NOT switch
    branch), so any gate that "allows a 180 switch" would be granting the
    classical artifact an exemption.  C1 below is an absolute test; C2 is
    a regression test against the base model.

Base fields come from the cache (fields_osc.npz); the candidate is
inferred live.  Pinned hbs==1.0.1 (HBS_PYTHON_DIR) before hbsn_exp.
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.ndimage import map_coordinates

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_osc import COLS, FLIPS, _images
from hbsn_exp import grid, nets

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
FIELD_CACHE = os.path.join(REPO, "runs/cache_osc/fields_osc.npz")
OUT_DIR = os.path.join(REPO, "runs/cache_osc")

R_CORE = 0.95  # full weight inside this radius
W_EDGE = 0.1  # weight at |z| = 1 (the rim is down-weighted, not dropped)
COARSE, FINE = 5.0, 0.5  # theta search steps (deg)
FLIP_WIN = 5.0  # deg around each classical flip for the continuity peak
JUMP = 0.20  # adjacent-d above this counts as a discontinuity
TOL = 0.005  # per-psi F regression tolerance
MIN_WIN = 0.88  # fraction of psi that must not regress
# Absolute-frame gate.  F removes each psi's best rotation and C only
# differences neighbouring psi, so BOTH are exactly invariant to a rigid
# re-framing of the model: F(R_alpha h, cf) = F(h, cf), and R_alpha is an
# isometry so every adjacent-d is unchanged.  Measured, that blind spot was
# live -- a ckpt trained on a mis-framed target scored a clean
# F 0.0442 / win 94.9% / peak 0.42x.
#
# The statistic: rotate the BASE field onto the candidate,
#   phi(psi) = argmin_theta wRMS(|R_theta base|, |cand|),
# and take the circular mean.  |R| is how consistent that offset is across
# psi; phi is its direction.  Reference must be BASE, and it must be
# applied to the FIELD.  The obvious alternative -- differencing
# theta*(cf, cand) against theta*(cf, base) -- does not work: cf is the
# CLASSICAL field, which flips branch at its own seams, so theta*(cf, base)
# swings 0 <-> 180 across the family (measured 179, 1, 14, 1, 179, ...) and
# the difference reports classical's seam motion as a frame error.  Every
# candidate "failed" that version, which is the tell.
#
# Measured floors: base 1.00; V1/V3/v10/v11/v20dr3/v22/v23/igt_base
# 0.43-0.53; the three mis-framed ckpts 0.01-0.03.  R_MIN sits in that gap.
# phi only carries information when |R| is high, so both are required: a
# clean global 180 flip gives |R| ~ 1 with phi ~ 180 and fails on phi.
ORIENT_TOL = 15.0  # deg; |phi| when the offset is consistent
FRAME_R_MIN = 0.25  # min |R|; healthy floor measured at 0.43
# The base row is read from the field cache while the candidate is
# inferred live.  Re-inferring base from its own ckpt reproduces the cache
# to 0.06% of mean |B| on average, but the flip peak is a max over a
# narrow window, where that same gap shows up as ~1% (0.0617 cached vs
# 0.0623 live).  Without this tolerance a model fails its OWN peak gate,
# so the gate would be meaningless.  2% covers the noise with margin.
PEAK_TOL = 1.02

# legacy class name kept so older imports do not break
_bands = None


def _radial_weight(mask, z):
    """Full inside R_CORE, linear taper to W_EDGE at the rim."""
    r = np.abs(z)
    t = np.clip((r - R_CORE) / (1.0 - R_CORE), 0.0, 1.0)
    return mask * (1.0 - (1.0 - W_EDGE) * t)


def _rot_mod(cf, z, mask, angles):
    """|R_theta cf| for a batch of angles -- ops.rotate_mu math, modulus.

    R_theta B(z) = e^{-2i theta} B(e^{i theta} z); the phase factor has
    unit modulus, so only the inverse-mapped re/im entries are needed.
    """
    th = np.deg2rad(np.asarray(angles, dtype=np.float64))
    wz = z[None] * np.exp(1j * th[:, None, None])
    coords = np.stack([grid.CY - wz.imag * grid.R, wz.real * grid.R + grid.CX])
    re = map_coordinates(
        cf.real.astype(np.float64), coords, order=1, mode="constant", cval=0.0
    )
    im = map_coordinates(
        cf.imag.astype(np.float64), coords, order=1, mode="constant", cval=0.0
    )
    return np.sqrt(re * re + im * im) * mask[None]


def _wrms(a, b, w):
    return float(np.sqrt(np.sum((a - b) ** 2 * w) / w.sum()))


def _frame_stats(hf_base, amp, sel, w, z, mask):
    """Circular mean of phi(psi) = angle rotating the BASE field onto the
    candidate.  Returns (phi_deg, |R|).  See ORIENT_TOL for why the
    reference is base's field and not classical."""
    ph = np.deg2rad(
        np.array([_best_theta(hf_base[i], amp[i], w, z, mask)[0] for i in sel])
    )
    zv = np.exp(1j * ph).mean()
    return float(np.rad2deg(np.angle(zv))), float(abs(zv))


def _best_theta(cf, amp_h, w, z, mask):
    """theta* (deg) and F = wRMS at theta*, coarse 5 deg + fine 0.5 deg."""

    def evals(angs):
        rm = _rot_mod(cf, z, mask, angs)
        return np.sqrt(
            ((rm - amp_h[None]) ** 2 * w[None]).sum((1, 2)) / w.sum()
        )

    coarse = evals(np.arange(0.0, 360.0, COARSE))
    b = float(np.argmin(coarse) * COARSE)
    fine = np.arange(b - COARSE, b + COARSE + FINE / 2, FINE)
    ef = evals(fine)
    j = int(np.argmin(ef))
    return float((fine[j]) % 360.0), float(ef[j])


def _adjacent_d(mods, w, lo, hi):
    """wRMS between consecutive modulus fields (steps [lo,hi))."""
    return np.array([_wrms(mods[i], mods[i + 1], w) for i in range(lo, hi)])


def _infer_all(net, imgs, device, batch=64):
    net = net.to(device).eval()
    out = np.zeros((len(imgs), 128, 128), dtype=np.complex64)
    with torch.no_grad():
        for lo in range(0, len(imgs), batch):
            x = (
                torch.tensor(imgs[lo : lo + batch].astype(np.float32))[
                    :, None
                ].to(device)
                / 255.0
            )
            p = net(x).cpu().numpy()
            out[lo : lo + batch] = p[:, 0] + 1j * p[:, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="osc evaluation standard F+C")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--ft-cache",
        default=None,
        help="reuse cached candidate fields instead of inferring",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=2,
        help="psi subsample for F (1 = every 0.25 deg)",
    )
    args = ap.parse_args()
    tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))

    d = np.load(FIELD_CACHE)
    ths, cf, hf_base = d["ths"], d["cf"], d["hf"].astype(complex)
    imgs = _images(ths)
    z, mask = grid.get_ghbs_grid()
    w = _radial_weight(mask, z)
    valid = np.array([not np.all(np.isnan(c)) for c in cf])
    cf = np.where(valid[:, None, None], cf, 0).astype(complex)
    amp_base = np.abs(hf_base).astype(np.float32)
    if args.ft_cache:
        hf = np.load(args.ft_cache)["hf"].astype(complex)
    else:
        net = nets.build_net(args.ckpt)
        print(
            f"[{tag}] inferring {len(ths)} fields from {args.ckpt}", flush=True
        )
        hf = _infer_all(
            net, imgs, "cuda" if torch.cuda.is_available() else "cpu"
        )
    amp_ft = np.abs(hf).astype(np.float32)
    sel = list(np.where(valid)[0][:: args.stride])
    for ps in COLS:  # always keep the display columns
        k = int(np.argmin(np.abs(ths - ps)))
        if valid[k] and k not in sel:
            sel.append(k)
    sel = sorted(sel)
    print(
        f"[{tag}] valid psi {int(valid.sum())}/{len(ths)}; "
        f"F sampled {len(sel)} (stride {args.stride})",
        flush=True,
    )

    # ---- F: fidelity, rotation removed, radially weighted ----
    rows = []
    for i in sel:
        t_b, f_b = _best_theta(cf[i], amp_base[i], w, z, mask)
        t_f, f_f = _best_theta(cf[i], amp_ft[i], w, z, mask)
        rows.append((ths[i], f_b, f_f, t_b, t_f))
    r = {round(x[0], 2): x for x in rows}
    fb = np.array([x[1] for x in rows])
    ff = np.array([x[2] for x in rows])
    print(
        "\n== F: wRMS_w(|h|, |R_theta* cf|), weight 1 inside "
        f"|z|<={R_CORE} tapering to {W_EDGE} at the rim"
    )
    print(
        f"{'psi':>7} | {'F_base':>7} {'F_cand':>7} {'delta':>8} | "
        f"{'th_base':>7} {'th_cand':>7}"
    )
    for ps in COLS:
        k = round(ths[int(np.argmin(np.abs(ths - ps)))], 2)
        if k not in r:
            print(f"{ps:7.1f} | classical NaN (dead zone)")
            continue
        x = r[k]
        print(
            f"{x[0]:7.2f} | {x[1]:7.4f} {x[2]:7.4f} {x[2] - x[1]:+8.4f} | "
            f"{x[3]:7.1f} {x[4]:7.1f}"
        )
    worse = [x[0] for x in rows if x[2] > x[1] + TOL]
    win = 1.0 - len(worse) / len(rows)
    print(
        f"mean   | {fb.mean():7.4f} {ff.mean():7.4f} "
        f"{ff.mean() - fb.mean():+8.4f}   ({100 * (1 - ff.mean() / fb.mean()):.1f}% down)"
    )
    print(
        f"per-psi| regressed (>{TOL}) at {len(worse)}/{len(rows)} "
        f"({win * 100:.1f}% kept)"
        + (f"  {[f'{x:.2f}' for x in worse[:12]]}" if worse else "")
    )

    # ---- C: continuity of the raw sequence, no rotation, no exceptions ----
    print(
        "\n== C: adjacent-d wRMS (0.25 deg grid, no rotation removed, "
        "no branch-switch exemption)"
    )
    dc_b = _adjacent_d(amp_base, w, 0, len(ths) - 1)
    dc_f = _adjacent_d(amp_ft, w, 0, len(ths) - 1)
    print(
        f"global  base: max {dc_b.max():.4f} mean {dc_b.mean():.4f} "
        f"jumps>{JUMP} {int((dc_b > JUMP).sum())}"
    )
    print(
        f"global  cand: max {dc_f.max():.4f} mean {dc_f.mean():.4f} "
        f"jumps>{JUMP} {int((dc_f > JUMP).sum())}"
    )
    mid = 0.5 * (ths[:-1] + ths[1:])
    print(f"{'flip':>7} | {'peak_base':>9} {'peak_cand':>9} {'ratio':>6}")
    pk_b, pk_f = [], []
    for f in FLIPS:
        near = np.where(np.abs(mid - f) <= FLIP_WIN)[0]
        pb, pf = float(dc_b[near].max()), float(dc_f[near].max())
        pk_b.append(pb)
        pk_f.append(pf)
        print(f"{f:7.2f} | {pb:9.4f} {pf:9.4f} {pf / pb:6.2f}")
    peak_b, peak_f = max(pk_b), max(pk_f)
    print(f"worst   | {peak_b:9.4f} {peak_f:9.4f} {peak_f / peak_b:6.2f}")
    # theta* plan, reported for information only (NOT a gate)
    th_b = np.array([x[3] for x in rows])
    th_f = np.array([x[4] for x in rows])
    print(
        f"theta*   | base p50 {np.median(th_b):.0f} cand p50 {np.median(th_f):.0f}"
        f"  (info only; degenerate where the field is near-symmetric)"
    )

    # ---- O: absolute frame -- the candidate must live in BASE's frame ----
    # Measured on the BASE FIELD, not on classical: classical flips branch
    # at its seams, so differencing theta* against it reports classical's
    # seam motion as a frame error and fails everything.
    phi, Rr = _frame_stats(hf_base, amp_ft, sel, w, z, mask)
    print(
        f"\n== O: base field rotated onto candidate; "
        f"phi = mean offset, |R| = consistency (1 = rigid re-framing)"
    )
    print(f"phi {phi:7.1f} deg   |R| {Rr:5.2f}   (healthy floor |R| ~ 0.43)")

    # Every "better than base" comparison is <=, not <: with < a model
    # cannot pass its own gate (base scores exactly base), which makes the
    # baseline itself FAIL and the whole gate meaningless.
    gates = {
        "F improves": bool(ff.mean() <= fb.mean()),
        "F no per-psi regression": bool(win >= MIN_WIN),
        "C no discontinuity": bool(dc_f.max() <= JUMP),
        "C flip peak <= base": bool(peak_f <= peak_b * PEAK_TOL),
        "O matches base frame": bool(
            Rr >= FRAME_R_MIN and abs(phi) <= ORIENT_TOL
        ),
    }
    print("\ngates:", gates)
    print(f"VERDICT: {'PASS' if all(gates.values()) else 'FAIL'}")

    np.savez_compressed(
        os.path.join(OUT_DIR, f"eval_standard_{tag}.npz"),
        rows=np.array(rows),
        ths=ths,
        d_adj_base=dc_b,
        d_adj_cand=dc_f,
        frame_phi=phi,
        frame_R=Rr,
        w=w,
    )
    print(f"\nsaved: {OUT_DIR}/eval_standard_{tag}.npz")


if __name__ == "__main__":
    main()
