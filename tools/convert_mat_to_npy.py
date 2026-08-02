"""Convert img/**/*.mat (MAT v5, single key `hbs`, float64 (H,W,2)) to float32 .npy (CHW).

- Output: <same path>.npy, shape (2,H,W) float32, contiguous — the training format.
- .mat files are NEVER deleted: they remain the authoritative backup.
- Resumable: existing .npy files are skipped; writes are atomic (tmp + os.replace).
- Manifest: <root>/manifest.json {mat_rel_path: sha256_of_npy} written after a full pass.

Usage: python tools/convert_mat_to_npy.py [--root img] [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io as sio


def convert_one(mat_path: str) -> tuple[str, str]:
    """Convert one .mat to .npy. Returns (mat_path, sha256) or (mat_path, "") if skipped."""
    mat_path = Path(mat_path)
    npy_path = mat_path.with_suffix(".npy")
    if npy_path.exists():
        return str(mat_path), ""  # already converted (resume)
    try:
        hbs = sio.loadmat(mat_path)["hbs"]  # (H, W, 2) float64
        arr = np.ascontiguousarray(hbs.astype(np.float32).transpose(2, 0, 1))
        tmp = npy_path.with_suffix(".npy.tmp")
        # np.save on a path APPENDS ".npy" — pass a file object to avoid .tmp.npy
        with tmp.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp, npy_path)  # atomic publish
        digest = hashlib.sha256(npy_path.read_bytes()).hexdigest()
        return str(mat_path), digest
    except Exception as e:  # one corrupt file must not kill the batch
        print(f"ERROR {mat_path}: {e}", flush=True)
        return str(mat_path), "ERROR"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="img", help="root dir to scan for *.mat")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    mats = sorted(root.rglob("*.mat"))
    if not mats:
        print(f"No .mat files under {root}")
        return 1

    manifest: dict[str, str] = {}
    converted = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, str(m)) for m in mats]
        for i, fut in enumerate(as_completed(futures), 1):
            path, digest = fut.result()
            if digest == "ERROR":
                errors += 1
            elif digest:
                manifest[path] = digest
                converted += 1
            if i % 1000 == 0 or i == len(mats):
                print(f"progress {i}/{len(mats)} (converted={converted}, errors={errors})", flush=True)

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"done: {converted} converted, {len(mats) - converted} skipped, {errors} errors; manifest -> {manifest_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
