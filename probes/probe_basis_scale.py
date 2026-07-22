"""Probe 3 — basis scale calibration (synthetic / real energy ratio).

The combinator assembles synthetic samples from scaled basis images. If the
synthetic stream carries R x the energy of real frames, the basis scale must be
divided by R. This probe builds the full configured training pipeline, compares
mean per-image energy (sum) of synthetic vs real batches, and prints corrected
torch_scale factors:

  new_scale = current_scale / ratio      (per slot)

Reference: cam3 ratio 3.58 at scale 1.0 -> 0.28; cam2 target keeps 0.26.
A correctly calibrated config prints ratio ~= 1.0 for both slots.

Feeds: data.basis_transforms -> torch_scale.scale_factor (input and target)

Run from repo root (needs the datasets; slow on first run due to caching):
  python -m probes.probe_basis_scale --config CLEAR26_sgm_cam3 --batches 8
"""
from __future__ import annotations

import argparse

import numpy as np

from probes.common import basis_scales, load_experiment


def batch_sums(dataset, n_batches):
    sums_in, sums_tgt, peaks_in, peaks_tgt = [], [], [], []
    for k, batch in enumerate(dataset):
        if k >= n_batches:
            break
        inputs, targets = batch[0], batch[1]
        x = np.asarray(inputs.detach().cpu(), dtype=np.float64)
        y = np.asarray(targets.detach().cpu(), dtype=np.float64)
        sums_in += list(x.sum(axis=(1, 2, 3)))
        sums_tgt += list(y.sum(axis=(1, 2, 3)))
        peaks_in += list(x.max(axis=(1, 2, 3)))
        peaks_tgt += list(y.max(axis=(1, 2, 3)))
    return map(np.asarray, (sums_in, sums_tgt, peaks_in, peaks_tgt))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="CLEAR26_sgm_cam3")
    ap.add_argument("--batches", type=int, default=8, help="batches per stream")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    scale_in, scale_tgt = basis_scales(cfg)

    from utils import build_datasets  # heavy import; keep local
    ds = build_datasets(cfg)

    syn_si, syn_st, syn_pi, syn_pt = batch_sums(ds["train_dataset"], args.batches)
    real_si, real_st, real_pi, real_pt = batch_sums(ds["test_dataset"], args.batches)

    print(f"config {args.config}   batches {args.batches}   "
          f"synthetic n={len(syn_si)}   real n={len(real_si)}\n")
    for role, cur, syn_s, real_s, syn_p, real_p in (
        ("input ", scale_in, syn_si, real_si, syn_pi, real_pi),
        ("target", scale_tgt, syn_st, real_st, syn_pt, real_pt),
    ):
        ratio = float(syn_s.mean() / real_s.mean())
        print(f"{role}: energy synth {syn_s.mean():9.1f}  real {real_s.mean():9.1f}  "
              f"ratio {ratio:5.2f}   peak synth {syn_p.mean():.3f}  real {real_p.mean():.3f}")
        print(f"        current torch_scale {cur:.3f}  ->  suggested {cur / ratio:.3f}"
              f"{'   (ok, keep)' if abs(ratio - 1) < 0.1 else ''}")


if __name__ == "__main__":
    main()
