"""Probe 2 — sensor noise photon-transfer fit ("spatial noise-gap" method).

Fits the affine Poisson-Gaussian model used by torch_sensor_noise:

  var(counts) = a * signal + b        (a: counts, b: counts^2)

from single frames, in the same image space training sees (background-
subtracted counts). Method: horizontal+vertical neighbor differences share the
local signal, so var(diff)/2 estimates pixel noise; a robust (MAD) per-signal-bin
estimate suppresses speckle structure, then a weighted linear fit gives a, b.

Estimator caveat: spatial structure biases the estimate upward, so treat the
result as an upper bound and keep param_jitter ~2.0 in the config to span the
uncertainty. Reference fits (this method): cam1 a=0.106 b=2.81 (count_scale
1600), cam3 a=0.118 b=4.82 (count_scale 2100).

Feeds: combinator -> torch_sensor_noise.{a, b, count_scale}
(count_scale itself comes from probe_normalizer, not from this fit.)

Run from repo root:
  python -m probes.probe_sensor_noise --config CLEAR26_sgm_cam3
"""
from __future__ import annotations

import argparse

import numpy as np

from probes.common import (counts_loader, iter_pairs, load_experiment,
                           remap_maxes, slot_cameras)

MAD_TO_SIGMA = 1.4826


def pair_diffs(img: np.ndarray):
    """Neighbor differences and their local signal (pair mean)."""
    dh, sh = img[:, 1:] - img[:, :-1], 0.5 * (img[:, 1:] + img[:, :-1])
    dv, sv = img[1:, :] - img[:-1, :], 0.5 * (img[1:, :] + img[:-1, :])
    return (np.concatenate([dh.ravel(), dv.ravel()]),
            np.concatenate([sh.ravel(), sv.ravel()]))


def fit_noise(diffs, signals, n_bins, min_per_bin, s_max):
    keep = (signals >= 0) & (signals <= s_max)
    diffs, signals = diffs[keep], signals[keep]
    edges = np.quantile(signals, np.linspace(0, 1, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (signals >= lo) & (signals < hi)
        if m.sum() < min_per_bin:
            continue
        d = diffs[m]
        sigma_d = MAD_TO_SIGMA * np.median(np.abs(d - np.median(d)))
        rows.append((float(np.median(signals[m])), float(sigma_d**2 / 2.0), int(m.sum())))
    s = np.array([r[0] for r in rows])
    var = np.array([r[1] for r in rows])
    wts = np.array([r[2] for r in rows], dtype=np.float64)
    a, b = np.polyfit(s, var, 1, w=np.sqrt(wts))
    resid = var - (a * s + b)
    r2 = 1.0 - float((wts * resid**2).sum() / (wts * (var - var.mean()) ** 2).sum())
    return float(a), float(b), r2, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="CLEAR26_sgm_cam3")
    ap.add_argument("--slot", choices=["input", "target"], default="input")
    ap.add_argument("--sql", default=None, help="SQL key (default: eval_sql_key)")
    ap.add_argument("--limit", type=int, default=200, help="max frames")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--min-per-bin", type=int, default=2000)
    ap.add_argument("--max-signal-frac", type=float, default=0.6,
                    help="fit only below this fraction of the camera normalizer")
    ap.add_argument("--table", action="store_true", help="print per-bin table")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    cams, maxes = slot_cameras(cfg), remap_maxes(cfg)
    i = 0 if args.slot == "input" else 1
    cam, count_scale = cams[i], maxes[i]
    load = counts_loader(cfg, cam)

    all_d, all_s = [], []
    for paths in iter_pairs(cfg, sql_key=args.sql, limit=args.limit):
        d, s = pair_diffs(load(paths[i]))
        all_d.append(d)
        all_s.append(s)
    diffs, signals = np.concatenate(all_d), np.concatenate(all_s)

    a, b, r2, rows = fit_noise(diffs, signals, args.bins, args.min_per_bin,
                               args.max_signal_frac * count_scale)

    print(f"config {args.config}   slot {args.slot} ({cam})   frames {len(all_d)}   "
          f"fit range 0..{args.max_signal_frac * count_scale:.0f} counts")
    if args.table:
        print(f"\n{'signal':>10} {'var':>10} {'pixels':>10}")
        for s_med, var, n in rows:
            print(f"{s_med:>10.1f} {var:>10.2f} {n:>10d}")
    print(f"\nvar = a*signal + b fit (R^2 {r2:.3f}):")
    print(f"  torch_sensor_noise:  a: {a:.3f}   b: {b:.2f}   count_scale: {count_scale:.0f}")


if __name__ == "__main__":
    main()
