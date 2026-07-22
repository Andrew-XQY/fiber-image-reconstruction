"""Probe 1 — per-camera count normalizer.

Measures p99.5 of per-image peaks on background-subtracted eval counts, adds a
safety margin, and rounds up. The result feeds, per camera:

  data.transforms  ->  torch_remap_range.current_max
  combinator       ->  torch_sensor_noise.count_scale   (input camera only)

Reference (2026-07 dataset): cam3 p99.5 ~1952 -> 2100, cam2 ~821 -> 900,
cam1 -> 1600. One fixed constant per camera across basis/train/val/test
preserves intensity linearity; do NOT use per-image min-max.

Run from repo root:
  python -m probes.probe_normalizer --config CLEAR26_sgm_cam3
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from probes.common import counts_loader, iter_pairs, load_experiment, slot_cameras


def suggest(p995: float, margin: float, step: int) -> int:
    return int(math.ceil(p995 * margin / step) * step)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="CLEAR26_sgm_cam3")
    ap.add_argument("--sql", default=None, help="SQL key (default: eval_sql_key)")
    ap.add_argument("--limit", type=int, default=None, help="max pairs")
    ap.add_argument("--margin", type=float, default=1.07)
    ap.add_argument("--step", type=int, default=100, help="round up to multiple")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    cam_in, cam_tgt = slot_cameras(cfg)
    load_in, load_tgt = counts_loader(cfg, cam_in), counts_loader(cfg, cam_tgt)

    peaks = {cam_in: [], cam_tgt: []}
    for p_in, p_tgt in iter_pairs(cfg, sql_key=args.sql, limit=args.limit):
        peaks[cam_in].append(load_in(p_in).max())
        peaks[cam_tgt].append(load_tgt(p_tgt).max())

    n = len(peaks[cam_in])
    print(f"config {args.config}   pairs {n}   margin {args.margin}   step {args.step}\n")
    for cam, role in ((cam_in, "input"), (cam_tgt, "target")):
        v = np.asarray(peaks[cam])
        p995 = float(np.percentile(v, 99.5))
        s = suggest(p995, args.margin, args.step)
        print(f"{cam} ({role}): peak p50 {np.percentile(v, 50):7.1f}   "
              f"p99.5 {p995:7.1f}   max {v.max():7.1f}   ->  current_max = {s}")
        if role == "input":
            print(f"    also set combinator torch_sensor_noise.count_scale = {s}")


if __name__ == "__main__":
    main()
