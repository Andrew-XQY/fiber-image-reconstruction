"""Probe 4 — synthetic beam-distribution matching.

Measures peak, relative widths, centroid radius, footprint, and empty rate on
real eval targets in normalized training space. With ``--synth`` it also
measures rendered training targets from one of three coefficient sources:
``mixed`` (the configured production stream), ``sgm``, or ``real``.

Use ``mixed`` for the final end-to-end calibration because the real and SGM
branches can intentionally bracket the eval distribution. Use ``sgm`` and
``real`` to diagnose either branch in isolation.

Run from repo root:
  python -m probes.probe_sgm_prior --config CLEAR26_690_cam3
  python -m probes.probe_sgm_prior --config CLEAR26_690_cam3 --synth
  python -m probes.probe_sgm_prior --config CLEAR26_690_cam3 --synth --source sgm
"""
from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np

from probes.common import (beam_marginals, counts_loader, iter_pairs,
                           load_experiment, remap_maxes, slot_cameras,
                           summarize)

KEYS = ("peak", "sigma_x_rel", "sigma_y_rel", "centroid_r_rel", "footprint", "empty")


def collect(images, thr):
    stats = {k: [] for k in KEYS}
    n = 0
    for img in images:
        m = beam_marginals(np.asarray(img, dtype=np.float32).squeeze(), thr)
        for k in KEYS:
            stats[k].append(m[k])
        n += 1
    return stats, n


def real_targets(cfg, sql_key, limit):
    _, cam_tgt = slot_cameras(cfg)
    _, tgt_max = remap_maxes(cfg)
    load = counts_loader(cfg, cam_tgt)
    for _, p_tgt in iter_pairs(cfg, sql_key=sql_key, limit=limit):
        yield load(p_tgt) / tgt_max


def synth_targets(cfg, limit, source="mixed"):
    from utils import build_datasets  # heavy import; keep local
    cfg = deepcopy(cfg)
    if source == "sgm":
        cfg["weighted_stream"] = {"real_weight": 0.0, "sgm_weight": 1.0}
    elif source == "real":
        cfg["weighted_stream"] = {"real_weight": 1.0, "sgm_weight": 0.0}
    ds = build_datasets(cfg)
    n = 0
    for batch in ds["train_dataset"]:
        for img in np.asarray(batch[1].detach().cpu()):
            if n >= limit:
                return
            yield img
            n += 1


def report(title, stats, n, thr):
    print(f"\n== {title} (n={n}, footprint_thr={thr}) ==")
    for k in KEYS[:-1]:
        print("  " + summarize(stats[k], k))
    print(f"  {'empty':<16} {100 * np.nanmean(stats['empty']):.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="CLEAR26_sgm_cam3")
    ap.add_argument("--sql", default=None, help="SQL key (default: eval_sql_key)")
    ap.add_argument("--limit", type=int, default=None, help="max real targets")
    ap.add_argument("--synth", action="store_true",
                    help="also Monte-Carlo the selected synthetic stream")
    ap.add_argument("--synth-samples", type=int, default=512)
    ap.add_argument("--source", choices=["mixed", "sgm", "real"], default="mixed",
                    help="pattern source to probe (default: configured mixture)")
    ap.add_argument("--thr", type=float, default=0.05, help="footprint threshold")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    sim = cfg["simulation"]
    print(f"config {args.config}   current prior: intensity {sim.get('intensity_range')}   "
          f"center_radius {sim.get('center_radius_range')}   aspect {sim.get('aspect_range')}")

    stats, n = collect(real_targets(cfg, args.sql, args.limit), args.thr)
    report("REAL eval targets", stats, n, args.thr)

    if args.synth:
        stats_s, n_s = collect(
            synth_targets(cfg, args.synth_samples, args.source), args.thr
        )
        report(f"SYNTHETIC rendered targets ({args.source})", stats_s, n_s, args.thr)
        print("\nCompare synthetic and real distributions; a small empty fraction "
              "can be part of the configured prior.")


if __name__ == "__main__":
    main()
