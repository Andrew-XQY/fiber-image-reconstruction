"""Backfill the add-on beam scores (notebooks/beam_scores.py) into existing
runs: sample-relative Gaussian width errors, and with --pixel-scale/--eval-db
also the physical (mm) values and scores.

Per <run>/inference/: new keys are added to _fit_metrics.json (or refreshed on
a re-run) and, for mm, new columns are appended to beam_parameters.csv. Every
existing key, value and column stays exactly as it was. A run is skipped when
beam_parameters.csv is missing or its valid-fit row count differs from the
saved score_selected_samples (the notebook scored it with an extra filter).

    python probes/refresh_beam_scores.py ~/Desktop/HPC/temp \\
        --pixel-scale ~/Desktop/AI_Workspace/dataset/chromox_pixel_scale.json \\
        --eval-db ~/Desktop/AI_Workspace/dataset/processed/Friday/405_realbeam_eval_256/dataset.db \\
        [--dry-run]

--eval-db is any evaluation dataset whose target-camera crop matches the runs
being patched (all CLEAR26 real-beam datasets use the same 512 px cam2 crop).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
from beam_scores import (  # noqa: E402
    append_mm_columns,
    compute_sample_relative_width_scores,
    frame_mm_from_dataset,
    scale_metrics_to_mm,
)


def selected_rows(csv_path: Path) -> pd.DataFrame:
    """Same row selection as the notebook score cell: finite, non-zero fits."""
    raw = pd.read_csv(csv_path, skipinitialspace=True)
    raw.columns = raw.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    fit_columns = [c for c in raw if "gaussian" in c or "moments" in c]
    fit_values = raw[fit_columns].apply(pd.to_numeric, errors="coerce")
    fit_valid = np.isfinite(fit_values).all(axis=1) & ~fit_values.eq(0).any(axis=1)
    return raw.loc[fit_valid].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_roots", nargs="+", type=Path, help="folders holding <run>/inference/")
    parser.add_argument("--pixel-scale", type=Path, help="chromox_pixel_scale.json (enables mm)")
    parser.add_argument("--eval-db", type=Path, help="dataset.db of a matching evaluation dataset")
    parser.add_argument("--frame-px", type=int, default=256, help="evaluation frame size in pixels")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()
    if bool(args.pixel_scale) != bool(args.eval_db):
        parser.error("--pixel-scale and --eval-db go together")
    frame_mm = None
    if args.pixel_scale:
        frame_mm = frame_mm_from_dataset(args.eval_db, args.pixel_scale)
        print(f"frame size: {frame_mm[0]:.4f} x {frame_mm[1]:.4f} mm (h x v)")

    metric_files = sorted(
        path for root in args.run_roots for path in root.glob("*/inference/_fit_metrics.json")
    )
    for metrics_path in metric_files:
        run = metrics_path.parent.parent.name
        csv_path = metrics_path.with_name("beam_parameters.csv")
        if not csv_path.is_file():
            print(f"SKIP  {run}: no beam_parameters.csv")
            continue
        original = json.loads(metrics_path.read_text(encoding="utf-8"))
        df = selected_rows(csv_path)
        expected = original.get("score_selected_samples")
        if expected is not None and expected != len(df):
            print(f"SKIP  {run}: {len(df)} valid rows != score_selected_samples {expected}")
            continue
        scores = compute_sample_relative_width_scores(df)
        if frame_mm is not None:
            scores.update(scale_metrics_to_mm(original, *frame_mm, frame_px=args.frame_px))
        action = "refresh" if all(k in original for k in scores) else "add"
        summary = "  ".join(
            f"{dim}: sample-relative mae {scores[f'gaussian_sample_relative_{dim}_width_mae']:.3f} bias {scores[f'gaussian_sample_relative_{dim}_width_bias']:+.3f}"
            for dim in ("h", "v")
        )
        if frame_mm is not None:
            summary += f"  | mm mae: width {scores['gaussian_h_width_mae_mm']:.3f}/{scores['gaussian_v_width_mae_mm']:.3f}" \
                       f" centroid {scores['gaussian_h_centroid_mae_mm']:.3f}/{scores['gaussian_v_centroid_mae_mm']:.3f}"
        if args.dry_run:
            print(f"DRY   {run} (n={len(df)}, would {action}): {summary}")
            continue
        if frame_mm is not None:
            append_mm_columns(csv_path, *frame_mm, frame_px=args.frame_px)
        merged = {**original, **scores}  # existing keys keep their values and order
        metrics_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"OK    {run} (n={len(df)}, {action}): {summary}")
    print(f"{len(metric_files)} metric files scanned")


if __name__ == "__main__":
    main()
