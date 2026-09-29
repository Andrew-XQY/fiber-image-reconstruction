"""Add-on beam parameter scores for the evaluation notebook: sample-relative
width errors and physical (mm) values. Kept free of xflow/matplotlib imports so
probes/refresh_beam_scores.py can patch existing runs with exactly the same code.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

GAUSSIAN_PARAMS = ("h_centroid", "v_centroid", "h_width", "v_width")


def compute_sample_relative_width_scores(df: pd.DataFrame, fit_method: str = "gaussian") -> dict:
    """Relative beam width error per sample: (predicted - true) / true.

    Reads only ``label_/reconstructed_{fit_method}_{h,v}_width`` and returns
    new ``{fit_method}_sample_relative_{h,v}_width_{mae,medae,bias}`` keys, so the result
    can be merged into an existing metrics dict without touching other values.
    """
    results = {}
    for dim in ("h", "v"):
        y = pd.to_numeric(df[f"label_{fit_method}_{dim}_width"], errors="coerce").to_numpy(dtype=float)
        yh = pd.to_numeric(df[f"reconstructed_{fit_method}_{dim}_width"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(yh) & (y > 0)
        rel = (yh[m] - y[m]) / y[m]
        key = f"{fit_method}_sample_relative_{dim}_width"
        results[f"{key}_mae"] = float(np.mean(np.abs(rel))) if rel.size else float("nan")
        results[f"{key}_medae"] = float(np.median(np.abs(rel))) if rel.size else float("nan")
        results[f"{key}_bias"] = float(np.mean(rel)) if rel.size else float("nan")
    return results


# ---------------------------------------------------------------- physical units
# Frame-normalised parameters (xflow normalize_beam_parameters): widths are
# divided by the frame size in pixels, centroids by frame size - 1. Multiplying
# by the physical frame size (with that same reference) gives millimetres.

def frame_mm_from_dataset(db_path, pixel_scale_json, camera: str = "cam2") -> tuple[float, float]:
    """Physical size (mm) of the evaluation frame, (horizontal, vertical).

    = the target camera's export crop_box, in original camera pixels (recorded
    in run_info of dataset.db), x the measured mm per original pixel
    (chromox_pixel_scale.json). Binning after the crop cancels out.
    """
    scale = json.loads(Path(pixel_scale_json).read_text(encoding="utf-8"))
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT value FROM run_info WHERE key LIKE '%export_config_json'").fetchall()
    finally:
        con.close()
    spans = set()
    for (value,) in rows:
        hooks = json.loads(value)["hooks"]["image"][camera]
        box = next(h["params"] for h in hooks if h["name"] == "crop_box")
        spans.add((box["x2"] - box["x1"], box["y2"] - box["y1"]))
    if len(spans) != 1:
        raise ValueError(f"{db_path}: expected one {camera} crop_box, found {spans}")
    (span_x, span_y), = spans
    return span_x * scale["x_mm_per_pixel"], span_y * scale["y_mm_per_pixel"]


def mm_factor(param: str, frame_mm: float, frame_px: int = 256) -> float:
    """Multiplier turning a frame-normalised value of `param` into millimetres."""
    reference = frame_px - 1 if param.endswith("centroid") else frame_px
    return frame_mm * reference / frame_px


def append_mm_columns(csv_path, frame_mm_h: float, frame_mm_v: float,
                      frame_px: int = 256, fit_method: str = "gaussian") -> list[str]:
    """Add ``label_/reconstructed_{fit_method}_{param}_mm`` columns to a
    beam_parameters.csv. Existing columns are kept byte for byte; already
    present mm columns are recomputed in place. Returns the mm column names.
    """
    csv_path = Path(csv_path)
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8").splitlines()))
    header, body = rows[0], rows[1:]
    columns = [(f"{prefix}_{fit_method}_{param}", param)
               for prefix in ("label", "reconstructed") for param in GAUSSIAN_PARAMS]
    new_names = [f"{name}_mm" for name, _ in columns]
    keep = [i for i, name in enumerate(header) if name not in new_names]
    index = {name: i for i, name in enumerate(header)}
    out = [[header[i] for i in keep] + new_names]
    for row in body:
        values = []
        for name, param in columns:
            factor = mm_factor(param, frame_mm_h if param.startswith("h_") else frame_mm_v, frame_px)
            text = row[index[name]].strip()
            values.append(repr(float(text) * factor) if text else "")
        out.append([row[i] for i in keep] + values)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(out)
    return new_names


def scale_metrics_to_mm(metrics: dict, frame_mm_h: float, frame_mm_v: float,
                        frame_px: int = 256, fit_method: str = "gaussian") -> dict:
    """Existing frame-normalised MAE/RMSE re-expressed in millimetres.

    MAE and RMSE scale linearly with the unit, so no per-sample recomputation
    is needed. Adds ``{fit_method}_{param}_{mae,rmse}_mm`` for the four Gaussian
    parameters, their mean ``{fit_method}_{mae,rmse}_mm`` (same definition as
    the existing ``{fit_method}_{mae,rmse}``) and the frame size used.
    """
    out = {}
    for metric in ("mae", "rmse"):
        keys = []
        for param in GAUSSIAN_PARAMS:
            frame_mm = frame_mm_h if param.startswith("h_") else frame_mm_v
            key = f"{fit_method}_{param}_{metric}_mm"
            out[key] = float(metrics[f"{fit_method}_{param}_{metric}"]) * mm_factor(param, frame_mm, frame_px)
            keys.append(key)
        out[f"{fit_method}_{metric}_mm"] = sum(out[k] for k in keys) / len(keys)
    out["eval_frame_mm_h"] = float(frame_mm_h)
    out["eval_frame_mm_v"] = float(frame_mm_v)
    return out
