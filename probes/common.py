"""Shared helpers for the calibration probes. See probes/README.md.

All probes work in one of two image spaces:
- "counts":     load_image16 -> tensor -> background-subtracted counts (0..4095)
- "normalized": counts / torch_remap_range.current_max  (training space)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_utils import detect_machine, load_config  # noqa: E402
from utils import resolve_dataset_dir  # noqa: E402
from xflow import ConfigManager, SqlProvider  # noqa: E402
from xflow.data import build_transforms_from_config  # noqa: E402


# ---------------------------------------------------------------- config

def load_experiment(name: str) -> dict:
    """Resolved experiment config, e.g. load_experiment("CLEAR26_sgm_cam3")."""
    return ConfigManager(
        load_config(f"{name}.yaml", machine=detect_machine(), resolve=True)
    ).get()


def _walk_transforms(node, name):
    """Yield params dicts of every transform called `name`, in config order."""
    if isinstance(node, list):
        for item in node:
            yield from _walk_transforms(item, name)
    elif isinstance(node, dict):
        if node.get("name") == name:
            yield node.get("params", {}) or {}
        for value in node.values():
            yield from _walk_transforms(value, name)


def slot_cameras(cfg) -> tuple[str, str]:
    """(input_camera, target_camera) from the torch_subtract_background slots."""
    cams = [p.get("camera") for p in
            _walk_transforms(cfg["data"]["transforms"]["torch"], "torch_subtract_background")]
    if len(cams) < 2:
        raise RuntimeError("Could not find two torch_subtract_background slots in config.")
    return cams[0], cams[1]


def remap_maxes(cfg) -> tuple[float, float]:
    """(input, target) torch_remap_range.current_max from the training transforms."""
    maxes = [float(p["current_max"]) for p in
             _walk_transforms(cfg["data"]["transforms"]["torch"], "torch_remap_range")]
    if len(maxes) < 2:
        raise RuntimeError("Could not find two torch_remap_range slots in config.")
    return maxes[0], maxes[1]


def basis_scales(cfg) -> tuple[float, float]:
    """(input, target) torch_scale.scale_factor from basis_transforms."""
    scales = [float(p["scale_factor"]) for p in
              _walk_transforms(cfg["data"]["basis_transforms"]["torch"], "torch_scale")]
    if len(scales) < 2:
        raise RuntimeError("Could not find two torch_scale slots in basis_transforms.")
    return scales[0], scales[1]


# ---------------------------------------------------------------- data access

def dataset_dir_and_db(cfg, source: str | None = None) -> tuple[Path, Path]:
    """Default source = last entry of data.dataset_sources (the eval dataset)."""
    source = source or cfg["data"]["dataset_sources"][-1]
    d = resolve_dataset_dir(cfg, source)
    return d, d / cfg["dataset_structure"]["db"].lstrip("/\\")


def iter_pairs(cfg, sql_key: str | None = None, source: str | None = None,
               limit: int | None = None):
    """Yield (input_path, target_path) absolute Paths from a config SQL key."""
    sql_key = sql_key or cfg["data"]["eval_sql_key"]
    dataset_dir, db_path = dataset_dir_and_db(cfg, source)
    provider = SqlProvider(
        sources={"connection": db_path, "sql": cfg["sql"][sql_key]},
        output_config={"list": cfg["data"].get("provider_output_column", "image_pair")},
    )
    for i, item in enumerate(provider()):
        if limit is not None and i >= limit:
            return
        left, right = str(item).split("|")
        yield dataset_dir / left, dataset_dir / right


def counts_loader(cfg, camera: str):
    """Callable: png path -> float32 (H, W) background-subtracted counts.

    Uses the exact registry transforms the trainer uses, so the probe measures
    the same image space as training.
    """
    background_params = [
        p for p in _walk_transforms(
            cfg["data"]["transforms"]["torch"], "torch_subtract_background"
        )
        if p.get("camera") == camera and p.get("background_source")
    ]
    if not background_params:
        raise RuntimeError(
            f"Could not find torch_subtract_background for camera {camera!r}."
        )

    background_transform = dict(background_params[0])
    background_transform.setdefault("clip_min", 0.0)
    chain = build_transforms_from_config([
        {"name": "load_image16"},
        {"name": "torch_to_tensor"},
        {"name": "torch_subtract_background",
         "params": background_transform},
    ])
    def load(path):
        x = str(path)
        for fn in chain:
            x = fn(x)
        return np.asarray(x, dtype=np.float32).squeeze()
    return load


# ---------------------------------------------------------------- statistics

def beam_marginals(img: np.ndarray, footprint_thr: float = 0.05) -> dict:
    """The five SGM-prior marginals of one normalized image (H, W).

    peak            max pixel value
    sigma_x/y_rel   intensity-weighted RMS width / canvas width
    centroid_r_rel  centroid distance from canvas center / canvas width
    footprint       fraction of pixels above footprint_thr
    empty           peak < footprint_thr
    """
    h, w = img.shape
    peak = float(img.max())
    out = {"peak": peak, "footprint": float((img > footprint_thr).mean()),
           "empty": float(peak < footprint_thr)}
    total = float(img.sum())
    if total <= 0:
        out.update(sigma_x_rel=np.nan, sigma_y_rel=np.nan, centroid_r_rel=np.nan)
        return out
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((img * xs).sum() / total)
    cy = float((img * ys).sum() / total)
    sx = float(np.sqrt((img * (xs - cx) ** 2).sum() / total))
    sy = float(np.sqrt((img * (ys - cy) ** 2).sum() / total))
    out.update(
        sigma_x_rel=sx / w,
        sigma_y_rel=sy / h,
        centroid_r_rel=float(np.hypot(cx / (w - 1) - 0.5, cy / (h - 1) - 0.5)),
    )
    return out


def summarize(values, label: str) -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return f"{label:<16} (no finite values)"
    return (f"{label:<16} mean {v.mean():7.3f}   p5 {np.percentile(v, 5):7.3f}"
            f"   p50 {np.percentile(v, 50):7.3f}   p95 {np.percentile(v, 95):7.3f}")
