import os
from xflow.extensions.style.aps import *
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from pathlib import Path
import pandas as pd
import shutil

from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, List, Optional
from utils import SAMPLE_FLATTENED, REGRESSION, GAN 

def collect_and_copy_by_keyword(src_dir: str | os.PathLike,
                                dst_dir: str | os.PathLike,
                                keyword: str,
                                case_sensitive: bool = False) -> List[Tuple[Path, Path]]:
    """
    Recursively scan `src_dir` for files whose *basename* contains `keyword`,
    and copy each match into `dst_dir` named as:
        <parent-folder-name>_<original-filename>

    - Only the final path component (filename) is matched.
    - Matching is case-insensitive by default (set `case_sensitive=True` to disable).
    - If a destination filename already exists, it will be overwritten.
    - Returns a list of (source_path, destination_path) for all copied files.
    """
    src = Path(src_dir).expanduser().resolve()
    dst = Path(dst_dir).expanduser().resolve()

    if not src.exists() or not src.is_dir():
        raise ValueError(f"Source directory does not exist or is not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    if not keyword:
        raise ValueError("`keyword` must be a non-empty string.")

    needle = keyword if case_sensitive else keyword.lower()

    copied: List[Tuple[Path, Path]] = []
    for root, _, files in os.walk(src, followlinks=False):
        root_path = Path(root)
        parent_name = root_path.name if root_path != src else src.name

        for fname in files:
            hay = fname if case_sensitive else fname.lower()
            if needle in hay:  # match on basename only
                src_path = root_path / fname
                dst_path = dst / f"{parent_name}_{fname}"

                if src_path.is_file():
                    shutil.copy2(src_path, dst_path)  # overwrite if exists
                    copied.append((src_path, dst_path))

    return copied


def log_scale_lists(lists):
    """Log-scale each list in lists based on common min-max."""
    # Flatten all values to find global min and max
    all_values = np.concatenate([np.array(lst) for lst in lists])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    if min_val <= 0:
        # Shift all values if min <= 0 to avoid log(0)
        shift = abs(min_val) + 1e-8
        min_val += shift
        max_val += shift
        lists = [[x + shift for x in lst] for lst in lists]
    # Log scale: log(x) where x is normalized to [min_val, max_val]
    log_lists = [np.log(np.array(lst)) for lst in lists]
    return log_lists

def plot_iterables(data, palette=APS_COLORS):
    """
    Plot each iterable in `data` as a line. 
    X is the default 1..N index where N is the longest series length.
    Y range is set from the global min/max across all values.
    
    Args:
        data: iterable of iterables (e.g., list of lists/tuples).
        palette: list of matplotlib-compatible color codes (cycled if shorter).
    """
    import math
    from itertools import cycle
    import matplotlib.pyplot as plt

    # Normalize input to list of lists and sanity checks
    series = [list(s) for s in data if s is not None]
    if not series or all(len(s) == 0 for s in series):
        raise ValueError("Provide at least one non-empty iterable.")

    # Longest x and global y-range
    max_len = max(len(s) for s in series)
    y_min, y_max = None, None
    for s in series:
        for v in s:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    y_min = fv if y_min is None or fv < y_min else y_min
                    y_max = fv if y_max is None or fv > y_max else y_max
            except (TypeError, ValueError):
                continue
    if y_min is None:  # all values were non-numeric/NaN
        y_min, y_max = 0.0, 1.0

    x_full = list(range(1, max_len + 1))
    color_cycle = cycle(palette if palette else [None])

    fig, ax = plt.subplots()
    for s, c in zip(series, color_cycle):
        y = []
        for v in s:
            try:
                fv = float(v)
                y.append(fv if math.isfinite(fv) else float("nan"))
            except (TypeError, ValueError):
                y.append(float("nan"))
        ax.plot(x_full[:len(y)], y, color=c, linewidth=1.5)

    ax.set_xlim(1, max_len)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Training epoches")
    ax.set_ylabel("Loss (MSE)")
    # ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def list_subfolders_abs(dir_path):
    """
    Given a directory path, return a list of absolute paths of all folders directly under that directory.
    """
    abs_dir_path = os.path.abspath(dir_path)
    subfolders = [os.path.join(abs_dir_path, name)
                 for name in os.listdir(abs_dir_path)
                 if os.path.isdir(os.path.join(abs_dir_path, name))]
    return sorted(subfolders)
    
    


# ---- helpers ---------------------------------------------------------------
PARAM_KEYS = ["h_centroid", "v_centroid", "h_width", "v_width"]


def evaluate_to_df(
    model: torch.nn.Module,
    test_loader,                 # yields (inputs, targets), both (B,C,H,W)
    device: torch.device | str,
    extract_beam_parameters,     # returns dict or None
    mode: str,                   # "img2img" or "regression" or "flattened"
) -> pd.DataFrame:
    model.eval()
    rows = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
            inputs  = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            B = inputs.shape[0]

            for i in range(B):
                # predicted params
                if mode == "img2img":
                    pred_res = extract_beam_parameters(outputs[i])
                    pred_vals = {k: pred_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(pred_res, dict) else {k: -1.0 for k in PARAM_KEYS}
                elif mode == "regression":
                    vec = outputs[i].reshape(-1).detach().cpu().tolist()
                    pred_vals = {
                        "h_centroid": vec[0] if len(vec) > 0 else -1.0,
                        "v_centroid": vec[1] if len(vec) > 1 else -1.0,
                        "h_width":    vec[2] if len(vec) > 2 else -1.0,
                        "v_width":    vec[3] if len(vec) > 3 else -1.0,
                    }
                elif mode == "flattened":
                    # Assume output and target are 1D flattened square images
                    pred_vec = outputs[i].reshape(-1).detach().cpu().numpy()
                    side = int(np.sqrt(pred_vec.shape[0]))
                    pred_img = pred_vec.reshape(side, side)
                    pred_res = extract_beam_parameters(torch.from_numpy(pred_img))
                    pred_vals = {k: pred_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(pred_res, dict) else {k: -1.0 for k in PARAM_KEYS}
                else:
                    raise ValueError("mode must be 'img2img', 'regression', or 'flattened'")

                # ground truth params (from target image)
                if mode == "flattened":
                    gt_vec = targets[i].reshape(-1).detach().cpu().numpy()
                    side = int(np.sqrt(gt_vec.shape[0]))
                    gt_img = gt_vec.reshape(side, side)
                    gt_res = extract_beam_parameters(torch.from_numpy(gt_img))
                    gt_vals = {k: gt_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(gt_res, dict) else {k: -1.0 for k in PARAM_KEYS}
                else:
                    gt_res = extract_beam_parameters(targets[i])
                    gt_vals = {k: gt_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(gt_res, dict) else {k: -1.0 for k in PARAM_KEYS}

                # row
                row = {
                    **gt_vals,
                    **{f"{k}_pred": v for k, v in pred_vals.items()},
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df
 

def add_beamparam_metrics(df: pd.DataFrame, metrics=('RMSE','MSE','MAE')) -> pd.DataFrame:
    allowed = {'RMSE','MSE','MAE'}
    metrics = [m.upper() for m in metrics]
    if any(m not in allowed for m in metrics):
        raise ValueError(f"metrics must be subset of {allowed}")

    out = df.copy()
    params = [c for c in out.columns if not c.endswith('_pred') and f"{c}_pred" in out.columns]

    for p in params:
        gt = out[p]
        pr = out[f"{p}_pred"]
        valid = (gt != -1) & (pr != -1)
        err = pr - gt

        if 'MSE' in metrics:
            col = f"{p}_MSE"
            out[col] = -1.0
            out.loc[valid, col] = (err.pow(2))[valid]

        if 'RMSE' in metrics:
            col = f"{p}_RMSE"
            out[col] = -1.0
            out.loc[valid, col] = (err.abs())[valid]  # sqrt(MSE) per-sample == |err|

        if 'MAE' in metrics:
            col = f"{p}_MAE"
            out[col] = -1.0
            out.loc[valid, col] = (err.abs())[valid]

    for m in metrics:
        cols = [f"{p}_{m}" for p in params]
        mean_col = f"{m}_mean"
        out[mean_col] = out[cols].mean(axis=1)
        out.loc[(out[cols] == -1).any(axis=1), mean_col] = -1.0

    return out


def add_extraction_state_and_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,  # e.g. {"MAE": 0.05, "RMSE": 0.08}
) -> pd.DataFrame:
    out = df.copy()

    # Detect parameter bases from "<p>" and matching "<p>_pred"
    params = [c for c in out.columns if not c.endswith("_pred") and f"{c}_pred" in out.columns]
    if not params:
        return out  # nothing to do

    gt_cols   = params
    pred_cols = [f"{p}_pred" for p in params]

    # Extraction states: 1 if all four are valid (!=-1), else 0
    out["ground_truth_extraction_state"] = (out[gt_cols]   != -1).all(axis=1).astype(int)
    out["prediction_extraction_state"]   = (out[pred_cols] != -1).all(axis=1).astype(int)

    # Threshold fails on per-row metric means (if present), e.g., "RMSE_mean"
    if thresholds:
        for m, thr in thresholds.items():
            m_up = m.upper()
            mean_col = f"{m_up}_mean"
            if mean_col in out.columns:
                out[f"{m_up}_threshold_fail"] = (out[mean_col] > float(thr)).astype(int)

    return out


def summarize_error_columns_to_json(df: pd.DataFrame, json_path: str | Path) -> pd.DataFrame:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # rows to keep: both extraction states == 1 (if present)
    mask = pd.Series(True, index=df.index)
    if "ground_truth_extraction_state" in df.columns:
        mask &= df["ground_truth_extraction_state"] == 1
    if "prediction_extraction_state" in df.columns:
        mask &= df["prediction_extraction_state"] == 1
    dff = df[mask]

    # exclude original params, their _pred, extraction/threshold flags
    exclude = set(PARAM_KEYS + [f"{k}_pred" for k in PARAM_KEYS] + [
        "ground_truth_extraction_state",
        "prediction_extraction_state",
    ])
    candidates = [
        c for c in dff.columns
        if c not in exclude
        and not c.endswith("_threshold_fail")
        and "extraction_state" not in c
        and pd.api.types.is_numeric_dtype(dff[c])
    ]

    stats = {}
    for c in candidates:
        col = dff[c]
        n_valid = col.notna().sum()
        stats[f"{c}_avg"] = float(col.mean()) if n_valid > 0 else None
        stats[f"{c}_std"]  = float(col.std(ddof=1)) if n_valid > 1 else None

    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    return df


def save_single_sample_triplet(
    model: torch.nn.Module,
    x: torch.Tensor,              # (C,H,W) or (1,C,H,W) or (N,) for flattened
    y: torch.Tensor,              # (C,H,W) or (1,C,H,W) or (N,) for flattened
    device: str | torch.device,
    out_dir: str | Path = "results/single_sample",
    prefix: str = "index",        # files: index_0_input.png, index_1_target.png, index_2_pred.png
    channel: int = 0,             # which channel to visualize if multi-channel
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mode: str = "img2img",       # add mode argument
) -> None:
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _to_2d(t: torch.Tensor):
        t = t.detach().cpu()
        if mode == "flattened":
            arr = t.reshape(-1).numpy()
            side = int(np.sqrt(arr.shape[0]))
            return arr.reshape(side, side)
        if t.ndim == 3:   # (C,H,W)
            return t[channel].numpy()
        if t.ndim == 2:   # (H,W)
            return t.numpy()
        raise TypeError(f"Expected (C,H,W), (H,W), or flattened, got {tuple(t.shape)}")

    def _save(arr2d, path):
        fig, ax = plt.subplots(figsize=(6, 6))   # 1:1
        ax.imshow(arr2d, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    with torch.no_grad():
        if mode == "flattened":
            xin = x.to(device).float().reshape(-1)
            xbat = xin.unsqueeze(0)  # (1,N)
            pred = model(xbat)[0]    # (N,)
        else:
            xin = x.to(device).float()
            xbat = xin if xin.ndim == 4 else xin.unsqueeze(0)   # (1,C,H,W)
            pred = model(xbat)[0]                               # (C,H,W)

    x_img    = _to_2d(x[0] if (x.ndim == 4 and mode != "flattened") else x)
    y_img    = _to_2d(y[0] if (y.ndim == 4 and mode != "flattened") else y)
    pred_img = _to_2d(pred)

    _save(x_img,    out_dir / f"{prefix}_0_input.png")
    _save(y_img,    out_dir / f"{prefix}_1_target.png")
    _save(pred_img, out_dir / f"{prefix}_2_pred.png")


def plot_sanity(df: pd.DataFrame, bins: int = 50, save_dir: str | Path | None = None) -> None:
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    def _scatter_pair(ax, x, y, label, color):
        m = (x != -1) & (y != -1) & np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=10, alpha=0.7, label=label, c=color)

    # 1) Centroid parity (square)
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    x_hc, y_hc = df["h_centroid"].to_numpy(), df["h_centroid_pred"].to_numpy()
    x_vc, y_vc = df["v_centroid"].to_numpy(), df["v_centroid_pred"].to_numpy()

    _scatter_pair(ax1, x_hc, y_hc, "h_centroid", "blue")
    _scatter_pair(ax1, x_vc, y_vc, "v_centroid", "gold")

    all_x = np.concatenate([x_hc[np.isfinite(x_hc)], x_vc[np.isfinite(x_vc)]]) if len(df) else np.array([0,1])
    all_y = np.concatenate([y_hc[np.isfinite(y_hc)], y_vc[np.isfinite(y_vc)]]) if len(df) else np.array([0,1])
    mask_valid = (all_x != -1) & (all_y != -1)
    lo = float(min(all_x[mask_valid].min(), all_y[mask_valid].min())) if mask_valid.any() else 0.0
    hi = float(max(all_x[mask_valid].max(), all_y[mask_valid].max())) if mask_valid.any() else 1.0
    ax1.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax1.set_xlabel("Ground truth"); ax1.set_ylabel("Prediction"); ax1.set_title("Centroid parity")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi); ax1.set_aspect("equal", adjustable="box")
    ax1.legend(); ax1.grid(alpha=0.3)
    if save_dir: fig1.savefig(save_dir / "centroid_parity.png", dpi=150, bbox_inches="tight")
    else: plt.show()
    plt.close(fig1)

    # 2) Width parity (square)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    x_hw, y_hw = df["h_width"].to_numpy(), df["h_width_pred"].to_numpy()
    x_vw, y_vw = df["v_width"].to_numpy(), df["v_width_pred"].to_numpy()

    _scatter_pair(ax2, x_hw, y_hw, "h_width", "blue")
    _scatter_pair(ax2, x_vw, y_vw, "v_width", "gold")

    all_x = np.concatenate([x_hw[np.isfinite(x_hw)], x_vw[np.isfinite(x_vw)]]) if len(df) else np.array([0,1])
    all_y = np.concatenate([y_hw[np.isfinite(y_hw)], y_vw[np.isfinite(y_vw)]]) if len(df) else np.array([0,1])
    mask_valid = (all_x != -1) & (all_y != -1)
    lo = float(min(all_x[mask_valid].min(), all_y[mask_valid].min())) if mask_valid.any() else 0.0
    hi = float(max(all_x[mask_valid].max(), all_y[mask_valid].max())) if mask_valid.any() else 1.0
    ax2.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax2.set_xlabel("Ground truth"); ax2.set_ylabel("Prediction"); ax2.set_title("Width parity")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi); ax2.set_aspect("equal", adjustable="box")
    ax2.legend(); ax2.grid(alpha=0.3)
    if save_dir: fig2.savefig(save_dir / "width_parity.png", dpi=150, bbox_inches="tight")
    else: plt.show()
    plt.close(fig2)
    
    
