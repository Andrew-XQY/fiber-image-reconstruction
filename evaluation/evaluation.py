import os
import glob
from xflow.extensions.style.aps import *
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from pathlib import Path
import pandas as pd
import shutil
import math

from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, List, Optional, Mapping, Union
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
    test_loader,                 # yields (inputs, targets) OR (inputs, params, targets) when mode='regression'
    device: torch.device | str,
    extract_beam_parameters,     # returns dict or None
    mode: str,                   # "img2img" or "regression" or "flattened"
) -> pd.DataFrame:
    model.eval()
    rows = []

    def _vec_to_params(x):
        if isinstance(x, torch.Tensor):
            v = x.reshape(-1).detach().cpu().tolist()
        else:
            v = list(x)
        return {
            "h_centroid": v[0] if len(v) > 0 else -1.0,
            "v_centroid": v[1] if len(v) > 1 else -1.0,
            "h_width":    v[2] if len(v) > 2 else -1.0,
            "v_width":    v[3] if len(v) > 3 else -1.0,
        }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
            # Support (inputs, targets) and (inputs, params, targets) in regression mode
            if mode == "regression" and isinstance(batch, (tuple, list)) and len(batch) == 3:
                inputs, params, targets = batch
            else:
                inputs, targets = batch
                params = None

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
                    pred_vals = _vec_to_params(outputs[i])
                elif mode == "flattened":
                    pred_vec = outputs[i].reshape(-1).detach().cpu().numpy()
                    side = int(np.sqrt(pred_vec.shape[0]))
                    pred_img = pred_vec.reshape(side, side)
                    pred_res = extract_beam_parameters(torch.from_numpy(pred_img))
                    pred_vals = {k: pred_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(pred_res, dict) else {k: -1.0 for k in PARAM_KEYS}
                else:
                    raise ValueError("mode must be 'img2img', 'regression', or 'flattened'")

                # ground truth params
                if mode == "flattened":
                    gt_vec = targets[i].reshape(-1).detach().cpu().numpy()
                    side = int(np.sqrt(gt_vec.shape[0]))
                    gt_img = gt_vec.reshape(side, side)
                    gt_res = extract_beam_parameters(torch.from_numpy(gt_img))
                    gt_vals = {k: gt_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(gt_res, dict) else {k: -1.0 for k in PARAM_KEYS}
                elif mode == "regression" and params is not None:
                    gt_vals = _vec_to_params(params[i])
                else:
                    gt_res = extract_beam_parameters(targets[i])
                    gt_vals = {k: gt_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(gt_res, dict) else {k: -1.0 for k in PARAM_KEYS}

                rows.append({
                    **gt_vals,
                    **{f"{k}_pred": v for k, v in pred_vals.items()},
                })

    df = pd.DataFrame(rows)
    return df

 

# def add_beamparam_metrics(df: pd.DataFrame, metrics=('RMSE','MSE','MAE')) -> pd.DataFrame:
#     allowed = {'RMSE','MSE','MAE'}
#     metrics = [m.upper() for m in metrics]
#     if any(m not in allowed for m in metrics):
#         raise ValueError(f"metrics must be subset of {allowed}")

#     out = df.copy()
#     params = [c for c in out.columns if not c.endswith('_pred') and f"{c}_pred" in out.columns]

#     for p in params:
#         gt = out[p]
#         pr = out[f"{p}_pred"]
#         valid = (gt != -1) & (pr != -1)
#         err = pr - gt

#         if 'MSE' in metrics:
#             col = f"{p}_MSE"
#             out[col] = -1.0
#             out.loc[valid, col] = (err.pow(2))[valid]

#         if 'RMSE' in metrics:
#             col = f"{p}_RMSE"
#             out[col] = -1.0
#             out.loc[valid, col] = (err.abs())[valid]  # sqrt(MSE) per-sample == |err|

#         if 'MAE' in metrics:
#             col = f"{p}_MAE"
#             out[col] = -1.0
#             out.loc[valid, col] = (err.abs())[valid]

#     for m in metrics:
#         cols = [f"{p}_{m}" for p in params]
#         mean_col = f"{m}_mean"
#         out[mean_col] = out[cols].mean(axis=1)
#         out.loc[(out[cols] == -1).any(axis=1), mean_col] = -1.0

#     return out

def add_beamparam_metrics(df: pd.DataFrame, metrics=('RMSE','MSE','MAE')) -> pd.DataFrame:
    import numpy as np
    out = df.copy()
    metrics = [m.upper() for m in metrics]
    allowed = {'RMSE','MSE','MAE'}
    if any(m not in allowed for m in metrics):
        raise ValueError(f"metrics must be subset of {allowed}")

    params = [c for c in out.columns if not c.endswith('_pred') and f"{c}_pred" in out.columns]

    for p in params:
        gt = out[p].astype(float)
        pr = out[f"{p}_pred"].astype(float)
        valid = (gt != -1) & (pr != -1)
        err = pr - gt
        # per-parameter errors (use NaN for invalid)
        if 'MSE' in metrics:
            out[f"{p}_MSE"] = np.where(valid, (err ** 2), np.nan)
        if 'MAE' in metrics:
            out[f"{p}_MAE"] = np.where(valid, err.abs(), np.nan)

    # row-wise means across parameters
    if 'MSE' in metrics:
        mse_cols = [f"{p}_MSE" for p in params]
        out["MSE_mean"] = out[mse_cols].mean(axis=1)
    if 'RMSE' in metrics:
        mse_cols = [f"{p}_MSE" for p in params]
        out["RMSE_mean"] = out[mse_cols].mean(axis=1).pow(0.5)   # sqrt(mean of squares)
    if 'MAE' in metrics:
        mae_cols = [f"{p}_MAE" for p in params]
        out["MAE_mean"] = out[mae_cols].mean(axis=1)

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
    x: torch.Tensor,              # (C,H,W) or (1,C,H,W) or (N,) / (1,N) for flattened
    y: torch.Tensor,              # (C,H,W) or (1,C,H,W) or (N,) / (1,N) for flattened
    device: str | torch.device,
    out_dir: str | Path = "results/single_sample",
    prefix: str = "index",        # -> index_0_input.png, index_1_target.png, index_2_pred.png
    channel: int = 0,             # used only for multi-channel image tensors
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mode: str = "img2img",        # "img2img" or "flattened"
    add_edge_hist: bool = False,  # add edge-fixed projection curves
    add_centroid: bool = False,   # add red dashed centroid lines
    use_gaussian_fit: bool = False,  # NEW: draw Gaussian-fit curve instead of raw histogram
    edge_line_width: float = 3.0,  # control orange line width for edge projections
    edge_hist_scale: float = 0.22  # control scale_frac for edge projection histograms
) -> None:
    """
    If mode='flattened': model expects flattened input and returns flattened output.
    Images are assumed square and reshaped for visualization.

    add_edge_hist: draw orange projection curves (sum along axes) inside the image
                   — y-projection fixed to the RIGHT edge, x-projection fixed to the BOTTOM edge.
    add_centroid:  draw red dashed lines at the centroid (Gaussian mean if use_gaussian_fit else weighted mean).
    use_gaussian_fit: if True, fit a Gaussian (via moments) to each projection (after background removal)
                      and plot that curve instead of the smoothed histogram.
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _to_2d(t: torch.Tensor) -> np.ndarray:
        """Return a (H,W) numpy array for plotting."""
        t = t.detach().cpu()
        if mode == "flattened":
            arr = t.reshape(-1).numpy()
            N = arr.size
            side = math.isqrt(N)
            if side * side != N:
                raise ValueError(f"Flattened length {N} is not a perfect square.")
            return arr.reshape(side, side)
        # image modes
        if t.ndim == 3:         # (C,H,W)
            return t[channel].numpy()
        if t.ndim == 2:         # (H,W)
            return t.numpy()
        raise TypeError(f"Expected (C,H,W), (H,W), or flattened 1D; got shape {tuple(t.shape)}")

    def _overlay_edge_projections(ax: plt.Axes, img: np.ndarray, draw_centroid: bool, use_gauss: bool, line_width: float = 3.0, edge_hist_scale: float = 0.22) -> None:
        """
        Edge-fixed orange curves; optional red dashed centroid lines.
        Anchors curves to exact image edges and zero-clips tiny tails.
        """
        img = np.asarray(img, dtype=float)
        H, W = img.shape
        scale_frac = edge_hist_scale # inward extent as a fraction of height/width
        eps = 1e-12                 # numeric guard
        zero_floor = 1e-6           # values below this (after normalization) are set to 0

        # ---- Background handling ----
        img_min = float(img.min())
        img_max = float(img.max())
        if img_min < 0.0:
            img = (img - img_min) / (img_max - img_min + eps)  # rescale to [0,1]
        else:
            img = img - img_min  # subtract uniform background

        # ---- Projections (onto x and y) ----
        x_proj = img.sum(axis=0)   # length W
        y_proj = img.sum(axis=1)   # length H

        # ---- Light smoothing (reflect-padded moving average) ----
        def _smooth1d(v: np.ndarray, k: int = 7) -> np.ndarray:
            k = max(3, int(k) | 1)  # odd, >=3
            pad = k // 2
            vp = np.pad(v, (pad, pad), mode="reflect")
            kernel = np.ones(k, dtype=float) / k
            return np.convolve(vp, kernel, mode="valid")

        x_sm = np.clip(_smooth1d(x_proj, k=7), 0.0, None)
        y_sm = np.clip(_smooth1d(y_proj, k=7), 0.0, None)

        # ---- Choose curve: Gaussian fit (moments) or smoothed histogram ----
        def _gaussian_from_moments(v: np.ndarray) -> tuple[np.ndarray, float, float]:
            n = v.size
            idx = np.arange(n, dtype=float)
            tot = v.sum()
            if tot <= eps:
                mu = (n - 1) / 2.0
                sigma = max(n / 6.0, 1e-6)
                g = np.zeros(n, dtype=float)
                return g, mu, sigma
            mu = (idx * v).sum() / tot
            var = ((idx - mu) ** 2 * v).sum() / (tot + eps)
            sigma = max(np.sqrt(max(var, eps)), 1e-6)
            g = np.exp(-0.5 * ((idx - mu) / sigma) ** 2)
            return g, mu, sigma

        if use_gauss:
            x_curve, mu_x, _ = _gaussian_from_moments(x_sm)
            y_curve, mu_y, _ = _gaussian_from_moments(y_sm)
        else:
            x_curve = x_sm
            y_curve = y_sm
            xtot = x_sm.sum()
            ytot = y_sm.sum()
            mu_x = (np.arange(W) * x_sm).sum() / (xtot + eps) if xtot > 0 else (W - 1) / 2.0
            mu_y = (np.arange(H) * y_sm).sum() / (ytot + eps) if ytot > 0 else (H - 1) / 2.0

        # ---- Force baseline to zero & normalize ----
        x_curve = np.maximum(x_curve - x_curve.min(), 0.0)
        y_curve = np.maximum(y_curve - y_curve.min(), 0.0)
        x_norm = x_curve / (x_curve.max() + eps)
        y_norm = y_curve / (y_curve.max() + eps)

        # ---- Hard zero tiny tails so they sit ON the axes ----
        x_norm[x_norm < zero_floor] = 0.0
        y_norm[y_norm < zero_floor] = 0.0
        # Also pin endpoints to 0 to avoid smoothing/Gaussian tail lift
        if x_norm.size > 0:
            x_norm[0] = 0.0
            x_norm[-1] = 0.0
        if y_norm.size > 0:
            y_norm[0] = 0.0
            y_norm[-1] = 0.0

        # ---- Map to exact EDGE coordinates (no half-pixel offset) ----
        # Bottom edge is y = -0.5; move inward by +inset
        bottom_inset = x_norm * (scale_frac * H)
        xs = np.arange(W, dtype=float)          # pixel centers along x
        ys = (H - 0.5) - bottom_inset              # start exactly on bottom edge

        # Right edge is x = W - 0.5; move inward by -inset
        right_inset = y_norm * (scale_frac * W)
        ys2 = np.arange(H, dtype=float)         # pixel centers along y
        xs2 = (W - 0.5) - right_inset           # start exactly on right edge

        # ---- Draw (no anti-aliasing, butt caps) to avoid subpixel bleed ----
        line_kw = dict(linewidth=line_width, color="orange", clip_on=True,
                antialiased=False, solid_capstyle="butt", solid_joinstyle="miter")
        ax.plot(xs, ys, **line_kw)
        ax.plot(xs2, ys2, **line_kw)

        if draw_centroid:
            ax.axvline(mu_x, linestyle="--", linewidth=1.2, color="red",
                    antialiased=False)
            ax.axhline(mu_y, linestyle="--", linewidth=1.2, color="red",
                    antialiased=False)

    def _save(arr2d: np.ndarray, path: Path, with_hist: bool = False, with_centroid: bool = False) -> None:
        H, W = arr2d.shape
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw image exactly from edge to edge; no axis padding.
        ax.imshow(
            arr2d, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal",
            extent=(-0.5, W - 0.5, H - 0.5, -0.5), interpolation="nearest"
        )
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)         # inverted to match image coords
        ax.margins(0.0)                    # no extra margin
        ax.autoscale(False)                # keep limits fixed
        ax.axis("off")

        if with_hist:
            _overlay_edge_projections(ax, arr2d, draw_centroid=with_centroid, use_gauss=use_gaussian_fit, line_width=edge_line_width, edge_hist_scale=edge_hist_scale)

        # Remove figure-frame padding as well
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    with torch.inference_mode():
        if mode == "flattened":
            xbat = x.to(device).float().reshape(1, -1)  # (1,N)
            pred = model(xbat)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = pred.reshape(-1)  # ensure (N,)
        else:
            xbat = x.to(device).float()
            if xbat.ndim == 3:          # (C,H,W) -> (1,C,H,W)
                xbat = xbat.unsqueeze(0)
            pred = model(xbat)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            # ensure (C,H,W) for visualization
            pred = pred[0] if pred.ndim == 4 else pred

    x_img    = _to_2d(x[0] if (x.ndim == 4 and mode != "flattened") else x)
    y_img    = _to_2d(y[0] if (y.ndim == 4 and mode != "flattened") else y)
    pred_img = _to_2d(pred)

    # Input: unchanged (no overlays)
    _save(x_img, out_dir / f"{prefix}_0_input.png", with_hist=False, with_centroid=False)
    # Target & Pred: add overlays as requested
    _save(y_img, out_dir / f"{prefix}_1_target.png",
          with_hist=add_edge_hist, with_centroid=add_centroid)
    _save(pred_img, out_dir / f"{prefix}_2_pred.png",
          with_hist=add_edge_hist, with_centroid=add_centroid)




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
    
    
def plot_history_curves(folder: str,
                        metrics: str,
                        out_dir: str = "results/plots",
                        epoch_range: list[int] | tuple[int, int] | None = None,
                        smooth: bool = False,
                        show_minor_ticks: bool = False,
                        use_line_styles: bool = False,
                        show_grid: bool = False,
                        line_width: float = 1.8,
                        advanced_smooth: int | None = None,
                        show_legend: bool = True) -> str:
    """
    Group *_history.json by the first token before '-'. For each group, compute an epoch-wise
    average of `metrics` across runs (using only runs that contain that epoch). Plot ONE line per group.

    Args:
        folder: Directory containing files like Pix2pix-YYYYmmddHHMMSS_history.json
        metrics: JSON key to plot (e.g., "val_overall_mae")
        out_dir: Output directory for the PDF
        epoch_range: Optional [start_idx, end_idx] (0-based, inclusive). If None, plot full length.
        smooth: If True, plot a smoothed curve (moving average); otherwise plot raw line.

    Returns:
        Path to the saved PDF.
    """
    def _smooth_same(y: np.ndarray) -> np.ndarray:
        """Moving average with 'same' length, NaN-aware, odd window."""
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 3:
            return y
        # choose an odd window ~ n/20, clipped to [3, n or n-1]
        w = max(3, int(round(n / 20)) or 3)
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n-1 if n % 2 == 0 else n
        if w < 3:
            return y
        kernel = np.ones(w, dtype=float)
        good = np.isfinite(y).astype(float)
        y_filled = np.where(np.isfinite(y), y, 0.0)
        num = np.convolve(y_filled, kernel, mode="same")
        den = np.convolve(good, kernel, mode="same")
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        return out

    def _advanced_smooth(y: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
        """Savitzky-Golay smoothing, NaN-aware."""
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError("scipy is required for advanced smoothing. Please install it via 'pip install scipy'.")
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < window_length or window_length < 3:
            return y
        # window_length must be odd and <= n
        if window_length % 2 == 0:
            window_length += 1
        if window_length > n:
            window_length = n if n % 2 == 1 else n - 1
        # Fill NaNs for filtering, then restore NaNs after
        nan_mask = ~np.isfinite(y)
        y_filled = np.where(nan_mask, np.nanmean(y), y)
        y_smooth = savgol_filter(y_filled, window_length=window_length, polyorder=polyorder, mode='interp')
        y_smooth[nan_mask] = np.nan
        return y_smooth

    folder_path = Path(folder)
    files = sorted(folder_path.glob("*_history.json"))
    if not files:
        raise FileNotFoundError(f"No *_history.json files found in: {folder_path}")

    APS_COLORS = [
        "#4477AA", "#FFA500", "#228833", "#CCBB44", "#66CCEE",
        "#AA3377", "#BBBBBB", "#000000", "#44AA99", "#FFA500",
        "#332288", "#88CCEE"
    ]


    # Collect metric sequences by model prefix
    groups: dict[str, list[list[float]]] = {}
    for f in files:
        name = f.stem  # e.g., "Pix2pix-20250825203106_history"
        model = name.split("-")[0] if "-" in name else name.replace("_history", "")
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[skip] Failed to read {f.name}: {e}")
            continue

        seq = data.get(metrics, None)
        if isinstance(seq, list) and len(seq) > 0:
            groups.setdefault(model, []).append(seq)
        else:
            print(f"[skip] '{metrics}' not found or empty in {f.name}")

    if not groups:
        raise ValueError(f"No valid '{metrics}' series found in: {folder_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path / f"{metrics}.pdf"

    fig, ax = plt.subplots(figsize=(6, 4))
    # Experienced researchers often use both color and line style to distinguish many lines.
    # Common line styles: solid, dashed, dashdot, dotted, etc.
    LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]

    for i, (model, seqs) in enumerate(groups.items()):
        # Skip Pix2pix train_loss
        if model == "Pix2pix" and metrics == "train_loss":
            continue
        # Epoch-wise average (use only runs that have t)
        max_len = max(len(s) for s in seqs)
        mean_curve = []
        for t in range(max_len):
            vals = []
            for s in seqs:
                if t < len(s) and s[t] is not None:
                    try:
                        v = float(s[t])
                        vals.append(v)
                    except (TypeError, ValueError):
                        pass
            mean_curve.append(np.mean(vals) if vals else np.nan)

        # Apply epoch_range slicing (0-based, inclusive)
        if epoch_range is not None:
            start = max(0, int(epoch_range[0]))
            end = min(max_len - 1, int(epoch_range[1]))
            if start > end:
                raise ValueError(f"Invalid epoch_range: start {start} > end {end}")
        else:
            start, end = 0, max_len - 1

        y = np.array(mean_curve[start:end + 1], dtype=float)
        # Advanced smoothing for MAE metrics
        if advanced_smooth is not None and "MAE" in metrics.upper():
            y = _advanced_smooth(y, window_length=int(advanced_smooth))
        elif smooth:
            y = _smooth_same(y)
        x = np.arange(start + 1, end + 1 + 1)  # show epochs as 1-based on x-axis

        color = APS_COLORS[i % len(APS_COLORS)]
        # Replace underscores with hyphens for visualization label
        vis_model = model.replace('_', '-')
        if use_line_styles:
            style = LINE_STYLES[i % len(LINE_STYLES)]
            ax.plot(x, y, label=vis_model, linewidth=line_width, color=color, linestyle=style)
        else:
            ax.plot(x, y, label=vis_model, linewidth=line_width, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metrics)
    ax.set_title(metrics)
    # Optional light dashed grid (both axes)
    if show_grid:
        ax.grid(True, linestyle="--", color="#cccccc", alpha=0.5, axis="both")
    # legend without frame
    if show_legend and len(groups) > 1:
        ax.legend(loc="best", fontsize=8, frameon=False)
    # Add minor ticks and ticks on all four axes if requested
    if show_minor_ticks:
        ax.xaxis.set_minor_locator(plt.AutoMinorLocator())
        ax.yaxis.set_minor_locator(plt.AutoMinorLocator())
        ax.tick_params(axis="both", which="minor", length=4, direction="in", top=True, right=True)
        ax.tick_params(axis="both", which="major", length=7, direction="in", top=True, right=True)
    else:
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return str(pdf_path)


def plot_metric_box_by_model(root_dir: str, metric: str, out_dir: str, show_grid: bool = False):
    """
    Recursively scan `root_dir` for files named like MODEL-YYYYMMDDhhmmss_summary.json.
    For `metric` (e.g. 'RMSE_mean'), read `<metric>_avg` and `<metric>_std`.
    Plot a box plot per model using the list of `<metric>_avg` values across its files.
    Save a PDF to `out_dir` named `{metric}_by_model_boxplot.pdf`.
    Also print the per-model mean of `<metric>_avg` and `<metric>_std`.
    """
    root = Path(root_dir)
    files = list(root.rglob("*_summary.json"))
    if not files:
        print("No *_summary.json files found.")
        return None

    avg_key = f"{metric}_avg"
    std_key = f"{metric}_std"
    avg_samples, std_samples = {}, {}

    for p in files:
        model = p.name.split("-")[0]
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if avg_key in obj and std_key in obj:
            avg_samples.setdefault(model, []).append(float(obj[avg_key]))
            std_samples.setdefault(model, []).append(float(obj[std_key]))

    if not avg_samples:
        print(f"No files had keys {avg_key} and {std_key}.")
        return None

    models = sorted(avg_samples.keys())
    data = [avg_samples[m] for m in models]
    # Replace underscores with hyphens for visualization
    vis_models = [m.replace('_', '-') for m in models]

    # Minimal APS-like look: clean axes, inward ticks, optional grid.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.boxplot(data, labels=vis_models, showmeans=True)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    # Remove bounding box (hide all spines except bottom and left)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(direction="in", top=False, right=False)
    if show_grid:
        ax.yaxis.grid(True, linestyle="--", color="#cccccc", alpha=0.5)
    fig.tight_layout()

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir_path / f"{metric}_by_model_boxplot.pdf"
    fig.savefig(pdf_path, format="pdf")
    plt.show()

    for m in models:
        avgs = avg_samples[m]
        stds = std_samples.get(m, [])
        avg_mean = sum(avgs) / len(avgs)
        std_mean = (sum(stds) / len(stds)) if stds else None
        print(f"{m}: avg_mean={avg_mean:.6g}, std_mean={None if std_mean is None else round(std_mean,6)}, n={len(avgs)}")

    return str(pdf_path)

def plot_metrics_grouped_bars_by_model(root_dir: str, metrics: list[str], out_dir: str, with_std: bool = True, sort_by_mae: bool = False):
    """
    Scan `root_dir` recursively for MODEL-YYYYMMDDhhmmss_summary.json.
    For each metric in `metrics`, read `<metric>_avg` and `<metric>_std`,
    average across files per model, and draw grouped bars (one group per model,
    one bar per metric). If `with_std` is True, show error bars using the
    averaged `<metric>_std`. Saves `{out_dir}/metrics_by_model_barplot.pdf`.
    """
    root = Path(root_dir)
    files = list(root.rglob("*_summary.json"))
    if not files:
        print("No *_summary.json files found.")
        return None

    values, stds = {}, {}
    for p in files:
        model = p.name.split("-")[0]
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for met in metrics:
            a, s = f"{met}_avg", f"{met}_std"
            if a in data and s in data:
                values.setdefault((model, met), []).append(float(data[a]))
                stds.setdefault((model, met), []).append(float(data[s]))

    # keep models that have all requested metrics
    models = sorted({m for (m, _) in values})
    models = [m for m in models if all((m, met) in values for met in metrics)]
    if not models:
        print("No models have all requested metrics.")
        return None
    
    # aggregates (calculate before sorting)
    avg = {(m, met): sum(values[(m, met)]) / len(values[(m, met)]) for m in models for met in metrics}
    err = {(m, met): sum(stds[(m, met)]) / len(stds[(m, met)])     for m in models for met in metrics}
    
    # Sort by MAE if requested, otherwise keep current order
    if sort_by_mae:
        # Find MAE metric (case insensitive)
        mae_metric = None
        for met in metrics:
            if 'mae' in met.lower():
                mae_metric = met
                break
        if mae_metric:
            models = sorted(models, key=lambda m: avg[(m, mae_metric)])
    
    # Replace underscores with hyphens for visualization
    vis_models = [m.replace('_', '-') for m in models]

    # plot
    K = len(metrics)
    width = 0.8 / K
    x = list(range(len(models)))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    # Color palette for bars (reuse APS_COLORS from above if available)
    # APS colors: blue, orange, green, then rest
    # APS colors: blue, yellow, green, then rest (orange is #EE6677, yellow is #CCBB44)
    # BAR_COLORS = ["#27569F", "#FF9100", "#2F8814", "#EE6677", "#66CCEE", "#AA3377", "#BBBBBB", "#000000", "#44AA99", "#FFA500", "#332288", "#88CCEE"]
    BAR_COLORS = ["#228833", "#FFA500", "#4477AA"]
    
    # Find lowest RMSE_mean and MAE_mean values and their colors
    min_lines = []
    for target_metric in ["RMSE_MEAN", "MAE_MEAN"]:
        for j, met in enumerate(metrics):
            if met.upper() == target_metric:
                vals = [avg[(m, met)] for m in models]
                min_val = min(vals)
                color = BAR_COLORS[j % len(BAR_COLORS)]
                min_lines.append((min_val, color, met))
    # Plot horizontal dashed lines for min RMSE and MAE (background)
    for min_val, color, met in min_lines:
        ax.axhline(min_val, linestyle="--", color=color, alpha=0.7, linewidth=1.5, zorder=1)
    # Plot bars and collect bar colors (zorder=2 for bars)
    for j, met in enumerate(metrics):
        xs = [i - 0.4 + width/2 + j*width for i in x]
        heights = [avg[(m, met)] for m in models]
        bar_color = BAR_COLORS[j % len(BAR_COLORS)]
        # Legend label: remove '_mean' suffix if present
        legend_label = met[:-5] if met.endswith('_mean') else met
        if with_std:
            yerrs = [err[(m, met)] for m in models]
            ax.bar(xs, heights, width=width, yerr=yerrs, capsize=3, label=legend_label, color=bar_color, zorder=2)
        else:
            ax.bar(xs, heights, width=width, label=legend_label, color=bar_color, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(vis_models)
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False)

    # Full rectangular boundary box
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(direction="in", top=False, right=False)
    fig.tight_layout()

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / "metrics_by_model_barplot.pdf"
    fig.savefig(pdf_path, format="pdf", transparent=True)
    plt.show()
    
    # Return dictionary with final metric values
    result_dict = {}
    for model in models:
        result_dict[model] = {}
        for metric in metrics:
            mean_val = avg[(model, metric)]
            std_val = err[(model, metric)]
            result_dict[model][metric] = f"{mean_val:.6f} ± {std_val:.6f}"
    
    return result_dict




# Parse model size and time based on Slurm .out files
import re


def _parse_duration_to_seconds(s: str) -> float:
    """
    Parse durations like '40.05s', '1m 23.4s', '2h 3m 1s', '12:34', or '1:02:03' to seconds.
    """
    s = s.strip()
    # Unit-based forms: h/m/s
    unit_matches = re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', s, flags=re.I)
    if unit_matches:
        mult = {'h': 3600.0, 'm': 60.0, 's': 1.0}
        return sum(float(v) * mult[u.lower()] for v, u in unit_matches)

    # Colon forms: hh:mm:ss(.x) or mm:ss(.x)
    if ':' in s:
        try:
            parts = [float(p) for p in s.split(':')]
            if len(parts) == 3:
                h, m, sec = parts
            elif len(parts) == 2:
                h, (m, sec) = 0.0, parts
            else:
                return float(s)
            return h * 3600.0 + m * 60.0 + sec
        except ValueError:
            pass

    # Plain float seconds or trailing 's'
    s = s.rstrip().rstrip('sS').strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_training_logs(root_dir: Union[str, Path], save_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Scan `root_dir` recursively for .out files, extract model name, model size, and total training time (minutes).

    Extracts:
      - Model name from lines like: "Model:               TransmissionMatrix"
      - Model size from lines like: "Size:                256.00 MB" (kept as-is string)
        * If missing, falls back to estimating from "Parameters: <N> total" assuming float32 (4 bytes/param)
          and formats as "<XX.XX MB*".
      - Epoch times from lines like: "Epoch 47 completed in 40.05s - ..." and sums them.

    Args:
        root_dir: Folder to search under (recursive) for *.out files.
        save_csv: If provided, path to save the resulting CSV.

    Returns:
        pd.DataFrame with columns: ['model', 'size', 'total_train_time_min']
    """
    root = Path(root_dir)
    rows = []

    for fp in sorted(root.rglob("*.out")):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # Skip unreadable files
            continue

        # Model name
        m = re.search(r'(?mi)^\s*Model:\s*(.+?)\s*$', text)
        model_name = m.group(1).strip() if m else ""

        # Size string (prefer direct 'Size:' line)
        sm = re.search(r'(?mi)^\s*Size:\s*([^\r\n]+)$', text)
        size_str = sm.group(1).strip() if sm else ""

        # Fallback: estimate from Parameters: <num> total  (float32)
        if not size_str:
            pm = re.search(r'(?mi)^\s*Parameters:\s*([\d_,.]+)\s*total', text)
            if pm:
                num_txt = pm.group(1).replace(",", "").replace("_", "")
                try:
                    n_params = float(num_txt)
                    mb = n_params * 4.0 / (1024.0 ** 2)
                    size_str = f"{mb:.2f} MB*"
                except Exception:
                    pass

        # Epoch durations
        # Match 'Epoch <num> completed in <time>' and capture the <time> up to ' -' or EOL
        epoch_times = re.findall(r'(?mi)Epoch\s+\d+\s+completed\s+in\s+([^\r\n-]+)', text)
        total_seconds = sum(_parse_duration_to_seconds(t) for t in epoch_times)
        total_minutes = round(total_seconds / 60.0, 3)

        rows.append({
            "model": model_name,
            "size": size_str,
            "total_train_time_min": total_minutes,
        })

    df = pd.DataFrame(rows, columns=["model", "size", "total_train_time_min"])
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
    return df


def plot_model_error_boxplot(
    input_dir: str,
    metric: str,
    output_pdf: str,
    order_by: str = "alpha",        # "alpha" or "metric"
    show_mean_marker: bool = True,  # green triangle mean marker
    spacing: float = 1.3            # >1.0 increases horizontal gaps
) -> None:
    """
    Create plain per-model box plots from CSV files in `input_dir`.

    Files must be named like: <MODEL>-<anything>.csv (e.g., CAE-2025082504.csv)
    `metric` must match a column name (case-insensitive).

    If each run for a model has a per-sample ID column (any of sample_id/image_id/id/index)
    and IDs are consistent across runs, the function averages the metric per sample across runs
    before plotting. Otherwise, it concatenates per-sample metrics across runs.

    Styling: white-filled boxes, median + whiskers (no fliers), optional green triangle mean.
    """
    csv_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    def find_col(df: pd.DataFrame, name: str) -> Optional[str]:
        m = {c.lower(): c for c in df.columns}
        return m.get(name.lower())

    id_candidates = ["sample_id", "image_id", "id", "index"]
    per_model: Dict[str, List[pd.DataFrame]] = {}

    # Load
    for p in csv_paths:
        model = Path(p).stem.split("-", 1)[0]
        df = pd.read_csv(p)

        mcol = find_col(df, metric)
        if mcol is None:
            raise KeyError(f"Metric '{metric}' not found in {p}. Columns: {list(df.columns)}")

        idcol = None
        for cand in id_candidates:
            c = find_col(df, cand)
            if c is not None:
                idcol = c
                break

        sub = df[[mcol] + ([idcol] if idcol else [])].copy()
        sub[mcol] = pd.to_numeric(sub[mcol], errors="coerce")
        sub = sub.dropna(subset=[mcol])
        if idcol:
            sub = sub.rename(columns={idcol: "_sample_id_"})
        per_model.setdefault(model, []).append(sub)

    # Aggregate per model
    model_names, data_arrays = [], []
    for model, runs in sorted(per_model.items()):
        if not runs:
            continue
        all_have_ids = all("_sample_id_" in r.columns for r in runs)
        if all_have_ids:
            merged = None
            for i, r in enumerate(runs):
                r_i = r.set_index("_sample_id_").sort_index()
                r_i.columns = [f"run{i}"]
                merged = r_i if merged is None else merged.join(r_i, how="inner")
            vals = merged.mean(axis=1).values  # per-sample mean across seeds
        else:
            vals = pd.concat([r[r.columns[0]] for r in runs], axis=0).values

        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        model_names.append(model)
        data_arrays.append(vals)

    if not data_arrays:
        raise RuntimeError("No valid metric data collected for plotting.")

    # Order
    if order_by.lower() == "metric":
        med = [float(np.median(a)) for a in data_arrays]
        order = np.argsort(med)  # lower is better
    else:  # alphabetical
        order = np.argsort(np.array(model_names, dtype=object))
    model_names = [model_names[i] for i in order]
    data_arrays = [data_arrays[i] for i in order]

    # IBIC-friendly minimal styling
    plt.rcParams.update({
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(3.4, 2.6), dpi=300)  # ~0.85 column width
    ax = plt.gca()
    ax.set_axisbelow(True)  # grid behind boxes
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Positions (add extra spacing)
    n = len(model_names)
    positions = 1 + np.arange(n) * spacing

    # Plain boxplot with white fill; no fliers
    meanprops = dict(marker="^", markerfacecolor="green", markeredgecolor="green", markersize=4)
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        vert=True,
        patch_artist=True,     # enables facecolor
        showmeans=show_mean_marker,
        meanprops=meanprops if show_mean_marker else None,
        showfliers=False,
        whis=1.5,
        widths=0.5
    )

    # White boxes (so grid doesn't "bleed" visually)
    for b in bp["boxes"]:
        b.set(facecolor="white", edgecolor="black", linewidth=1.0)
    for elem in ["medians", "whiskers", "caps"]:
        for a in bp[elem]:
            a.set(color="black", linewidth=1.0)

    # X labels: replace "_" with "-"
    labels = [name.replace("_", "-") for name in model_names]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(metric)

    fig.tight_layout()
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_metric_box_by_model_csv(
    root_dir: str,
    metric: str,
    out_dir: str,
    show_grid: bool = False,
    spacing: float = 1.8,             # larger gaps between boxes
    show_mean_marker: bool = True,    # small green triangle
    show_legend: bool = True,         # control legend display
    show_violin: bool = False,        # violin overlay (ignored if show_hist=True)
    show_hist: bool = False,          # histogram overlay instead of violin
    hist_bins: int = 20,              # histogram bin count
    hist_max_halfwidth: float = 0.35, # max half-width of hist bars
    overlay_clip_quantiles: Optional[Tuple[float, float]] = None,  # clip overlay only
    overlay_clip_max: Optional[float] = None,   # NEW: drop overlay values > this
    y_limit_quantiles: Optional[Tuple[float, float]] = None,       # robust axis limits
    use_symlog: bool = False,         # optional heavy-tail y-scale
    symlog_linthresh: float = 1e-2,   # linear threshold for symlog
    box_width: float = 0.45,          # width of the box in boxplot
    hist_alpha: float = 0.25,         # transparency of histogram bins
    sort_by_mae: bool = False         # sort models by MAE metric values
):
    # --- load & group ---
    root = Path(root_dir)
    csv_paths = sorted(root.glob("*.csv"))
    if not csv_paths:
        print("No *.csv files found.")
        return None

    def find_col(df: pd.DataFrame, name: str) -> Optional[str]:
        mapping = {c.lower(): c for c in df.columns}
        return mapping.get(name.lower())

    id_candidates = ["sample_id", "image_id", "id", "index"]
    per_model: Dict[str, List[pd.DataFrame]] = {}

    for p in csv_paths:
        model = p.stem.split("-", 1)[0]
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        mcol = find_col(df, metric)
        if mcol is None:
            continue
        idcol = next((find_col(df, cand) for cand in id_candidates if find_col(df, cand)), None)

        sub = df[[mcol] + ([idcol] if idcol else [])].copy()
        sub[mcol] = pd.to_numeric(sub[mcol], errors="coerce")
        sub = sub.dropna(subset=[mcol])
        if idcol:
            sub.rename(columns={idcol: "_sample_id_"}, inplace=True)
        per_model.setdefault(model, []).append(sub)

    # --- aggregate per model ---
    models = sorted(per_model.keys())
    model_data: Dict[str, np.ndarray] = {}
    for model in models:
        runs = per_model[model]
        if not runs:
            continue
        all_have_ids = all("_sample_id_" in r.columns for r in runs)
        if all_have_ids:
            merged = None
            for i, r in enumerate(runs):
                r_i = r.set_index("_sample_id_").sort_index()
                r_i.columns = [f"run{i}"]
                merged = r_i if merged is None else merged.join(r_i, how="inner")
            vals = merged.mean(axis=1).values
        else:
            vals = pd.concat([r[r.columns[0]] for r in runs], axis=0).values
        vals = vals[~np.isnan(vals)]
        if vals.size:
            model_data[model] = vals

    if not model_data:
        print(f"No valid metric '{metric}' found in CSV files.")
        return None

    # order + labels
    if sort_by_mae and 'mae' in metric.lower():
        # Sort by mean values of the current metric (ascending order for MAE)
        models = sorted(model_data.keys(), key=lambda m: np.mean(model_data[m]))
    else:
        models = sorted(model_data.keys(), key=lambda s: s.lower())
    data = [model_data[m] for m in models]
    vis_labels = [m.replace("_", "-") for m in models]

    # --- overlay-only clipping (does NOT affect boxplot data) ---
    def clip_quant(arr: np.ndarray, q: Tuple[float, float]) -> np.ndarray:
        lo = np.quantile(arr, q[0]); hi = np.quantile(arr, q[1])
        return arr[(arr >= lo) & (arr <= hi)]

    overlay_data: List[np.ndarray] = []
    for arr in data:
        ov = arr
        if overlay_clip_max is not None:
            ov = ov[ov <= overlay_clip_max]
        if overlay_clip_quantiles is not None and ov.size:
            ov = clip_quant(ov, overlay_clip_quantiles)
        overlay_data.append(ov)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.set_axisbelow(True)
    if show_grid:
        ax.yaxis.grid(True, linestyle="--", color="#cccccc", alpha=0.5)

    positions = 1 + np.arange(len(data)) * spacing

    # --- overlay first (behind boxes) ---
    if show_hist:
        for pos, vals in zip(positions, overlay_data):
            if vals.size == 0:
                continue
            counts, edges = np.histogram(vals, bins=hist_bins, density=True)
            if counts.max() <= 0:
                continue
            for c, y0, y1 in zip(counts, edges[:-1], edges[1:]):
                half_w = (c / counts.max()) * hist_max_halfwidth
                rect = Rectangle((pos - half_w, y0), 2 * half_w, y1 - y0,
                                 facecolor="0.7", edgecolor=None,
                                 alpha=hist_alpha, linewidth=0)
                # Do not add label, so bins are not included in legend
                ax.add_patch(rect)
    elif show_violin:
        v = ax.violinplot(overlay_data, positions=positions, widths=0.9,
                          showmeans=False, showmedians=False, showextrema=False,
                          bw_method="scott")
        for body in v["bodies"]:
            body.set_facecolor("0.7"); body.set_edgecolor("0.4")
            body.set_alpha(0.25); body.set_linewidth(0.5)

    # --- boxplot (full, unclipped data) ---
    bp = ax.boxplot(
        data, positions=positions, labels=None, vert=True,
        patch_artist=True, showmeans=show_mean_marker, showfliers=False,
        whis=1.5, widths=box_width,
        meanprops=dict(marker="^", markersize=3,
                       markerfacecolor="green", markeredgecolor="green"),
        medianprops=dict(color="orange", linewidth=1.2),
    )
    for b in bp["boxes"]:
        b.set(facecolor="white", edgecolor="black", linewidth=1.0)
    for elem in ["whiskers", "caps"]:
        for a in bp[elem]:
            a.set(color="black", linewidth=1.0)

    ax.set_xticks(positions); ax.set_xticklabels(vis_labels, rotation=0)
    ax.set_xlabel("Model"); ax.set_ylabel(metric)
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    ax.tick_params(direction="in", top=False, right=False)

    # optional robust y-limits & symlog
    if y_limit_quantiles is not None:
        all_vals = np.concatenate(data)
        ylo = np.quantile(all_vals, y_limit_quantiles[0])
        yhi = np.quantile(all_vals, y_limit_quantiles[1])
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            ax.set_ylim(ylo, yhi)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=symlog_linthresh)

    if show_legend:
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        handles = [
            Line2D([0], [0], color="orange", linewidth=1.2, label="Median"),
            Line2D([0], [0], marker="^", color="white",
                   markerfacecolor="green", markeredgecolor="green",
                   markersize=7, linestyle="None", label="Mean"),
        ]
        # Do not add histogram bin patch to legend
        if show_violin:
            handles.insert(0, mpatches.Patch(facecolor="0.7", edgecolor="0.4",
                                             alpha=0.25, label="Distribution (violin)"))
        ax.legend(handles=handles, loc="best", frameon=True,
                  facecolor="white", edgecolor="black")

    fig.tight_layout()
    out_dir_path = Path(out_dir); out_dir_path.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir_path.joinpath(f"{metric}_by_model_boxplot.pdf")
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    return str(pdf_path)