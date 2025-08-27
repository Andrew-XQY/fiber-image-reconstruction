import os
from xflow.extensions.style.aps import *
import json
import numpy as np
import matplotlib.pyplot as plt

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
) -> None:
    """
    If mode='flattened': model expects flattened input and returns flattened output.
    Images are assumed square and reshaped for visualization.
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

    def _save(arr2d: np.ndarray, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(arr2d, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.axis("off")
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
    
    
def plot_history_curves(folder: str,
                        metrics: str,
                        out_dir: str = "results/plots",
                        epoch_range: list[int] | tuple[int, int] | None = None,
                        smooth: bool = False,
                        show_minor_ticks: bool = False,
                        use_line_styles: bool = False) -> str:
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

    folder_path = Path(folder)
    files = sorted(folder_path.glob("*_history.json"))
    if not files:
        raise FileNotFoundError(f"No *_history.json files found in: {folder_path}")

    APS_COLORS = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
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
        if smooth:
            y = _smooth_same(y)
        x = np.arange(start + 1, end + 1 + 1)  # show epochs as 1-based on x-axis

        color = APS_COLORS[i % len(APS_COLORS)]
        if use_line_styles:
            style = LINE_STYLES[i % len(LINE_STYLES)]
            ax.plot(x, y, label=model, linewidth=1.8, color=color, linestyle=style)
        else:
            ax.plot(x, y, label=model, linewidth=1.8, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metrics)
    ax.set_title(metrics)
    # grid OFF; legend without frame
    if len(groups) > 1:
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
    fig.savefig(pdf_path, bbox_inches="tight")
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

    # Minimal APS-like look: clean axes, inward ticks, optional grid.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.boxplot(data, labels=models, showmeans=True)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
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

def plot_metrics_grouped_bars_by_model(root_dir: str, metrics: list[str], out_dir: str, with_std: bool = True):
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

    # aggregates
    avg = {(m, met): sum(values[(m, met)]) / len(values[(m, met)]) for m in models for met in metrics}
    err = {(m, met): sum(stds[(m, met)]) / len(stds[(m, met)])     for m in models for met in metrics}

    # plot
    K = len(metrics)
    width = 0.8 / K
    x = list(range(len(models)))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for j, met in enumerate(metrics):
        xs = [i - 0.4 + width/2 + j*width for i in x]
        heights = [avg[(m, met)] for m in models]
        if with_std:
            yerrs = [err[(m, met)] for m in models]
            ax.bar(xs, heights, width=width, yerr=yerrs, capsize=3, label=met)
        else:
            ax.bar(xs, heights, width=width, label=met)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
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
    fig.savefig(pdf_path, format="pdf")
    plt.show()
    return str(pdf_path)