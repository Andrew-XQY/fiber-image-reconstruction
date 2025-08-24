import os
from xflow.extensions.style.aps import *
import json
import numpy as np

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
    return [os.path.join(abs_dir_path, name)
            for name in os.listdir(abs_dir_path)
            if os.path.isdir(os.path.join(abs_dir_path, name))]
    
    



import torch
import os
from pathlib import Path
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, List
from utils import SAMPLE_FLATTENED, REGRESSION, GAN 

# ---- helpers ---------------------------------------------------------------
PARAM_KEYS = ["h_centroid", "v_centroid", "h_width", "v_width"]


def evaluate_to_csv(
    model: torch.nn.Module,
    test_loader,                 # yields (inputs, targets), both (B,C,H,W)
    device: torch.device | str,
    extract_beam_parameters,     # returns dict or None
    mode: str,                   # "img2img" or "regression"
    csv_path: str | Path,
) -> pd.DataFrame:
    model.eval()
    rows = []
    csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)

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
                else:
                    raise ValueError("mode must be 'img2img' or 'regression'")

                # ground truth params (from target image)
                gt_res = extract_beam_parameters(targets[i])
                gt_vals = {k: gt_res.get(k, -1.0) for k in PARAM_KEYS} if isinstance(gt_res, dict) else {k: -1.0 for k in PARAM_KEYS}

                # row
                row = {
                    **gt_vals,
                    **{f"{k}_pred": v for k, v in pred_vals.items()},
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[OK] {len(df)} rows -> {csv_path}")
    return df