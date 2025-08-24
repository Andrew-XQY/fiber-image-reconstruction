import sys
import torch
import torchvision.utils as vutils
import math
from pathlib import Path


def metric_debug(extract_fn):
    def metric(pred, target):
        # --- DEBUG: force a constant output ---
        return {"val_debug_mae": 0.0, "val_debug_rmse": 0.0}
    return metric

def debug_extract_fn(img, **kwargs):
    try:
        print("[DEBUG] extract_fn input shape:", getattr(img, "shape", None), type(img))
        return extract_beam_parameters_flat(img, **kwargs)
    except Exception as e:
        print("[WARNING] extract_fn failed with:", e)
        return None

def make_beam_param_metric(extract_fn):
    def metric(pred, target):
        
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu()

        sums_abs, sums_sq, counts = {}, {}, {}
        B = len(pred)

        for i in range(B):
            p = extract_fn(pred[i])
            t = extract_fn(target[i])
            if p is None or t is None:
                continue  # skip invalid samples

            # Add "overall" from original values only
            p_vals = [float(v) for v in p.values()]
            t_vals = [float(v) for v in t.values()]
            if p_vals:
                p = {**p, "overall": math.fsum(p_vals) / len(p_vals)}
            if t_vals:
                t = {**t, "overall": math.fsum(t_vals) / len(t_vals)}

            for k in p.keys():
                if k not in t:
                    continue  # skip if target missing this key
                diff = float(p[k]) - float(t[k])
                sums_abs[k] = sums_abs.get(k, 0.0) + abs(diff)      # MAE parts
                sums_sq[k]  = sums_sq.get(k, 0.0)  + diff * diff    # MSE parts
                counts[k]   = counts.get(k, 0) + 1

        out = {}
        for k, n in counts.items():
            out[f"val_{k}_mae"]  = sums_abs[k] / n
            out[f"val_{k}_mse"]  = sums_sq[k] / n
            out[f"val_{k}_rmse"] = (sums_sq[k] / n) ** 0.5
        return out

    return metric

# --- Param-based metric (no image extraction) ---
def make_param_metric(keys=("h_centroid","v_centroid","h_width","v_width")):
    import torch, math

    def _to_vec(x):
        # x: Tensor/np/list shaped (4,) or (B,4). Return 1D list[float].
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        x = x.reshape(-1).tolist()
        return x

    def metric(pred, target):
        p = _to_vec(pred)
        t = _to_vec(target)
        n = min(len(p), len(t))
        if n < 4:  # nothing to score
            return {}

        # take first len(keys) values
        p = p[:len(keys)]
        t = t[:len(keys)]

        out = {}
        diffs = [pi - ti for pi, ti in zip(p, t)]
        for k, d in zip(keys, diffs):
            out[f"val_{k}_mae"]  = abs(d)
            out[f"val_{k}_mse"]  = d*d
            out[f"val_{k}_rmse"] = abs(d)**0.5 if d >= 0 else (d*d)**0.5  # same as sqrt(mse)

        # simple overall
        overall_mae = sum(abs(d) for d in diffs) / len(diffs)
        overall_mse = sum(d*d for d in diffs) / len(diffs)
        out["val_overall_mae"]  = overall_mae
        out["val_overall_mse"]  = overall_mse
        out["val_overall_rmse"] = overall_mse ** 0.5
        return out

    return metric


# Extract beam parameters
def extract_beam_parameters_flat(flat_img, **kwargs):
    from xflow.extensions.physics.beam import extract_beam_parameters
    """
    Adapter for extract_beam_parameters to handle flattened square images.
    Supports both NumPy arrays and PyTorch tensors.
    """
    import numpy as np
    # Get total number of elements robustly
    if hasattr(flat_img, "numel"):
        n = flat_img.numel()
    elif hasattr(flat_img, "size"):
        n = flat_img.size
    else:
        n = len(flat_img)
    side = int(np.sqrt(n))
    img = flat_img.reshape((side, side))
    return extract_beam_parameters(img, **kwargs)


def save_tensor_image_and_exit(tensor: torch.Tensor, path: str = "results/debug.png") -> None:
    x = tensor.detach().cpu()
    x0 = x[0]
    print(f"[DEBUG] input batch shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"[DEBUG] x0 stats: min={x0.min().item():.6g}, max={x0.max().item():.6g}, mean={x0.mean().item():.6g}")

    # Visualize without hiding scale errors; fall back to min-max if needed
    img = x0
    if img.dim() == 3 and img.size(0) in (1, 3):
        img_to_save = img.clone()
        m, M = img_to_save.min(), img_to_save.max()
        if (M > m):
            img_to_save = (img_to_save - m) / (M - m)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        vutils.save_image(img_to_save, path)
        print(f"[DEBUG] Saved sample to {path}. Program stopped.")
    else:
        print("[DEBUG] Not saving: expected CHW with C=1 or 3.")

    sys.exit(0)