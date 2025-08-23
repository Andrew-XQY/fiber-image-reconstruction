import sys
import torch
import torchvision.utils as vutils
import math
from pathlib import Path

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