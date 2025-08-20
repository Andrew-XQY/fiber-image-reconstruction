# def make_beam_param_metric(extract_fn):
#     def metric(pred, target):
#         import torch, numpy as np
#         if isinstance(pred, torch.Tensor):   pred = pred.detach().cpu()
#         if isinstance(target, torch.Tensor): target = target.detach().cpu()

#         B = pred.shape[0]
#         sums, counts = {}, {}

#         for i in range(B):
#             p = extract_fn(pred[i])    # dict with 4 params
#             t = extract_fn(target[i])  # dict with 4 params
#             for k in p.keys():
#                 err = abs(float(p[k]) - float(t[k]))  # MAE component
#                 sums[k] = sums.get(k, 0.0) + err
#                 counts[k] = counts.get(k, 0) + 1

#         out = {f"val_{k}_mae": sums[k] / counts[k] for k in sums}
#         out["_bs"] = B  # for sample-weighted epoch averaging
#         return out
#     return metric

def make_beam_param_metric(extract_fn):
    def metric(pred, target):
        import torch, math
        if isinstance(pred, torch.Tensor):   pred = pred.detach().cpu()
        if isinstance(target, torch.Tensor): target = target.detach().cpu()

        sums_abs, sums_sq, counts = {}, {}, {}
        B = len(pred)

        for i in range(B):
            p = extract_fn(pred[i]) or {}
            t = extract_fn(target[i]) or {}

            # Add "overall" from original values only
            p_vals = [float(v) for v in p.values()]
            t_vals = [float(v) for v in t.values()]
            if p_vals:
                p = {**p, "overall": math.fsum(p_vals) / len(p_vals)}
            if t_vals:
                t = {**t, "overall": math.fsum(t_vals) / len(t_vals)}

            for k in p.keys():
                diff = float(p[k]) - float(t[k])
                sums_abs[k] = sums_abs.get(k, 0.0) + abs(diff)      # MAE parts
                sums_sq[k]  = sums_sq.get(k, 0.0)  + diff * diff    # MSE/RMSE parts
                counts[k]   = counts.get(k, 0) + 1

        out = {}
        for k, n in counts.items():
            out[f"val_{k}_mae"]  = sums_abs[k] / n
            out[f"val_{k}_mse"]  = sums_sq[k] / n
            out[f"val_{k}_rmse"] = (sums_sq[k] / n) ** 0.5
        return out
    return metric
