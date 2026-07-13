"""5-minute smoke test before full HPC training.

Run from repo root (after installing the patched pattern_gen.py + utils.py):
    EXPERIMENT_CONFIG=CLEAR26_sgm_cam3 python smoke_test.py

Checks:
  1. patched files installed together (else: TypeError unexpected kwarg here)
  2. generated TARGET marginals land in the real-beam bands
  3. generation throughput with validator on (per-sample path)
  4. dumps smoke_samples.png -- eyeball inputs/targets before committing GPU hours
"""
import os, time
import numpy as np
from config_utils import load_config, detect_machine
from utils import build_datasets
from xflow import ConfigManager

name = os.getenv("EXPERIMENT_CONFIG", "CLEAR26_sgm_cam3")
config = ConfigManager(
    load_config(f"{name}.yaml", experiment_name=f"{name}-smoke",
                machine=detect_machine(), resolve=True)
).get()

t0 = time.time()
bundle = build_datasets(config)
print(f"[1] build_datasets ok (incl. basis cache): {time.time()-t0:.1f}s")

train = bundle["train_dataset"]
S = {"peak": [], "fp": [], "r": [], "sx": []}
t0, n = time.time(), 0
it = iter(train)
for _ in range(6):                      # ~6 batches x 32 = ~192 samples
    x, y = next(it)
    for img in y.detach().cpu().numpy():
        img = np.squeeze(img); s = float(img.sum()); n += 1
        if s <= 0:
            S["peak"].append(0.0); continue
        h, w = img.shape
        ys_, xs_ = np.mgrid[0:h, 0:w]
        cx = (xs_*img).sum()/s/w; cy = (ys_*img).sum()/s/h
        S["peak"].append(float(img.max()))
        S["fp"].append(float((img > 0.05).mean()))
        S["r"].append(float(np.hypot(cx-0.5, cy-0.5)))
        S["sx"].append(float(np.sqrt((((xs_/w)-cx)**2*img).sum()/s)))
dt = time.time() - t0
per_s = n/dt
print(f"[3] generated {n} samples in {dt:.1f}s ({per_s:.1f}/s) -> "
      f"one epoch (5000) ~ {5000/per_s/60:.1f} min of generation")

# bands: validator bands for peak/fp/r; sx = real 0.082 [0.067,0.106] + kernel margin
bands = {"peak": (0.25, 0.85), "fp": (0.05, 0.20), "r": (0.0, 0.42), "sx": (0.05, 0.13)}
print("[2] target marginals (expect ~100% in-band with validator enabled):")
worst = 1.0
for k, v in S.items():
    v = np.array(v); lo, hi = bands[k]
    ok = float(((v >= lo) & (v <= hi)).mean()); worst = min(worst, ok)
    print(f"    {k:4s}: mean {v.mean():.3f}  p5-p95 [{np.percentile(v,5):.3f},"
          f"{np.percentile(v,95):.3f}]  in-band {ok:.0%}  "
          f"{'PASS' if ok > 0.9 else 'CHECK BANDS / ACCEPTANCE'}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
x, y = next(iter(train))
fig, ax = plt.subplots(2, 8, figsize=(20, 5))
for i in range(min(8, len(x))):
    ax[0, i].imshow(np.squeeze(x[i].detach().cpu().numpy()), vmin=0, vmax=1)
    ax[1, i].imshow(np.squeeze(y[i].detach().cpu().numpy()), vmin=0, vmax=1)
    ax[0, i].axis("off"); ax[1, i].axis("off")
ax[0, 0].set_title("inputs (synth speckle)"); ax[1, 0].set_title("targets (SGM rendered)")
plt.tight_layout(); plt.savefig("smoke_samples.png", dpi=120)
print("[4] wrote smoke_samples.png -- compare targets against a real eval beam")
print("ALL PASS -- launch training" if worst > 0.9 else "FIX CHECK ITEMS FIRST")
