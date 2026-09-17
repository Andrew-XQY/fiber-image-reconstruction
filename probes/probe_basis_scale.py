"""Probe 3 — basis scale calibration (synthetic / real energy ratio).

The combinator assembles synthetic samples from scaled basis images. If the
synthetic stream carries R x the energy of real frames, the basis scale must be
divided by R:

  new_scale = current_scale / ratio      (per slot)

The ratio is measured before any per-sample L2 normalization. L2 normalization
cancels the basis amplitude, so post-L2 batches cannot calibrate a basis scale.
Transforms before the L2 boundary (notably synthetic sensor noise) remain
active, and real eval samples are stopped at the equivalent boundary.

For clipping diagnostics, the probe disables the combinator clip only in its
private config copy, reapplies the removed downstream transforms, and compares
that unclipped result with the production clip range. The training config is
never modified.

Feeds: data.basis_transforms -> torch_scale.scale_factor (input and target)

Run from repo root (needs the datasets + xflow; slow on first basis cache):
  python -m probes.probe_basis_scale --config CLEAR26_405_cam3_sgm --batches 8
"""
from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np

from probes.common import basis_scales, load_experiment


_L2_TRANSFORMS = frozenset({"l2_normalize", "torch_l2_normalize"})
_MATERIAL_CLIPPED_SAMPLE_FRACTION = 0.02
_MATERIAL_CLIPPED_ENERGY_FRACTION = 0.001


def _contains_transform(node, names):
    """Return whether a nested transform config contains one of ``names``."""
    if isinstance(node, list):
        return any(_contains_transform(item, names) for item in node)
    if isinstance(node, dict):
        if node.get("name") in names:
            return True
        return any(_contains_transform(value, names) for value in node.values())
    return False


def _split_before_l2(transforms):
    """Split a sequential transform list at the stage containing the first L2."""
    transforms = deepcopy(transforms or [])
    for index, stage in enumerate(transforms):
        if _contains_transform(stage, _L2_TRANSFORMS):
            return transforms[:index], transforms[index:], True
    return transforms, [], False


def calibration_view(cfg):
    """Return an in-memory pre-L2 config and the removed transform suffixes."""
    probe_cfg = deepcopy(cfg)

    combinator_cfg = probe_cfg.setdefault("combinator", {})
    transform_cfg = combinator_cfg.setdefault("transforms", {}).setdefault(
        "torch", []
    )
    train_prefix, train_suffix, train_has_l2 = _split_before_l2(transform_cfg)
    combinator_cfg["transforms"]["torch"] = train_prefix

    eval_cfg = probe_cfg.setdefault("data", {}).setdefault(
        "eval_extra_transforms", {}
    )
    eval_transforms = eval_cfg.setdefault("torch", [])
    eval_prefix, eval_suffix, eval_has_l2 = _split_before_l2(eval_transforms)
    eval_cfg["torch"] = eval_prefix

    clip = cfg.get("combinator", {}).get("clip_output", (0.0, 1.0))
    if not isinstance(clip, (list, tuple)) or len(clip) != 2:
        raise ValueError("combinator.clip_output must contain exactly [min, max].")
    clip_bounds = float(clip[0]), float(clip[1])
    if not np.isfinite(clip_bounds).all() or clip_bounds[0] > clip_bounds[1]:
        raise ValueError(
            "combinator.clip_output must contain finite bounds with min <= max."
        )

    # The probe needs the true pre-clip values. This changes only the deep copy;
    # the removed suffix is reapplied sample-by-sample for the clipping report.
    combinator_cfg["clip_output"] = [float("-inf"), float("inf")]
    return (
        probe_cfg,
        train_suffix,
        eval_suffix,
        clip_bounds,
        train_has_l2,
        eval_has_l2,
    )


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _new_stats():
    return {
        "sums": [],
        "peaks": [],
        "mins": [],
        "clipped_samples": 0,
        "clipped_pixels": 0,
        "pixels": 0,
        "clip_delta": 0.0,
        "abs_energy": 0.0,
    }


def _record(stats, image, clip_bounds=None):
    image = _as_numpy(image)
    stats["sums"].append(float(image.sum()))
    stats["peaks"].append(float(image.max()))
    stats["mins"].append(float(image.min()))
    if clip_bounds is None:
        return

    lo, hi = clip_bounds
    outside = (image < lo) | (image > hi)
    stats["clipped_samples"] += int(outside.any())
    stats["clipped_pixels"] += int(outside.sum())
    stats["pixels"] += int(image.size)
    clipped = np.clip(image, lo, hi)
    stats["clip_delta"] += float(np.abs(image - clipped).sum())
    stats["abs_energy"] += float(np.abs(image).sum())


def _finalize(stats):
    for key in ("sums", "peaks", "mins"):
        stats[key] = np.asarray(stats[key], dtype=np.float64)
    return stats


def stream_stats(dataset, n_batches, downstream_transforms, clip_bounds):
    """Collect pre-L2 and downstream-pre-clip statistics for both slots."""
    pre = [_new_stats(), _new_stats()]
    downstream = [_new_stats(), _new_stats()]
    sample_count = 0

    for batch_index, batch in enumerate(dataset):
        if batch_index >= n_batches:
            break
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Expected each batch to contain an input/target pair.")

        # Keep the framework types here. The v3 suffix deliberately starts
        # with torch_l2_normalize for the input slot and numpy l2_normalize for
        # the target slot; converting the whole batch to numpy would violate
        # that production type contract before the suffix is replayed.
        inputs, targets = batch[0], batch[1]
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Input and target batch sizes do not match.")

        for index in range(inputs.shape[0]):
            sample = (inputs[index], targets[index])
            _record(pre[0], sample[0])
            _record(pre[1], sample[1])

            transformed = sample
            for transform in downstream_transforms:
                transformed = transform(transformed)
            if not isinstance(transformed, (tuple, list)) or len(transformed) < 2:
                raise ValueError(
                    "Downstream transforms must preserve the input/target pair."
                )
            _record(downstream[0], transformed[0], clip_bounds)
            _record(downstream[1], transformed[1], clip_bounds)
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError("The requested dataset/batch range produced no samples.")
    return tuple(map(_finalize, pre)), tuple(map(_finalize, downstream))


def _clip_summary(stats):
    samples = len(stats["sums"])
    sample_fraction = stats["clipped_samples"] / samples
    pixel_fraction = stats["clipped_pixels"] / max(1, stats["pixels"])
    energy_fraction = stats["clip_delta"] / max(
        np.finfo(np.float64).eps, stats["abs_energy"]
    )
    return sample_fraction, pixel_fraction, energy_fraction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="CLEAR26_405_cam3_sgm")
    ap.add_argument("--batches", type=int, default=8, help="batches per stream")
    args = ap.parse_args()
    if args.batches < 1:
        ap.error("--batches must be at least 1")

    cfg = load_experiment(args.config)
    scale_in, scale_tgt = basis_scales(cfg)
    (
        probe_cfg,
        train_suffix_cfg,
        eval_suffix_cfg,
        clip_bounds,
        train_has_l2,
        eval_has_l2,
    ) = calibration_view(cfg)

    # Seed only the probe-side torch noise stream; the combinator's NumPy RNG is
    # already seeded by build_datasets. This makes repeated calibrations stable.
    import torch
    from xflow.data import build_transforms_from_config
    from utils import build_datasets  # heavy imports; keep local

    torch.manual_seed(int(cfg.get("seed", 42)))
    train_suffix = build_transforms_from_config(train_suffix_cfg)
    eval_suffix = build_transforms_from_config(eval_suffix_cfg)
    ds = build_datasets(probe_cfg)

    syn_pre, syn_downstream = stream_stats(
        ds["train_dataset"], args.batches, train_suffix, clip_bounds
    )
    real_pre, real_downstream = stream_stats(
        ds["test_dataset"], args.batches, eval_suffix, clip_bounds
    )

    if train_has_l2 != eval_has_l2:
        print(
            "[WARNING] L2 boundary exists on only one stream; synthetic and real "
            "pipeline topology should be checked."
        )
    boundary = "pre-L2" if train_has_l2 or eval_has_l2 else "pre-clip"
    print(
        f"config {args.config}   batches {args.batches}   "
        f"synthetic n={len(syn_pre[0]['sums'])}   real n={len(real_pre[0]['sums'])}"
    )
    print(
        f"energy boundary: {boundary}; pre-boundary sensor-noise transforms retained; "
        "config was copied, not modified"
    )
    print(
        f"production clip range: [{clip_bounds[0]:g}, {clip_bounds[1]:g}] "
        "(disabled only while measuring true values)\n"
    )

    for role, current, syn, real, syn_out, real_out in (
        ("input ", scale_in, syn_pre[0], real_pre[0], syn_downstream[0], real_downstream[0]),
        ("target", scale_tgt, syn_pre[1], real_pre[1], syn_downstream[1], real_downstream[1]),
    ):
        real_energy = float(real["sums"].mean())
        if real_energy <= 0:
            raise RuntimeError(f"Real {role.strip()} mean energy is not positive.")
        ratio = float(syn["sums"].mean() / real_energy)
        if ratio <= 0 or not np.isfinite(ratio):
            raise RuntimeError(f"Invalid synthetic/real ratio for {role.strip()}: {ratio}")
        suggested = current / ratio
        print(
            f"{role}: {boundary} energy synth {syn['sums'].mean():9.1f}  "
            f"real {real_energy:9.1f}  ratio {ratio:5.2f}"
        )
        print(
            f"        {boundary} peak synth mean/max {syn['peaks'].mean():.3f}/"
            f"{syn['peaks'].max():.3f}  real {real['peaks'].mean():.3f}/"
            f"{real['peaks'].max():.3f}"
        )
        print(
            f"        current torch_scale {current:.3f}  ->  suggested {suggested:.3f}"
            f"{'   (ok, keep)' if abs(ratio - 1) < 0.1 else ''}"
        )

        sample_frac, pixel_frac, energy_frac = _clip_summary(syn_out)
        print(
            f"        downstream pre-clip peak synth mean/max "
            f"{syn_out['peaks'].mean():.3f}/{syn_out['peaks'].max():.3f}  "
            f"real {real_out['peaks'].mean():.3f}/{real_out['peaks'].max():.3f}"
        )
        print(
            f"        production clipping would affect {100 * sample_frac:.2f}% samples, "
            f"{100 * pixel_frac:.5f}% pixels, {100 * energy_frac:.3f}% |energy|"
        )
        if (
            sample_frac > _MATERIAL_CLIPPED_SAMPLE_FRACTION
            or energy_frac > _MATERIAL_CLIPPED_ENERGY_FRACTION
        ):
            if train_has_l2:
                action = (
                    "Basis scale cannot remove post-L2 clipping; adjust the downstream "
                    "post-L2 scale, then rerun."
                )
            else:
                action = "Apply the lower basis scale and rerun until this warning clears."
            print(f"        [WARNING] Material synthetic clipping. {action}")


if __name__ == "__main__":
    main()
