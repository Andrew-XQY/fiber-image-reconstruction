#!/usr/bin/env python3
"""Synthetic (laser basis x real-beam shape prior) vs real-beam images: manifold + distances.

Part 1 prepares two paired datasets per camera; Part 2 analyses only the files Part 1 saved.
Edit CONFIG, then run this file.
"""

DATA = "/Users/andrewxu/Desktop/AI_Workspace/dataset/processed"

CONFIG = {
    "results_root": "/Users/andrewxu/Documents/GitHub/fiber-image-reconstruction/results",
    "output_name": "synthetic_vs_real",   # results/YYYYMMDD_<output_name>; reruns get a time suffix
    "seed": 42,
    "reuse_prepared": None,               # folder of an earlier full run: skip Part 1 (then the
                                          # "experiments" settings below are NOT applied)

    # ---- Part 1: one entry per dataset, processed in a loop ----
    "experiments": [
        {
            "name": "405",
            "basis": f"{DATA}/Friday/405_stimuation_large_256",   # laser spots (basis)
            "real": f"{DATA}/Friday/405_realbeam_eval_256",       # real beam (evaluation)
            "cameras": ["cam1", "cam2", "cam3"],   # compared one at a time, same camera only
            "prior_camera": "cam2",                # screen: shape prior + basis spot positions
            "min_roi": {"basis": 0.75, "real": 0.9},   # prior-camera roi_energy_fraction cut
            "jitter_alpha": 0.0,                   # sub-cell prior jitter, 0 = off (training: 1.0)
            # Real screen image -> coefficient map (numpy steps). Baseline: resize + fixed 1/4095 only.
            "prior_steps": [
                # {"name": "subtract_background", "params": {   # the training YAML also does this
                #     "background_source": f"{DATA}/Friday/405_background_256",
                #     "camera": "cam2", "clip_min": 0.0}},
                {"name": "resize", "params": {"size": [34, 34], "interpolation": "area"}},
                {"name": "remap_range", "params": {"current_min": 0.0, "current_max": 4095.0}},
            ],
            # Image steps per camera (torch steps, YAML format). All empty = pure base pipeline.
            # Add the training steps back one by one to study their effect.
            "steps": {
                "load": {     # every loaded image (basis + real)
                    # "cam3": [{"name": "torch_subtract_background", "params": {
                    #               "background_source": f"{DATA}/Friday/405_background_256",
                    #               "camera": "cam3", "clip_min": 0.0}},
                    #          {"name": "torch_remap_range", "params": {
                    #               "current_min": 0.0, "current_max": 2100.0}}],
                    # "cam2": [{"name": "torch_subtract_background", "params": {   # also moves the
                    #               "background_source": f"{DATA}/Friday/405_background_256",  # basis positions
                    #               "camera": "cam2", "clip_min": 0.0}},
                    #          {"name": "torch_remap_range", "params": {
                    #               "current_min": 0.0, "current_max": 900.0}}],
                },
                "basis": {    # basis images only
                    # "cam2": [{"name": "torch_render_centroid_kernel", "params": {"sigma": 4.0}},
                    #          {"name": "torch_scale", "params": {"scale_factor": 0.26}}],
                    # "cam3": [{"name": "torch_scale", "params": {"scale_factor": 0.28}}],
                },
                "synth": {    # combined synthetic images. count_scale and the 0..1 clip are in remapped
                              # units: enable together with torch_remap_range (on raw counts use
                              # count_scale 1.0 and no clip)
                    # "cam3": [{"name": "torch_sensor_noise", "params": {
                    #               "a": 0.118, "b": 4.82, "count_scale": 2100.0}},
                    #          {"name": "torch_clip_range", "params": {
                    #               "clip_min": 0.0, "clip_max": 1.0}}],
                },
            },
        },
    ],

    # ---- Part 2: analyses, run in this order (see ANALYSES at the bottom) ----
    "analyses": ["examples", "manifold", "pairs", "stats"],
    "normalize": ["none", "l2"],          # none = raw intensity; l2 = unit-norm image (shape only)
    "pca_components": 50,                 # joint PCA before UMAP / t-SNE / dataset distances
    "umap": {"n_neighbors": 15, "min_dist": 0.1},
    "tsne": {"perplexity": 30},
    "pair_lines": True,                   # manifold: thin line from each synthetic to its real twin
    "n_splits": 10,                       # stats: random half/half splits to average over
    "n_examples": 6,
}

import json
import sqlite3
import warnings
from contextlib import closing
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from xflow.data import build_transforms_from_config
from xflow.data.transform import load_image16
from xflow.extensions.physics.pipeline import (
    BasisAccessor, SpatialNearestCombinator, make_centroid_position_extractor)

REAL, SYNTH, REFERENCE = "#eb6834", "#2a78d6", "#8a8a86"   # one color per dataset, everywhere
EPS = 1e-12
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "savefig.dpi": 130, "figure.constrained_layout.use": True})
warnings.filterwarnings("ignore", message="n_jobs value")   # UMAP: a fixed seed forces one thread


# ======================== Part 1: prepare the two datasets ========================

def query_pairs(folder, camera, prior_camera, min_roi):
    """Sync-matched (camera, prior camera) image pairs; same filters as the training SQL."""
    sql = """SELECT a.run_id, a.sample_id, a.png_path, b.png_path
             FROM images a JOIN images b ON a.sample_id = b.sample_id AND a.run_id = b.run_id
             WHERE a.camera = ? AND b.camera = ?
               AND a.png_path IS NOT NULL AND b.png_path IS NOT NULL
               AND a.stat_max > 0 AND a.stat_sum > 0 AND b.roi_energy_fraction >= ?
               AND abs(a.sync_time_ns - b.sync_time_ns) <= 1000000
             ORDER BY a.run_id, a.sample_id"""
    uri = Path(folder, "dataset.db").resolve().as_uri() + "?mode=ro"   # read-only
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute(sql, (camera, prior_camera, min_roi)).fetchall()
    assert rows, f"no {camera} x {prior_camera} pairs with roi_energy_fraction >= {min_roi} in {folder}"
    return [{"run_id": run, "sample_id": sample, "image": f"{folder}/{image}",
             "prior": f"{folder}/{prior}"} for run, sample, image, prior in rows]


def run_steps(images, steps, tensor=True):
    """Apply xflow steps per image: torch steps see (1, H, W), numpy steps see (H, W)."""
    fns = build_transforms_from_config(steps)
    out = []
    for x in images:
        x = np.asarray(x, dtype=np.float32).reshape(np.shape(x)[-2:])
        x = torch.from_numpy(x)[None] if tensor else x
        for fn in fns:
            x = fn(x)
        out.append(np.asarray(x, dtype=np.float32).reshape(x.shape[-2:]))
    return np.stack(out)


def prepare(exp, camera, folder):
    """Part 1: real images + 1-to-1 paired synthetic images of one camera -> files in folder."""
    torch.manual_seed(CONFIG["seed"])
    prior_camera = exp["prior_camera"]
    steps = lambda stage, cam: exp["steps"][stage].get(cam, [])
    load = lambda rows, key, s, **kw: run_steps([load_image16(r[key]) for r in rows], s, **kw)

    # Basis: laser-spot images of this camera, positioned by their prior-camera centroid.
    rows = query_pairs(exp["basis"], camera, prior_camera, exp["min_roi"]["basis"])
    basis = load(rows, "image", steps("load", camera) + steps("basis", camera))
    spots = load(rows, "prior", steps("load", prior_camera) + steps("basis", prior_camera))
    centroid = make_centroid_position_extractor(method="first_moment")
    positions = np.array([centroid(spot, i) for i, spot in enumerate(spots)])
    del spots

    # Real beam: the compared image, and its screen image as the shape prior (coefficient map).
    index = query_pairs(exp["real"], camera, prior_camera, exp["min_roi"]["real"])
    real = load(index, "image", steps("load", camera))
    priors = load(index, "prior", exp["prior_steps"], tensor=False)

    # Synthetic i = basis combined by prior i, so row i of real / synth / index is one pair.
    alpha = exp["jitter_alpha"]
    combinator = SpatialNearestCombinator(
        iter(priors), basis_positions=positions, skip_zero=True, eps=1e-8,
        jitter_mode="global_cell" if alpha else "none", jitter_alpha=alpha,
        clip_output=(-np.inf, np.inf))   # no hidden clip; add torch_clip_range as a synth step
    accessor = BasisAccessor(list(basis), list(range(len(basis))))
    rng = np.random.default_rng(CONFIG["seed"])
    synth = []
    for row in index:
        synth.append(combinator(accessor, rng))
        record = combinator.last_record   # how many basis items / how much weight built this sample
        row.update(n_basis_used=len(record.indices), weight_sum=sum(record.coefficients))
    synth = run_steps(synth, steps("synth", camera))

    np.savez_compressed(folder / "prepared.npz", real=real, synth=synth, priors=priors)
    pd.DataFrame(index).to_csv(folder / "index.csv", index_label="pair")
    print(f"[{folder.name}] {len(basis)} basis images -> {len(real)} real / synthetic pairs")


def load_prepared(folder):
    """Part 2 input: only the files written by Part 1."""
    arrays = np.load(folder / "prepared.npz")
    return {**{key: arrays[key] for key in arrays.files},
            "index": pd.read_csv(folder / "index.csv"), "name": folder.name}


# ======================== Part 2: analyses ========================

def features(data, normalize):
    """Images -> [N, pixels] for (real, synth)."""
    flat = [data[key].reshape(len(data[key]), -1) for key in ("real", "synth")]
    if normalize == "l2":
        flat = [x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS) for x in flat]
    return flat


def joint_pca(real, synth):
    """One PCA fitted on both datasets together -> (real scores, synth scores, kept variance)."""
    components = min(CONFIG["pca_components"], 2 * len(real) - 1)
    pca = PCA(n_components=components, random_state=CONFIG["seed"])
    scores = pca.fit_transform(np.vstack([real, synth]))
    return scores[:len(real)], scores[len(real):], float(pca.explained_variance_ratio_.sum())


def analyze_examples(data, out):
    """Sanity check: prior | real | synthetic | difference for a few evenly spaced pairs."""
    picks = np.linspace(0, len(data["real"]) - 1, CONFIG["n_examples"]).astype(int)
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.2 * len(picks)), squeeze=False)
    for row, i in zip(axes, picks):
        diff = data["synth"][i] - data["real"][i]
        panels = {"prior": data["priors"][i], "real": data["real"][i],
                  "synthetic": data["synth"][i], "synthetic - real": diff}
        for ax, (title, image) in zip(row, panels.items()):
            limit = np.abs(diff).max()
            style = {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit} if image is diff else {"cmap": "gray"}
            fig.colorbar(ax.imshow(image, **style), ax=ax, fraction=0.046)
            ax.set_title(f"pair {i}: {title} (max {image.max():.3g})", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{data['name']}: each panel has its own intensity scale")
    fig.savefig(out / "examples.png")
    plt.close(fig)
    return {}


def analyze_manifold(data, out):
    """Pixels -> joint PCA -> 2D PCA / UMAP / t-SNE maps; one row of maps per normalization."""
    norms, n = CONFIG["normalize"], len(data["real"])
    fig, axes = plt.subplots(len(norms), 3, figsize=(17, 5.5 * len(norms)), squeeze=False)
    coords = []
    for row, norm in zip(axes, norms):
        z = np.vstack(joint_pca(*features(data, norm))[:2])   # real scores, then synthetic
        umap = UMAP(n_neighbors=min(CONFIG["umap"]["n_neighbors"], len(z) - 1),
                    min_dist=CONFIG["umap"]["min_dist"], random_state=CONFIG["seed"])
        tsne = TSNE(perplexity=min(CONFIG["tsne"]["perplexity"], (len(z) - 1) / 3),
                    init="pca", random_state=CONFIG["seed"])
        maps = {"pca": z[:, :2], "umap": umap.fit_transform(z), "tsne": tsne.fit_transform(z)}
        for ax, (method, xy) in zip(row, maps.items()):
            if CONFIG["pair_lines"]:
                ax.add_collection(LineCollection(np.stack([xy[:n], xy[n:]], axis=1),
                                                 colors=REFERENCE, linewidths=0.4, alpha=0.5))
            for part, color, label in [(xy[:n], REAL, "real"), (xy[n:], SYNTH, "synthetic")]:
                ax.scatter(*part.T, s=16, c=color, edgecolors="white", linewidths=0.4, label=label)
            ax.set_title(f"{method.upper()}  |  normalize = {norm}")
            ax.set(xlabel=f"{method} 1", ylabel=f"{method} 2")
            coords.append(pd.DataFrame({"pair": np.tile(np.arange(n), 2), "normalize": norm,
                                        "dataset": np.repeat(["real", "synthetic"], n),
                                        "method": method, "x": xy[:, 0], "y": xy[:, 1]}))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", ncol=2,
               frameon=False, markerscale=2)
    fig.suptitle(f"{data['name']}: {n} real vs {n} synthetic images, lines join twins")
    fig.savefig(out / "manifold.png")
    plt.close(fig)
    pd.concat(coords).to_csv(out / "manifold.csv", index=False)
    return {}


def _cosine_distance(a, b):
    return 1 - (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), EPS)


PAIR_METRICS = {   # name -> fn(a [N, D], b [N, D]) -> [N] row-wise values; add new ones here
    "euclidean": lambda a, b: np.linalg.norm(b - a, axis=1),
    "relative_l2": lambda a, b: np.linalg.norm(b - a, axis=1) / np.maximum(np.linalg.norm(a, axis=1), EPS),
    "cosine_distance": _cosine_distance,
    "pearson_distance": lambda a, b: _cosine_distance(a - a.mean(1, keepdims=True),
                                                      b - b.mean(1, keepdims=True)),
    "energy_ratio": lambda a, b: b.sum(1) / np.maximum(a.sum(1), EPS),
}


def analyze_pairs(data, out):
    """Every synthetic image vs its own real twin; reference = real vs another random real."""
    real, synth = features(data, "none")
    order = np.random.default_rng(CONFIG["seed"]).permutation(len(real))
    other = np.empty_like(order)
    other[order] = np.roll(order, 1)   # a random partner that is never the sample itself
    table = data["index"].copy()
    fig, axes = plt.subplots(len(PAIR_METRICS), 2, figsize=(15, 3 * len(PAIR_METRICS)),
                             squeeze=False, width_ratios=[2, 1])
    result = {}
    for (name, fn), (ax_pairs, ax_hist) in zip(PAIR_METRICS.items(), axes):
        table[name], table[f"{name}_reference"] = fn(real, synth), fn(real, real[other])
        values = table[[name, f"{name}_reference"]].values
        bins = np.geomspace(values[values > 0].min(), values.max(), 40)   # log axes: many decades
        for column, color, label in [(f"{name}_reference", REFERENCE, "real vs another real (reference)"),
                                     (name, SYNTH, "synthetic vs its real twin")]:
            ax_pairs.plot(table[column].values, "o", ms=3, color=color, label=label)
            ax_hist.hist(table[column], bins=bins, histtype="step", linewidth=2, color=color)
            ax_hist.axvline(table[column].median(), color=color, linestyle=":")
            result[column] = float(table[column].median())
        ax_pairs.set(xlabel="pair index", ylabel=name, yscale="log")
        ax_hist.set(xlabel=f"{name} (dotted = median)", ylabel="pairs", xscale="log")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", ncol=2,
               frameon=False, markerscale=2)
    fig.suptitle(f"{data['name']}: per-pair distances (normalize = none)")
    fig.savefig(out / "pairs.png")
    plt.close(fig)
    table.to_csv(out / "pairs.csv", index=False)
    return result   # medians


def frechet(a, b):
    """Frechet (FID-style) distance between Gaussian fits of the two sets."""
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cross = np.sqrt(np.clip(np.linalg.eigvals(cov_a @ cov_b).real, 0, None)).sum()
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * cross)


def mmd_rbf(a, b):
    """Unbiased squared MMD, RBF kernel, pooled median bandwidth.

    Saturates near 2 - 2/e = 1.26 once the two sets no longer overlap.
    """
    d = cdist(np.vstack([a, b]), np.vstack([a, b]), "sqeuclidean")
    k, n, m = np.exp(-d / np.median(d[d > 0])), len(a), len(b)
    return float((k[:n, :n].sum() - n) / (n * (n - 1)) + (k[n:, n:].sum() - m) / (m * (m - 1))
                 - 2 * k[:n, n:].mean())


def energy_distance(a, b):
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def wasserstein(a, b):
    """Earth mover's distance: mean distance under the optimal 1-to-1 matching."""
    d = cdist(a, b)
    return float(d[linear_sum_assignment(d)].mean())


def classifier_accuracy(a, b):
    """Two-sample test by classifier: 0.5 = indistinguishable, 1.0 = trivially separable."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    labels = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return float(cross_val_score(model, np.vstack([a, b]), labels, cv=5).mean())


SET_METRICS = {   # name -> fn(A [n, k], B [n, k]) -> float; add new ones here
    "frechet": frechet, "mmd_rbf": mmd_rbf, "energy_distance": energy_distance,
    "wasserstein": wasserstein, "classifier_accuracy": classifier_accuracy,
}


def analyze_stats(data, out):
    """Dataset-level distances in joint-PCA space, each next to a real-vs-real floor.

    Pair indices are split into random disjoint halves A, B:
    synthetic_vs_real = d(synth[A], real[B]),  real_vs_real_floor = d(real[A], real[B]).
    Same sizes and different beam shots on both sides. *_std = spread over the splits: a value
    within floor_std of its floor is not distinguishable by that metric (metrics can disagree).
    """
    rng, n = np.random.default_rng(CONFIG["seed"]), len(data["real"])
    splits = [np.split(rng.permutation(n)[:n // 2 * 2], 2) for _ in range(CONFIG["n_splits"])]
    rows = []
    for norm in CONFIG["normalize"]:   # the same splits for every normalization
        real, synth, variance = joint_pca(*features(data, norm))
        for name, fn in SET_METRICS.items():
            v = np.array([[fn(synth[a], real[b]), fn(real[a], real[b])] for a, b in splits])
            rows.append({"normalize": norm, "metric": name,
                         "synthetic_vs_real": v[:, 0].mean(), "real_vs_real_floor": v[:, 1].mean(),
                         "synthetic_std": v[:, 0].std(), "floor_std": v[:, 1].std()})
        # Twin retrieval on all pairs: rank 0 = the real image nearest to synthetic i is its twin.
        d = cdist(synth, real)
        rank = (d < np.diag(d)[:, None]).sum(1)
        rows += [{"normalize": norm, "metric": "twin_top1_accuracy", "synthetic_vs_real": (rank == 0).mean()},
                 {"normalize": norm, "metric": "twin_median_rank", "synthetic_vs_real": np.median(rank)},
                 {"normalize": norm, "metric": "pca_kept_variance", "synthetic_vs_real": variance}]
    table = pd.DataFrame(rows)
    table.to_csv(out / "stats.csv", index=False)
    print(table.to_string(index=False, float_format="%.4g"))
    headline = table.melt(["normalize", "metric"], ["synthetic_vs_real", "real_vs_real_floor"]).dropna()
    return {f"{r.normalize}.{r.metric}.{r.variable}": r.value for r in headline.itertuples()}


ANALYSES = {   # name -> fn(data, out folder) -> dict of headline numbers; add new ones here
    "examples": analyze_examples, "manifold": analyze_manifold,
    "pairs": analyze_pairs, "stats": analyze_stats,
}


# ======================== Main loop ========================

def main():
    now = datetime.now()
    out = Path(CONFIG["results_root"]) / f"{now:%Y%m%d}_{CONFIG['output_name']}"
    if out.exists():
        out = out.with_name(f"{out.name}_{now:%H%M%S}")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2))

    summary = []
    for exp in CONFIG["experiments"]:
        for camera in exp["cameras"]:
            folder = out / f"{exp['name']}_{camera}"
            folder.mkdir()
            if CONFIG["reuse_prepared"]:
                prepared = Path(CONFIG["reuse_prepared"]) / folder.name
                print(f"[{folder.name}] reusing {prepared}")
            else:
                prepare(exp, camera, folder)
                prepared = folder
            data = load_prepared(prepared)
            results = {name: ANALYSES[name](data, folder) for name in CONFIG["analyses"]}
            summary.append({"experiment": exp["name"], "camera": camera, "pairs": len(data["real"]),
                            **pd.json_normalize(results, sep=".").iloc[0]})
            pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
