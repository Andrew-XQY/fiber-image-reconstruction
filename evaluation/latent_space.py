#!/usr/bin/env python3
"""Compare CLEAR26 image manifolds; edit CONFIG, then run this file."""

CONFIG = {
    "dataset_root": "/Users/andrewxu/Desktop/AI_Workspace/dataset/processed",
    "results_root": "/Users/andrewxu/Documents/GitHub/fiber-image-reconstruction/results",
    "output_name": "latent_space_mmf",  # YYYYMMDD_latent_space; reruns get a time suffix.
    "cameras": ["cam1", "cam2", "cam3"],
    "mmf_cohorts": {"mmf1": "cam3", "mmf2": "cam1"},  # Anchor camera for matched acquisitions.
    "camera_roles": {"cam1": "MMF2 output", "cam2": "shared reference", "cam3": "MMF1 output"},
    "n_samples": 100,               # Per dataset, per available camera; no replacement.
    "seed": 42,
    "exclude_patterns": ["*background*"],
    "exclude_merged": True,        # Metadata keys merge_source.* identify merged copies.
    "image_size": [256, 256],      # [height, width]; validated, never silently resized.
    "normalization": "camera_max", # camera_max: brightness + shape; l2: unit image norm.
    "pca_components": 50,          # Starting dimension; increase if variance is lost.
    "pca_min_variance": 0.90,      # None disables adaptive expansion.
    "pca_max_components": 512,     # Safety cap for adaptive PCA.
    "methods": ["umap", "tsne"],  # Same sampled images and PCA scores for both methods.
    "colors": {"beam_lightness": [0.38, 0.72], "other_hues": [0.22, 0.73]},
    "umap": {"n_neighbors": 30, "min_dist": 0.3, "n_epochs": 500},
    "tsne": {"perplexity": 30.0, "max_iter": 1500},
    "trustworthiness_neighbors": 15,
    "blas_threads": 4,
    "gif": {"frames": 120, "fps": 24, "dpi": 120, "figsize": [14, 8.5],
            "point_size": 12, "elevation": 24, "start_azimuth": 35},
    "save_interactive_html": True,
}

import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime


def read_database(db_path):
    """Open metadata read-only; never create or modify a source database."""
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def discover_datasets(config):
    root = Path(config["dataset_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    included, excluded = [], []
    for db in sorted(root.rglob("*.db")):
        dataset_id = db.parent.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(dataset_id.lower(), p.lower())
               for p in config["exclude_patterns"]):
            excluded.append({"dataset": dataset_id, "reason": "excluded folder pattern"})
            continue
        with closing(read_database(db)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
            required = {"image_id", "camera", "png_path", "sample_id", "run_id"}
            if not required <= columns:
                raise ValueError(f"{db}: images table missing {required - columns}")
            tables = {r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            info = dict(connection.execute("SELECT key, value FROM run_info")) if "run_info" in tables else {}
            merged = any(key.startswith("merge_source.") for key in info)
            if merged and config["exclude_merged"]:
                excluded.append({"dataset": dataset_id, "reason": "merged copy", "database": str(db)})
                continue
            counts = dict(connection.execute("SELECT camera, COUNT(*) FROM images GROUP BY camera"))
        name = db.parent.name.removesuffix("_256")
        match = re.match(r"(405|520|670|690)(?:_|$)", name)
        source = "real beam" if "realbeam" in name.lower() else "laser"
        # Wavelength is the folder label; for real beam it is NOT an electron wavelength.
        included.append({"dataset": dataset_id, "database": str(db), "counts": counts,
                         "day": db.parent.parent.name, "name": name,
                         "wavelength_label_nm": int(match[1]) if match else None,
                         "source": source, "merged": merged, "run_info": info})
    days = {d: i for i, d in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])}
    included.sort(key=lambda d: (days.get(d["day"], 9), d["name"], d["database"]))
    if not included:
        raise ValueError("No datasets remain after filtering.")
    ids = [d["dataset"] for d in included]
    if len(ids) != len(set(ids)):
        raise ValueError("Multiple databases in one dataset folder; choose one metadata DB per run.")
    return included, excluded


def sample_rows(dataset, camera, n_samples, seed, anchor_camera=None, return_count=False):
    """Stable RNG per dataset/camera; adding a run cannot change existing samples."""
    import numpy as np
    with closing(read_database(Path(dataset["database"]))) as connection:
        sql = "SELECT a.image_id, a.camera, a.png_path, a.sample_id, a.run_id FROM images a WHERE a.camera = ?"
        params = [camera]
        if anchor_camera is not None:
            # EXISTS avoids multiplying rows when a synchronized group has repeated frames.
            sql += (" AND EXISTS (SELECT 1 FROM images b WHERE b.camera = ?"
                    " AND b.run_id = a.run_id AND b.sample_id = a.sample_id)")
            params.append(anchor_camera)
        rows = [dict(r) for r in connection.execute(sql + " ORDER BY a.image_id", params)]
    if not rows:
        return ([], 0) if return_count else []
    paths = [row["png_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate image paths in {dataset['dataset']} / {camera}")
    key = f"{seed}|{dataset['dataset']}|{camera}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    indices = rng.choice(len(rows), size=min(n_samples, len(rows)), replace=False)
    sampled = [rows[int(i)] for i in sorted(indices)]
    return (sampled, len(rows)) if return_count else sampled


def load_camera(datasets, camera, config, anchor_camera=None):
    """Sample one camera from acquisitions containing the selected MMF anchor."""
    import numpy as np
    from PIL import Image
    selected = [(dataset, sample_rows(dataset, camera, config["n_samples"], config["seed"],
                                      anchor_camera, return_count=True)) for dataset in datasets]
    total = sum(len(rows) for _, (rows, _) in selected)
    height, width = config["image_size"]
    features = np.empty((total, height * width), dtype=np.float32)
    metadata, inventory = [], []
    index = 0
    for dataset, (rows, eligible) in selected:
        available = int(dataset["counts"].get(camera, 0))
        inventory.append({"dataset": dataset["dataset"], "camera": camera,
                          "anchor_camera": anchor_camera, "available": available,
                          "eligible_in_cohort": eligible, "sampled": len(rows),
                          "status": "missing camera" if not available else
                                    "no matched MMF acquisitions" if not rows else
                                    "fewer than requested" if len(rows) < config["n_samples"] else "ok"})
        if not rows:
            logging.warning("%s: %s / anchor %s: %s", dataset["dataset"], camera, anchor_camera, inventory[-1]["status"])
        for row in rows:
            path = Path(dataset["database"]).parent / row["png_path"]
            with Image.open(path) as image:
                values = np.asarray(image, dtype=np.float32)
            if values.shape != (height, width):
                raise ValueError(f"{path}: expected {(height, width)} grayscale pixels, got {values.shape}")
            if not np.isfinite(values).all() or values.min() < 0:
                raise ValueError(f"{path}: pixels must be finite and nonnegative")
            features[index] = values.ravel()
            metadata.append({"point_index": index, "dataset": dataset["dataset"],
                             "camera": camera, "image_id": row["image_id"],
                             "sample_id": row["sample_id"], "run_id": row["run_id"],
                             "anchor_camera": anchor_camera, "camera_role": config["camera_roles"][camera],
                             "image_path": str(path.resolve()), "source": dataset["source"],
                             "wavelength_label_nm": dataset["wavelength_label_nm"],
                             "pixel_max": float(values.max()), "pixel_mean": float(values.mean()),
                             "pixel_sum": float(values.sum(dtype=np.float64)),
                             "zero_image": bool(not values.any())})
            index += 1
    return features, metadata, inventory


def normalize(features, mode):
    import numpy as np
    if mode == "camera_max":
        scale = max(float(features.max()), 1e-12)
        features /= scale  # A single scalar over ALL runs in this camera.
        return {"normalization": mode, "camera_scale": scale}
    if mode == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features /= np.maximum(norms, 1e-12)  # Zero images remain zero.
        return {"normalization": mode, "camera_scale": None}
    raise ValueError(f"Unknown normalization: {mode}")


def fit_pca(features, config):
    """Fit shared PCA once per camera; reuse exactly these scores for both methods."""
    import numpy as np
    from sklearn.decomposition import PCA
    if features.shape[0] < 5:
        raise ValueError("At least five sampled images are needed per camera.")
    if not np.any(np.ptp(features, axis=0)):
        raise ValueError("All sampled images are identical; no manifold can be estimated.")
    limit = min(features.shape[0] - 1, features.shape[1])
    components = min(config["pca_components"], limit)
    pca = PCA(n_components=components, svd_solver="randomized", whiten=False,
              random_state=config["seed"])
    logging.info("PCA: %d images x %d pixels -> %d components", *features.shape, components)
    scores = pca.fit_transform(features)
    target = config["pca_min_variance"]
    maximum = min(config["pca_max_components"], limit)
    if target is not None and pca.explained_variance_ratio_.sum() < target and maximum > components:
        logging.info("PCA %d retains %.2f%%; expanding up to %d to target %.1f%%",
                     components, 100 * pca.explained_variance_ratio_.sum(), maximum, 100 * target)
        pca = PCA(n_components=maximum, svd_solver="randomized", whiten=False,
                  random_state=config["seed"])
        scores = pca.fit_transform(features)
        needed = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), target)) + 1
        components = min(maximum, max(components, needed))
        scores = np.ascontiguousarray(scores[:, :components])
    variance = pca.explained_variance_ratio_[:components]
    target_met = bool(target is None or float(variance.sum()) >= target)
    if not target_met:
        logging.warning("PCA cap retains %.2f%%, below target; see manifest before interpreting", 100 * variance.sum())
    details = {"pca_components": components,
               "pca_computed_components": int(pca.n_components_),
               "pca_variance_target_met": target_met,
               "pca_explained_variance_ratio": variance.tolist(),
               "pca_retained_variance": float(variance.sum())}
    logging.info("PCA retains %.2f%% variance in %d components", 100 * details["pca_retained_variance"], components)
    return scores, details


def embed_scores(scores, method, config):
    """Fit a nonlinear method to the camera's common PCA scores without labels."""
    import numpy as np
    from sklearn.manifold import TSNE, trustworthiness
    details = {"method": method}
    logging.info("Fitting %s in 3D to %d shared PCA samples", method, len(scores))
    if method == "umap":
        from umap import UMAP
        params = dict(config["umap"])
        params["n_neighbors"] = min(params["n_neighbors"], len(scores) - 1)
        reducer = UMAP(n_components=3, metric="euclidean", init="spectral",
                       random_state=config["seed"], transform_seed=config["seed"],
                       n_jobs=1, **params)
        coords = reducer.fit_transform(scores)
        details["effective_umap"] = params
    elif method == "tsne":
        params = dict(config["tsne"])
        params["perplexity"] = min(params["perplexity"], max(1.0, (len(scores) - 1) / 3))
        reducer = TSNE(n_components=3, metric="euclidean", init="pca", learning_rate="auto",
                       random_state=config["seed"], **params)
        coords = reducer.fit_transform(scores)
        details["effective_tsne"] = params
        details["tsne_kl_divergence"] = float(reducer.kl_divergence_)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")
    if coords.shape != (len(scores), 3) or not np.isfinite(coords).all():
        raise ValueError("Reducer returned invalid 3D coordinates.")
    neighbors = min(config["trustworthiness_neighbors"], (len(scores) - 1) // 2)
    details["trustworthiness_neighbors"] = neighbors
    details["trustworthiness_relative_to_pca"] = float(
        trustworthiness(scores, coords, n_neighbors=neighbors))
    return coords, details


def make_styles(datasets, config):
    """Reserve red exclusively for real beam; use identical circles for every run."""
    import colorsys
    from matplotlib.colors import to_hex
    beams = [d for d in datasets if d["source"] == "real beam"]
    others = [d for d in datasets if d["source"] != "real beam"]
    result = {}
    for group, is_beam in [(beams, True), (others, False)]:
        for i, dataset in enumerate(group):
            fraction = i / max(1, len(group) - 1)
            if is_beam:
                low, high = config["colors"]["beam_lightness"]
                rgb = colorsys.hls_to_rgb(0.0, low + (high - low) * fraction, 0.78)
            else:
                low, high = config["colors"]["other_hues"]
                rgb = colorsys.hls_to_rgb(low + (high - low) * fraction, 0.60, 0.68)
            result[dataset["dataset"]] = {
                "color": to_hex(rgb), "marker": "o",
                "label": dataset["day"][:3] + " / " + dataset["name"] + (" [merged]" if dataset["merged"] else ""),
            }
    return {d["dataset"]: result[d["dataset"]] for d in datasets}


def make_figure(coords, metadata, styles, camera, subtitle, axis_label, config):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    background, foreground = "#0d1422", "#e4eaf2"
    fig = plt.figure(figsize=config["gif"]["figsize"], dpi=config["gif"]["dpi"], facecolor=background)
    ax = fig.add_axes([0.015, 0.13, 0.62, 0.73], projection="3d", facecolor=background)
    groups = np.array([row["dataset"] for row in metadata])
    handles, labels = [], []
    for dataset_id, style in styles.items():
        mask = groups == dataset_id
        if not mask.any():
            continue
        ax.scatter(*coords[mask].T, s=config["gif"]["point_size"], c=style["color"],
                   marker="o", alpha=0.88, linewidths=0, depthshade=False)
        handles.append(Line2D([], [], linestyle="", marker=style["marker"],
                              color=style["color"], markersize=6))
        labels.append(f"{style['label']}  ({int(mask.sum())})")
    for j, axis in enumerate([ax.xaxis, ax.yaxis, ax.zaxis]):
        axis.set_pane_color((0.065, 0.09, 0.14, 1.0))
        axis.label.set_color(foreground)
        axis.line.set_color("#617185")
    for j, setlim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        low, high = float(coords[:, j].min()), float(coords[:, j].max())
        pad = max((high - low) * 0.08, 0.1)
        setlim(low - pad, high + pad)
    ax.set(xlabel=f"{axis_label} 1", ylabel=f"{axis_label} 2", zlabel=f"{axis_label} 3")
    ax.tick_params(colors="#98a8ba", labelsize=8, pad=1)
    ax.grid(False)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=config["gif"]["elevation"], azim=config["gif"]["start_azimuth"])
    fig.text(0.055, 0.943, f"CLEAR26  /  {camera}",
             fontsize=21, fontweight="bold", color=foreground)
    fig.text(0.055, 0.896, subtitle, fontsize=10.5, color="#aab9cc")
    legend = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.635, 0.50),
                        fontsize=8.0, labelcolor=foreground, frameon=False,
                        handletextpad=0.5, labelspacing=0.65, title="RUNS  /  sampled images")
    legend.get_title().set_color("#aab9cc")
    legend.get_title().set_fontsize(10)
    fig.text(0.055, 0.075, "Real beam: red shades   |   Other measurements: non-red   |   Same colors in every plot",
             fontsize=9, color="#aab9cc")
    fig.text(0.055, 0.043, "Independent fit per camera and acquisition set. Nonlinear cluster gaps and volumes are not calibrated dataset distances.",
             fontsize=9, color="#aab9cc")
    return fig, ax


def export_plots(coords, scores, metadata, styles, camera, method_name, metrics, out, config, view_label=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    method = method_name.upper()
    label = view_label or camera
    groups = len({row["dataset"] for row in metadata})
    subtitle = (f"{len(metadata):,} images / {groups} runs   |   Pixels -> PCA {metrics['pca_components']} -> {method} 3D"
                f"   |   {config['normalization']}   |   seed {config['seed']}")
    fig, ax = make_figure(coords, metadata, styles, label, subtitle, method, config)
    fig.savefig(out / f"{camera}_{method_name}.png", facecolor=fig.get_facecolor())
    frames = []
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, config["gif"]["frames"], endpoint=False)):
        ax.view_init(elev=config["gif"]["elevation"],
                     azim=config["gif"]["start_azimuth"] + np.degrees(angle))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(rgb).quantize(colors=256, dither=Image.Dither.NONE))
        if i % 30 == 0:
            logging.info("%s: rendering frame %d/%d", camera, i + 1, config["gif"]["frames"])
    target = out / f"{camera}_{method_name}_3d.gif"
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=round(1000 / config["gif"]["fps"] / 10) * 10,
                   loop=0, optimize=False, disposal=2)
    for frame in frames:
        frame.close()
    plt.close(fig)
    logging.info("Saved %s", target)
    if method_name == config["methods"][0]:
        fig, _ = make_figure(scores[:, :3], metadata, styles, label,
                             f"Linear PCA baseline   |   First 3 PCs: {100 * sum(metrics['pca_explained_variance_ratio'][:3]):.1f}% variance"
                             f"   |   {len(metadata):,} images / {groups} runs", "PC", config)
        fig.savefig(out / f"{camera}_pca_baseline.png", facecolor=fig.get_facecolor())
        plt.close(fig)
    if config["save_interactive_html"]:
        import plotly.graph_objects as go
        import html
        plot = go.Figure()
        for dataset_id, style in styles.items():
            indices = [i for i, row in enumerate(metadata) if row["dataset"] == dataset_id]
            if not indices:
                continue
            plot.add_trace(go.Scatter3d(
                x=coords[indices, 0], y=coords[indices, 1], z=coords[indices, 2], mode="markers",
                name=style["label"], marker={"size": 3, "color": style["color"], "symbol": "circle"},
                text=[html.escape(str(metadata[i]["image_id"])) for i in indices],
                hovertemplate="%{text}<extra>%{fullData.name}</extra>"))
        plot.update_layout(template="plotly_dark", title=f"{label} / {subtitle}",
                           scene={"xaxis_title": f"{method} 1", "yaxis_title": f"{method} 2",
                                  "zaxis_title": f"{method} 3"}, margin={"l": 0, "r": 0, "b": 0, "t": 70})
        plot.write_html(out / f"{camera}_{method_name}.html", include_plotlyjs=True)


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate_config(config):
    if config["n_samples"] < 1 or config["pca_components"] < 3:
        raise ValueError("n_samples must be positive and pca_components at least 3")
    if config["pca_max_components"] < config["pca_components"]:
        raise ValueError("pca_max_components must be at least pca_components")
    if config["pca_min_variance"] is not None and not 0 < config["pca_min_variance"] <= 1:
        raise ValueError("pca_min_variance must be None or in (0, 1]")
    if not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("methods must be nonempty and unique")
    if set(config["methods"]) - {"umap", "tsne"}:
        raise ValueError("methods may contain umap and tsne")
    low, high = config["colors"]["other_hues"]
    if not 0.15 <= low <= high <= 0.75:
        raise ValueError("Non-beam hues must stay in the non-red range [0.15, 0.75]")
    low, high = config["colors"]["beam_lightness"]
    if not 0.2 <= low <= high <= 0.85:
        raise ValueError("Beam red lightness must be in [0.2, 0.85]")
    if config["normalization"] not in {"camera_max", "l2"}:
        raise ValueError("normalization must be camera_max or l2")
    if not config["cameras"] or len(set(config["cameras"])) != len(config["cameras"]):
        raise ValueError("cameras must be nonempty and unique")
    if set(config["cameras"]) - {"cam1", "cam2", "cam3"}:
        raise ValueError("Only cam1, cam2, and cam3 are supported")
    if not config["mmf_cohorts"]:
        raise ValueError("At least one MMF acquisition cohort is required")
    for cohort, anchor in config["mmf_cohorts"].items():
        if not re.fullmatch(r"[a-z0-9_]+", cohort) or anchor not in {"cam1", "cam3"}:
            raise ValueError("Cohorts need a safe name and a fiber output anchor (cam1 or cam3)")
    if not re.fullmatch(r"[a-z0-9_]+", config["output_name"]):
        raise ValueError("output_name must use lowercase letters, digits, and underscores")
    if config["gif"]["frames"] < 2 or not 1 <= config["gif"]["fps"] <= 100:
        raise ValueError("GIF needs at least two frames and fps in [1, 100]")
    if config["trustworthiness_neighbors"] < 1:
        raise ValueError("trustworthiness_neighbors must be positive")


def main(config=None):
    config = CONFIG if config is None else config
    validate_config(config)
    datasets, excluded = discover_datasets(config)
    root = Path(config["results_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    out = root / f"{now:%Y%m%d}_{config['output_name']}"
    if out.exists():
        out = root / f"{now:%Y%m%d}_{config['output_name']}_{now:%H%M%S_%f}"
    out.mkdir()
    # Keep library caches and every generated artifact alongside these results.
    os.environ["MPLCONFIGDIR"] = str(out / ".cache" / "matplotlib")
    os.environ["NUMBA_CACHE_DIR"] = str(out / ".cache" / "numba")
    sys.dont_write_bytecode = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")], force=True)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from threadpoolctl import threadpool_limits
    packages = ["numpy", "scipy", "scikit-learn", "matplotlib", "pillow", "threadpoolctl"]
    if "umap" in config["methods"]:
        packages += ["umap-learn", "numba", "pynndescent"]
    if config["save_interactive_html"]:
        packages += ["plotly"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    styles = make_styles(datasets, config)
    manifest = {"status": "running", "started": now.isoformat(), "config": config,
                "python": sys.version, "executable": sys.executable, "versions": versions,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "datasets": datasets, "excluded": excluded, "styles": styles,
                "comparison_metrics": {}, "inventory": [],
                "cohort_definition": "Camera frames whose run_id and sample_id also contain the MMF anchor camera. Cohorts may overlap. Camera physical roles never change.",
                "expected_gifs": len(config["mmf_cohorts"]) * len(config["cameras"]) * len(config["methods"])}
    write_json(out / "manifest.json", manifest)
    logging.info("Output: %s; %d included datasets, %d excluded", out, len(datasets), len(excluded))
    try:
        with threadpool_limits(limits=config["blas_threads"]):
            for cohort, anchor_camera in config["mmf_cohorts"].items():
                for camera in config["cameras"]:
                    view_key = f"{cohort}_{camera}"
                    view_label = f"{cohort.upper()} acquisitions / {camera.upper()} ({config['camera_roles'][camera]})"
                    started = time.perf_counter()
                    features, metadata, inventory = load_camera(datasets, camera, config, anchor_camera)
                    for row in metadata + inventory:
                        row["cohort"] = cohort
                    manifest["inventory"].extend(inventory)
                    if len(features) < 5:
                        raise ValueError(f"{camera}: only {len(features)} usable samples")
                    scaling = normalize(features, config["normalization"])
                    scores, pca_metrics = fit_pca(features, config)
                    del features
                    pca_metrics.update(scaling)
                    pca_metrics.update({"cohort": cohort, "camera": camera, "anchor_camera": anchor_camera,
                                        "camera_role": config["camera_roles"][camera], "sample_count": len(metadata),
                                        "dataset_count": sum(r["sampled"] > 0 for r in inventory),
                                        "zero_images": sum(row["zero_image"] for row in metadata),
                                        "pca_elapsed_seconds": time.perf_counter() - started})
                    manifest["comparison_metrics"][view_key] = {}
                    for method in config["methods"]:
                        started = time.perf_counter()
                        coords, method_metrics = embed_scores(scores, method, config)
                        metrics = {**pca_metrics, **method_metrics}
                        rows = [{**row, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
                                 "color": styles[row["dataset"]]["color"]} for row, xyz in zip(metadata, coords)]
                        write_csv(out / f"{view_key}_{method}_samples.csv", rows)
                        np.savez_compressed(out / f"{view_key}_{method}_embedding.npz",
                                            coordinates=coords, pca_scores=scores,
                                            dataset=np.array([row["dataset"] for row in metadata]),
                                            image_id=np.array([str(row["image_id"]) for row in metadata]))
                        export_plots(coords, scores, rows, styles, view_key, method, metrics, out, config, view_label)
                        metrics["elapsed_seconds"] = time.perf_counter() - started
                        manifest["comparison_metrics"][view_key][method] = metrics
                        write_json(out / "manifest.json", manifest)
        if len(list(out.glob("*_3d.gif"))) != manifest["expected_gifs"]:
            raise RuntimeError("The expected GIF set is incomplete")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logging.exception("Run failed; manifest records the partial output")
        raise
    finally:
        write_csv(out / "inventory.csv", manifest["inventory"])
        write_json(out / "manifest.json", manifest)
    lines = ["# CLEAR26 dataset manifolds", "", f"Created: {now.isoformat()}", "",
             "One independent comparison per camera and MMF-associated acquisition set. UMAP and t-SNE reuse identical samples and PCA scores.",
             "MMF1 acquisitions require a cam3 frame; MMF2 acquisitions require a cam1 frame, joined on run_id and sample_id.",
             "These are overlapping acquisition cohorts, not six different physical camera/fiber combinations.",
             "Cam1 remains MMF2 output, cam3 remains MMF1 output, and cam2 remains the shared reference in both cohorts.",
             "All markers are circles. Real-beam runs use red shades; all other runs use non-red colors.",
             "Pixel intensities are preserved by default; brightness, exposure, offsets, and alignment can affect separation.",
             "No background subtraction, image registration, or learned encoder is applied.",
             "PCA scores are not whitened. Run labels and colors are used only for plotting.",
             "UMAP/t-SNE distances between islands and cluster volumes are not calibrated physical distances.",
             "Trustworthiness is measured against the retained PCA space, not the original full pixel space.",
             "Real-beam wavelength labels identify folder/optical context, not electron wavelengths.", "",
             "| Acquisition set / camera | Method | Runs | Images | PCA variance retained | Trustworthiness |", "|---|---|---:|---:|---:|---:|"]
    for camera, methods in manifest["comparison_metrics"].items():
        for method, metrics in methods.items():
            lines.append(f"| {camera} | {method} | {metrics['dataset_count']} | {metrics['sample_count']} | "
                         f"{metrics['pca_retained_variance']:.2%} | {metrics['trustworthiness_relative_to_pca']:.4f} |")
    lines += ["", "See manifest.json for config, versions, colors, excluded datasets, and missing cameras.",
              "The sample CSVs map every coordinate to its original image. NPZ files contain the retained PCA scores.",
              "The HTML plots are standalone and can hide individual runs through their legends.", ""]
    (out / "README.md").write_text("\n".join(lines))
    logging.info("Complete: %s", out)
    return out


if __name__ == "__main__":
    main()
