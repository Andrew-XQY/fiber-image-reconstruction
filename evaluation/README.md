# Dataset comparisons

Each experiment is a standalone Python script with its editable `CONFIG` dictionary at the top. Input databases are opened read-only. All generated artifacts and library caches go under the configured `results_root`; the repository ignores everything in `/results/` except `.gitkeep`.

## 1. Camera-wise image manifolds

Run from any working directory:

```bash
/opt/anaconda3/envs/torch/bin/python -B /Users/andrewxu/Documents/GitHub/fiber-image-reconstruction/evaluation/latent_space.py
```

The existing `torch` environment has all dependencies. For another Python 3.10+ environment, install `evaluation/requirements.txt` with that environment's Python, then run the script. `python3` on this Mac currently points to Apple's Python, which does not have these packages.

Defaults: 100 random eligible images per dataset/camera/acquisition set, seed 42, full 256×256 grayscale pixels, one shared maximum intensity scale per camera/acquisition set, PCA starting at 50 components without whitening (adaptively expanded to retain at least 90% variance, capped at 512), then both unsupervised 3D UMAP (`n_neighbors=30`, `min_dist=0.3`) and t-SNE (`perplexity=30`, `max_iter=1500`). PCA is fitted once per camera/acquisition set, jointly across its included runs; both methods reuse exactly the same sampled images and PCA scores. Cameras are never mixed. Cam1 and cam3 are the fiber cameras; cam2 is the screen/target camera in the reconstruction configuration.

Each database folder is a separate run/color, including separate days with the same run name. Background folders are excluded. The default also excludes the `405_realbeam_eval_256` merged copy, identified through `merge_source.*` metadata, because its original random/raster runs are present separately. Set `exclude_merged=False` to include it. Run labels retain the wavelength and acquisition descriptor. For real beam, the wavelength prefix refers to the folder/optical context, not an electron wavelength. Every point uses the same circle marker. Red is reserved exclusively for real-beam runs, with a distinct red shade for each measurement. Laser runs use green, blue, or violet hues. Each run keeps the same color across cameras and methods. Rotation uses a fixed elevation without viewpoint wobble.

The script samples each camera without replacement from acquisitions containing the selected MMF anchor camera. Matching uses both `run_id` and `sample_id`, with an `EXISTS` query so repeated anchor frames cannot multiply observations. It uses all images when fewer than the requested number exist, reports missing cameras, and records every sampled image ID/path. It requires the anchor camera and the plotted camera in the same database acquisition group; it does not require all three cameras. The two cohorts can overlap. Cohort eligibility and raw camera availability are recorded separately. Missing files, malformed metadata, repeated image paths, unexpected shapes, and nonfinite pixels fail explicitly. Zero images are retained and counted. Seeds are derived separately for each dataset/camera, so adding another run does not change existing samples.

### Configuration choices

- `n_samples`: number per run per camera/acquisition set, default 100.
- `mmf_cohorts`: `mmf1` is anchored to cam3, and `mmf2` to cam1. Each entry produces three camera views × two methods. The selection conditions are explicit and do not relabel the physical camera/fiber.
- `camera_roles`: labels identifying cam1 as MMF2 output, cam3 as MMF1 output, and cam2 as the shared reference.
- `methods`: `["umap", "tsne"]` by default; use either method alone if desired. A still linear PCA baseline is also saved.
- `colors`: brightness range for beam reds and a strictly non-red hue range for other measurements.
- `pca_components`, `pca_min_variance`, `pca_max_components`: start with 50 PCs; if they retain less than 90% of pixel variance, fit up to 512 PCs and retain the shortest prefix meeting the target (at least 50). Report if the cap still falls short. Set `pca_min_variance=None` to use a fixed dimension. This rule responds to retained variance, not run labels or visual separation.
- `normalization="camera_max"`: preserves relative brightness, offsets, and spatial structure across all runs of a camera. No per-run fitting/scaling, clipping, background subtraction, or resizing is applied.
- `normalization="l2"`: normalizes each image vector to unit norm for a brightness-invariant comparison; zero images remain zero. This can emphasize low-signal noise and retains spatial background patterns.
- `exclude_patterns`: case-insensitive patterns matched against the relative dataset folder.
- `save_interactive_html`: creates standalone offline Plotly plots with rotatable views, image IDs on hover, and clickable run legends.
- `gif`: frame count, frame rate, dimensions, and viewpoint. GIF timing is quantized to 10 ms by the format.

### Output

A new `results/YYYYMMDD_latent_space_mmf/` folder contains twelve rotating GIFs (two MMF-associated acquisition sets × three cameras × two methods), twelve still previews, six linear PCA baseline previews, optional interactive HTML files, method-specific sample/coordinate CSVs, compressed coordinates and PCA scores, `inventory.csv`, `manifest.json`, `run.log`, and a results README. Reruns on the same date receive a timestamp suffix instead of overwriting. `manifest.json` records the exact config, package versions, script hash, included/excluded runs, missing cameras, sampled zero images, retained PCA variance, and neighborhood trustworthiness relative to the retained PCA space. These are pixel-feature embeddings, not learned encoder latents. Coordinates from different acquisition sets, cameras, or methods cannot be compared directly.

The recorded physical mapping is cam1 = MMF2, cam3 = MMF1, and cam2 = the shared reference. The requested twelve views use two acquisition cohorts: MMF1 requires a cam3 frame in the same run/sample group, and MMF2 requires a cam1 frame. In each cohort, all three cameras are plotted separately. The cohorts are overlapping selections of measured acquisitions, not six physically different camera/fiber combinations. Titles preserve the actual physical camera role. Shared reference images and other measured images can legitimately occur in both cohorts; sample CSVs make the overlap auditable.

### Why this pipeline

The existing XFlow reducer is a reasonable wrapper around standard implementations. The notebook already pools datasets before fitting PCA and UMAP. This standalone version adds reproducible per-camera sampling, read-only metadata discovery, small-sample parameter caps, full precision image loading, provenance, camera inventory, an adaptive PCA variance check, and a linear control. PCA dimensionality is capped by both sample count and feature count, correcting a small-sample edge case in the notebook.

PCA preprocessing to a moderate dimension is documented for dense t-SNE input in [scikit-learn's TSNE reference](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). UMAP's neighborhood size and minimum distance control the balance of local structure and visual packing; see the [UMAP parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html). A fixed seed and single UMAP worker support reproducibility in the same environment; see [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).

Treat UMAP/t-SNE as exploratory views: nonlinear cluster gaps and volumes are not calibrated dataset distances or evidence of reconstruction performance. Brightness, exposure, offsets, alignment, and preprocessing can contribute to separation. UMAP does not reliably preserve density and can introduce apparent separations; see [UMAP's clustering cautions](https://umap-learn.readthedocs.io/en/latest/clustering.html). Check the PCA baseline and metadata before interpreting a visually separated run.

## 2. Synthetic vs real beam (domain gap)

```bash
/opt/anaconda3/envs/torch/bin/python -B /Users/andrewxu/Documents/GitHub/fiber-image-reconstruction/evaluation/synthetic_vs_real.py
```

`synthetic_vs_real.py` measures how far the training-style synthetic images are from the real-beam evaluation images, per camera (never across cameras). Everything is set in the `CONFIG` dict at the top; `CONFIG["experiments"]` is a list, one entry per dataset, processed in a loop.

**Part 1 – prepare.** For each camera it loads the laser basis and the real-beam evaluation pairs (same SQL filters as the training YAML), turns every real screen (cam2) image into a 34×34 coefficient map (area resize and a fixed ÷4095, nothing else), and lets XFlow's `SpatialNearestCombinator` build exactly one synthetic image per real image. Row *i* of `real`, `synth` and `index.csv` is one pair: synthetic *i* was built from the screen image of real sample *i*. The default is the pure baseline: raw 16-bit counts everywhere and no background subtraction anywhere (not on the basis, the real images or the prior), no remap, centroid kernel, scale, noise, jitter or clipping. The real images are the untouched PNGs. `steps` holds three hooks per camera (`load`: basis + real, `basis`, `synth`) in the YAML transform format; the training steps are there as commented lines to add back one by one.

**Part 2 – analyse.** Uses only `prepared.npz` + `index.csv` (set `reuse_prepared` to an earlier run folder to skip Part 1). Each analysis is one function registered in `ANALYSES`; metrics are registered in `PAIR_METRICS` and `SET_METRICS`.

| Analysis | Output | What it shows |
|---|---|---|
| `examples` | `examples.png` | prior, real, synthetic, difference for a few pairs (sanity check) |
| `manifold` | `manifold.png/.csv` | pixels → joint PCA → PCA / UMAP / t-SNE maps; lines join each synthetic to its real twin |
| `pairs` | `pairs.png/.csv` | per-pair Euclidean, relative L2, cosine, Pearson, energy ratio; grey reference = real vs another random real |
| `stats` | `stats.csv` | Fréchet, MMD, energy distance, Wasserstein, classifier accuracy, twin retrieval |

`normalize`: `none` compares raw intensities (brightness + shape), `l2` compares unit-norm images (shape only). Dataset-level distances are computed in the joint PCA space on random disjoint halves A, B of the pair indices: `synthetic_vs_real = d(synth[A], real[B])` next to `real_vs_real_floor = d(real[A], real[B])`, with `*_std` = spread over the splits: a value within `floor_std` of its floor is not distinguishable from real by that metric (metrics can disagree, e.g. the classifier picks up missing sensor noise that the distances ignore). `mmd_rbf` saturates near 1.26 once the two sets no longer overlap. `summary.csv` collects the headline numbers of every experiment/camera in one row each.

Checks done on 2026-09-18: with all steps off, the synthetic cam3 images equal the output of the production `CachedBasisPipeline` (same SQL, transforms and combinator) to float32 rounding, and with every training step enabled they match it to 9e-7. When adding `torch_subtract_background` back, add it for `cam2` under `load` as well: it moves the basis positions (about 0.75 px).

Other datasets that were test-run with the unchanged script: copy the 405 entry and change these fields (the 405 ROI cuts return zero pairs on 690).

| name | basis | real | background | `min_roi` basis / real | cam3 basis → pairs | note |
|---|---|---|---|---|---|---|
| 690 | `Thursday/690_reflection_large_256` | `Thursday/690_realbeam_256` | `Thursday/690_background_256` | 0.55 / 0.8 | 1764 → 570 | cam1 1773 → 570, cam2 2255 → 596 |
| 670 | `Wednesday/670_large_256` | `Wednesday/670_realbeam_256` | `Wednesday/670_background_256` | 0.55 / 0.8 | 884 → 2752 | largest set, about 4.5 min per camera |
| 520 | `Friday/520_stimulation_small_256` | `Friday/405_realbeam_eval_256` | `Friday/520_background_256` | 0.75 / 0.8 | 4093 → 437 | real set shared with 405, as in the YAML |

Basis folders without a camera: Wednesday `520_stimulation_large/small` have no cam3; Thursday `520_stimulation_small` and Tuesday `670_reflection_small` have no cam1.
