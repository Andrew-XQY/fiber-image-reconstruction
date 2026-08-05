# Calibration — dataset constants in a CLEAR26 config

**What this is.** Training data is synthetic (SGM patterns + a combinator over a
real basis). A few constants in the experiment yaml tie that synthetic stream to
the real camera. They are **measurements, not hyperparameters**.

**When to run it.** Any time a config is pointed at a new dataset, wavelength,
camera, exposure, or optics. Copying a config and changing only the paths gives
you wrong constants and a wasted HPC run.

---

## Run the four probes, in this order

Order matters — each probe measures in a space the previous one defined.

```bash
# repo root, needs the datasets + xflow
python -m probes.probe_normalizer   --config <CONFIG>
python -m probes.probe_sensor_noise --config <CONFIG>
python -m probes.probe_basis_scale  --config <CONFIG> --batches 8
python -m probes.probe_sgm_prior    --config <CONFIG> --synth
```

## Copy each result into the yaml

**1. `probe_normalizer`** → p99.5 of background-subtracted eval peaks, ×1.07, rounded up to 100.

```yaml
data.transforms.torch:  torch_remap_range.current_max     # both slots
combinator.transforms:  torch_sensor_noise.count_scale    # input slot only
```

**2. `probe_sensor_noise`** → fit of `var = a*signal + b` in counts.

```yaml
combinator.transforms:  torch_sensor_noise.a
combinator.transforms:  torch_sensor_noise.b
```

**3. `probe_basis_scale`** → pre-L2 synthetic/real energy ratio;
`new = current / ratio`.

```yaml
data.basis_transforms.torch:  torch_scale.scale_factor    # both slots
```

**4. `probe_sgm_prior`** → five beam marginals, real targets vs rendered targets.

```yaml
simulation.intensity_range
simulation.center_radius_range
simulation.std_1, simulation.std_2, simulation.aspect_range
simulation.orientation_range, simulation.shared_center, simulation.component_params  # when measured shape requires them
```

Write the measured numbers into the yaml as a dated comment block at the top,
the way `CLEAR26_690_cam3.yaml` and `CLEAR26_670_cam3_v3.yaml` do. The comments
are the provenance record — `probes/README.md` holds the reference values.

---

## Rules

- **Never tune these on eval metrics.** If a run looks bad, re-measure. Do not hand-fit.
- **`current_max` must exceed the eval max for that camera.** If the median
  normalized peak is above 1.0, every target is saturated and the run is dead
  before it starts. Check this first — cheapest catastrophic bug there is.
- **One fixed normalizer per camera**, shared by basis/train/val/test. Never
  per-image min-max — it destroys intensity linearity.
- **Calibrate basis scales before per-sample L2 and before clipping.** L2 removes
  amplitude information, while clipping understates energy. The probe creates an
  in-memory pre-L2 view, keeps earlier transforms such as sensor noise, and then
  replays the removed suffix to measure what `combinator.clip_output` would
  actually clip. Without L2, lower an excessive basis scale and rerun. With L2,
  downstream clipping is controlled by the post-L2 scale; changing the basis
  scale cannot remove it.
- **`intensity_range` matches the SGM *pattern* scale, not the rendered target.**
  The combinator has ~4.5× pattern→rendered gain.
- **`data.transforms` must contain no per-sample normalization.** It is also the
  basis-caching chain. Per-sample normalization belongs in `combinator.transforms`
  (train) and `data.eval_extra_transforms` (eval) only.

---

## Check before submitting to HPC

Per slot, in normalized units:

```
E_synth = coeff_sum_p50 * E_basis * torch_scale
E_synth / E_eval  ~=  1.0
```

This reproduces the documented `CLEAR26_690_cam3` operating point (1.03 input /
0.94 target) to about 1%, so it is a cheap check that a new calibration landed
in the right place. Then confirm:

- `probe_basis_scale` prints no material downstream-clipping warning
- `probe_sgm_prior --synth` marginals overlap the real ones (peak, sigma, centroid radius, footprint)

---

## If xflow or torch are not available

Probes 1, 2, and the real-target half of 4 are pure image statistics. Reproduce
them with numpy + PIL + sqlite3: run the config's SQL, load the 16-bit PNGs,
subtract the **per-pixel mean** of the background dataset's per-camera frames,
clip at 0.

**Validate the reimplementation against a documented reference before trusting
it on new data.** Known-good anchors:

- `CLEAR26_sgm_cam3` (405 nm): cam3 eval peak p99.5 = 1952
- `CLEAR26_690_cam3`: eval 570 pairs, cam3 max 2092.5, cam2 max 481.8
- `CLEAR26_690_cam3`: basis cam2 `E = sum/500` ≈ 377; coeff-map sum spread 2.1–4.8

Probe 3 and `probe_sgm_prior --synth` need the real pipeline. Do not guess them.
