# Calibration probes

Measurement scripts that produce the hard-coded constants in
`conf/experiments/CLEAR26*.yaml`. Training synthesizes samples (SGM patterns +
combinator over a real basis); these probes anchor every synthetic quantity to
measured camera data. Re-run them whenever the camera, optics, exposure, or
dataset changes — never tune these constants on eval metrics.

| Probe | Measures | Feeds (yaml) | cam1 | cam3 | cam2 (target) |
|---|---|---|---|---|---|
| `probe_normalizer` | p99.5 of bg-subtracted eval peaks + margin | `torch_remap_range.current_max`, `torch_sensor_noise.count_scale` | 1600 | 2100 | 900 |
| `probe_sensor_noise` | photon-transfer fit `var = a*signal + b` (spatial noise-gap) | `torch_sensor_noise.{a, b}` | a=0.106 b=2.81 | a=0.118 b=4.82 | — |
| `probe_basis_scale` | synth/real energy ratio of the built pipeline | `basis_transforms torch_scale.scale_factor` | 0.26 | 0.28 (=1/3.58) | 0.26 |
| `probe_sgm_prior` | 5 beam marginals, real vs rendered targets | `simulation.*`, target kernel/scale | — | see yaml | — |

The table records the original 405 nm references. The calibrated 690 nm
values and their provenance are documented in `CLEAR26_690_cam3.yaml`.

## Run (repo root, needs datasets + xflow)

```bash
python -m probes.probe_normalizer   --config CLEAR26_sgm_cam3
python -m probes.probe_sensor_noise --config CLEAR26_sgm_cam3
python -m probes.probe_basis_scale  --config CLEAR26_sgm_cam3 --batches 8
python -m probes.probe_sgm_prior    --config CLEAR26_sgm_cam3 --synth
```

For the 690 nm configuration, replace the config name with
`CLEAR26_690_cam3`. `probe_sgm_prior --synth` checks the configured 70/30
production mixture; add `--source sgm` or `--source real` to isolate a branch.

Order matters: normalizer → sensor noise → basis scale → SGM prior
(later probes measure in spaces defined by earlier constants).

## Key facts for agents

- Image spaces: probes measure either bg-subtracted **counts** (0..4095) or
  **normalized** counts / `current_max` — the exact spaces training sees
  (`probes/common.py` reuses the trainer's registry transforms).
- Noise model (`torch_sensor_noise`): `var = a*signal + b` in counts;
  `count_scale` maps normalized → counts. Spatial fit is an upper bound;
  `param_jitter: 2.0` spans the uncertainty.
- Amplitude gotcha: combinator has ~4.5× pattern→rendered gain. SGM
  `intensity_range` must match the real *pattern* scale (~0.12 raw/4095), not
  rendered targets (~0.55).
- `sgm_validator` stays disabled — priors are matched by construction; verify
  via `probe_sgm_prior --synth`, epoch_1 images, or `smoke_test.py`.
- One fixed normalizer per camera; never per-image min-max (kills linearity).

Provenance: original probe scripts were not preserved; these are clean
reimplementations of the procedures documented in the yaml comments (which
remain the source of truth for the published values above).
