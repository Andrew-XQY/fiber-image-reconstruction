# Calibration probes

These scripts measure camera and generated-image statistics for CLEAR26 configs.
Run them from the repository root after configuring `MACHINE` and the datasets
as described in the [README](../README.md).

```bash
python -m probes.probe_normalizer   --config CLEAR26_sgm_cam3
python -m probes.probe_sensor_noise --config CLEAR26_sgm_cam3
python -m probes.probe_basis_scale  --config CLEAR26_sgm_cam3 --batches 8
python -m probes.probe_sgm_prior    --config CLEAR26_sgm_cam3 --synth
```

Use this order: normalization, noise, basis scale, then beam priors. Later
measurements depend on the earlier settings. The scripts print measurements and
suggested values; update the experiment YAML after checking the results.

| Probe | Measurement | Config settings |
|---|---|---|
| `probe_normalizer` | Background-subtracted peak p99.5, with a 7% margin rounded up to 100 | `torch_remap_range.current_max`, input `torch_sensor_noise.count_scale` |
| `probe_sensor_noise` | Counts-space fit `variance = a × signal + b` | `torch_sensor_noise.a` and `.b` |
| `probe_basis_scale` | Synthetic/real energy ratio before per-sample L2; downstream clipping | `basis_transforms` scale factors; new scale = current scale / ratio |
| `probe_sgm_prior` | Target peak, widths, centroid radius and footprint | Gaussian priors and target-rendering settings |

`probe_sgm_prior --synth` uses the configured real/SGM mixture. Add
`--source sgm` or `--source real` to inspect one branch.

The probes default to the evaluation source. `probe_basis_scale` explicitly uses
`test_dataset` for its real-data comparison. For an independent test assessment,
fit these constants using a separate calibration pool. Recheck them when the
camera, wavelength, exposure, optics or dataset changes.

## Measurement spaces

- Background subtraction uses the per-pixel mean of the selected camera's
  background frames and clips negative residual counts to zero.
- A fixed normalizer per camera preserves relative intensity across images.
  Counts-space noise uses `count_scale` to convert normalized values to counts.
- Measure basis amplitude before per-sample L2 normalization. L2 cancels a
  uniform amplitude scale; downstream clipping in an L2 pipeline is controlled
  by the post-L2 scale. The basis-scale probe replays those later transforms to
  measure clipping separately.
- SGM coefficients and rendered targets have different amplitudes. The recorded
  405 nm combinator gain is about 4.5; check the rendered result rather than
  equating the coefficient peak with the target peak.
- Per-sample normalization belongs after basis combination, with the matching
  evaluation transform. Applying it independently to each cached basis image
  changes the superposition model.

## Recorded references

The original 405 nm settings use normalizers 1600 (cam1), 2100 (cam3), and 900
(cam2). Noise coefficients are `(a, b) = (0.106, 2.81)` for cam1 and
`(0.118, 4.82)` for cam3. Basis scales are 0.26, 0.28 and 0.26 respectively.
The configs retain the measured reference statistics for each wavelength.

The original probe scripts were not preserved. These scripts reconstruct the
procedures recorded in the config comments; those recorded values are not logs
of a new probe run. Keep the dataset identity, date and relevant measurements
with any recalibrated config.
