import sys
import torch
import torchvision.utils as vutils
import math
import numpy as np
from pathlib import Path
from copy import deepcopy
from xflow import SqlProvider, PyTorchPipeline, instantiate, show_model_info
from xflow.data import build_transforms_from_config
from xflow.utils import resolve_resource_dir

from xflow.extensions.physics.pipeline import (
    CachedBasisPipeline,
    RetryPolicy,
    SpatialNearestCombinator,
    make_centroid_position_extractor,
)
from xflow.extensions.physics.nnls_pipeline import NNLSCoefficientMapCombinator
from xflow.extensions.physics import pattern_gen

SAMPLE_FLATTENED = ['SHL_DNN']
REGRESSION = ['ERN'] # Encoder-regressor
GAN = ['Pix2pix']


# ========================================
# Shared Data Preparation Helpers
# ========================================
def resolve_dataset_dir(config: dict, path_or_key: str) -> Path:
    p = Path(config["paths"].get(path_or_key, path_or_key)).expanduser().resolve()
    if p.is_dir():
        return p
    s = str(p).lower()
    if s.endswith(".tar.gz"):
        p = p.with_name(p.name[:-7])
    elif s.endswith(".tgz"):
        p = p.with_name(p.name[:-4])
    elif p.suffix.lower() in {".tar", ".zip"}:
        p = p.with_name(p.stem)
    resolve_resource_dir(str(p))  # folder exists OR extract same-name archive
    return p


# Rejection-sampling quality gate for generated (input, target) pairs, applied
# to the RENDERED target. Bands come from independent/measured beam statistics,
# NOT tuned on eval metrics. Config section: sgm_validator (see yaml).
def make_beam_target_validator(cfg: dict):
    peak_lo, peak_hi = cfg.get("peak_range", (0.0, 1.0))
    fp_lo, fp_hi = cfg.get("footprint_range", (0.0, 1.0))
    fp_thr = float(cfg.get("footprint_threshold", 0.05))
    r_max = float(cfg.get("centroid_radius_max", 1.0))

    def validator(sample, record):
        tgt = np.asarray(sample[1] if isinstance(sample, (tuple, list)) else sample)
        tgt = np.squeeze(tgt)
        peak = float(tgt.max())
        if not (peak_lo <= peak <= peak_hi):
            return (False, f"peak {peak:.3f} outside [{peak_lo}, {peak_hi}]")
        fp = float((tgt > fp_thr).mean())
        if not (fp_lo <= fp <= fp_hi):
            return (False, f"footprint {fp:.3f} outside [{fp_lo}, {fp_hi}]")
        s = float(tgt.sum())
        if s > 0 and r_max < 1.0:
            h, w = tgt.shape[-2:]
            ys, xs = np.mgrid[0:h, 0:w]
            cx = float((xs * tgt).sum()) / s / w
            cy = float((ys * tgt).sum()) / s / h
            r = math.hypot(cx - 0.5, cy - 0.5)
            if r > r_max:
                return (False, f"centroid r {r:.3f} > {r_max}")
        return True

    return validator


def build_datasets(config: dict) -> dict:
    dataset_sources = config["data"].get("dataset_sources", ["processed_chromox_cropped"])
    dataset_dirs = [resolve_dataset_dir(config, src) for src in dataset_sources]
    db_rel = config["dataset_structure"]["db"].lstrip("/\\")
    db_paths = [d / db_rel for d in dataset_dirs]
    provider_output_column = config["data"].get("provider_output_column", "image_path")

    def with_parent_dir(transforms_config, parent_dir):
        transforms_config = deepcopy(transforms_config)

        def apply(items):
            for t in items:
                if not isinstance(t, dict):
                    continue  # null = pass-through slot in multi_transform
                if t.get("name") == "add_parent_dir":
                    t.setdefault("params", {})["parent_dir"] = str(parent_dir)
                child_transforms = t.get("params", {}).get("transforms")
                if child_transforms:
                    apply(child_transforms)

        apply(transforms_config)
        return transforms_config

    def build_sgm_stream(cfg):
        simulation_cfg = cfg["simulation"]
        canvas = pattern_gen.DynamicPatterns(*simulation_cfg["canvas_size"])
        canvas.set_max_pixel_value(simulation_cfg.get("max_pixel_value", 255.0))
        canvas.set_postprocess_fns(
            build_transforms_from_config(simulation_cfg["process_functions"])
        )
        canvas._distributions = [
            pattern_gen.StaticGaussianDistribution(canvas)
            for _ in range(simulation_cfg["total_Guassian_num"])
        ]
        canvas.set_threshold(simulation_cfg["minimum_pixel_threshold"])

        # Optional config-driven priors; None -> legacy hardcoded behavior.
        # Requires xflow >= the pattern_gen patch adding these kwargs.
        def _as_range(key):
            v = simulation_cfg.get(key)
            return tuple(v) if v is not None else None

        return canvas.pattern_stream(
            std_1=simulation_cfg["std_1"],
            std_2=simulation_cfg["std_2"],
            max_intensity=simulation_cfg["max_intensity"],
            fade_rate=simulation_cfg["fade_rate"],
            area_boost_scale=simulation_cfg.get("area_boost_scale", 0.25),
            max_peak_intensity=simulation_cfg.get("max_peak_intensity"),
            distribution=simulation_cfg["distribution"],
            intensity_range=_as_range("intensity_range"),
            center_radius_range=_as_range("center_radius_range"),
            aspect_range=_as_range("aspect_range"),
        )

    # ====================================
    # Case 2 :single dataset (source).
    # ====================================
    # Disabled for CLEAR26. Case 3 / 3-b below is the active application path.

    pipeline_case = config["data"].get("pipeline_case", "spatial_nearest_combinator")
    supported_pipeline_cases = {
        "spatial_nearest_combinator",
        "nnls_coefficient_map",
    }
    if pipeline_case not in supported_pipeline_cases:
        raise ValueError(
            f"Unsupported dataset pipeline {pipeline_case!r}. Expected one of "
            f"{sorted(supported_pipeline_cases)}."
        )

    # ====================================
    # Case 3: existing nearest-neighbor path; Case 3-b: NNLS coefficient-map path.
    # Both use the same mixed (real images + SGM) pattern stream and data conventions.
    # ====================================
    train_sql_key = config["data"]["train_sql_key"]
    eval_sql_key = config["data"]["eval_sql_key"]
    pattern_sql_key = config["data"].get("pattern_sql_key", train_sql_key)
    train_db_path = db_paths[0]
    eval_db_path = db_paths[1] if len(db_paths) > 1 else train_db_path
    pattern_db_path = eval_db_path

    train_provider = SqlProvider(
        sources={"connection": train_db_path, "sql": config["sql"][train_sql_key]},
        output_config={"list": provider_output_column},
    )
    eval_provider = SqlProvider(
        sources={"connection": eval_db_path, "sql": config["sql"][eval_sql_key]},
        output_config={"list": provider_output_column},
    )
    val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"])
    pattern_source_provider = SqlProvider(
        sources={"connection": pattern_db_path, "sql": config["sql"][pattern_sql_key]},
        output_config={"list": provider_output_column},
    )

    train_transforms = build_transforms_from_config(
        with_parent_dir(config["data"]["transforms"]["torch"], train_db_path.parent)
    )
    basis_transforms_config = config["data"].get("basis_transforms", {}).get(
        "torch", []
    )
    if basis_transforms_config:
        train_transforms.extend(
            build_transforms_from_config(
                with_parent_dir(basis_transforms_config, train_db_path.parent)
            )
        )
    eval_transforms = build_transforms_from_config(
        with_parent_dir(config["data"]["transforms"]["torch"], eval_db_path.parent)
    )
    pattern_transforms = build_transforms_from_config(
        with_parent_dir(config["image_generator"]["transforms"], pattern_db_path.parent)
    )

    real_stream = pattern_gen.image_pattern_stream(
        source=pattern_source_provider,
        transforms=pattern_transforms,
        shuffle=False,
        seed=config["seed"],
        cache_transformed=config["image_generator"].get("cache_transformed", False),
    )
    sgm_stream = build_sgm_stream(config)
    mixed_stream = pattern_gen.weighted_stream(
        sources=[real_stream, sgm_stream],
        probabilities=[
            config["weighted_stream"]["real_weight"],
            config["weighted_stream"]["sgm_weight"],
        ],
        seed=config["seed"],
    )
    # Optional per-sample global intensity jitter (input AND target scaled
    # together) -> intensity equivariance. Scalar or [lo, hi]; 1.0 = legacy.
    intensity_scale = config["combinator"].get("intensity_scale", 1.0)
    if isinstance(intensity_scale, (list, tuple)):
        intensity_scale = tuple(float(v) for v in intensity_scale)
    else:
        intensity_scale = float(intensity_scale)

    combinator_transforms = build_transforms_from_config(
        config["combinator"]["transforms"]["torch"]
    )
    clip_output = tuple(config["combinator"].get("clip_output", (0.0, 1.0)))
    if pipeline_case == "nnls_coefficient_map":
        # Case 3-b: fit non-negative basis coefficients to each target map.
        nnls_cfg = config.get("nnls") or {}
        combinator = NNLSCoefficientMapCombinator(
            pattern_provider=mixed_stream,
            solve_component=int(nnls_cfg.get("solve_component", 1)),
            regularization=float(nnls_cfg.get("lambda", 0.0)),
            smoothness_neighbors=int(nnls_cfg.get("smoothness_neighbors", 4)),
            coefficient_threshold=float(
                nnls_cfg.get("coefficient_threshold", 1e-8)
            ),
            max_coefficient=nnls_cfg.get("max_coefficient"),
            maxiter=nnls_cfg.get("maxiter"),
            intensity_scale=intensity_scale,
            clip_output=clip_output,
            transforms=combinator_transforms,
        )
    else:
        # Existing Case 3 behavior remains the default and is unchanged.
        combinator = SpatialNearestCombinator(
            pattern_provider=mixed_stream,
            skip_zero=True,
            eps=1e-8,
            jitter_mode=config["combinator"].get("jitter_mode", "global_cell"),
            jitter_alpha=config["combinator"].get("jitter_alpha", 1.0),
            intensity_scale=intensity_scale,
            clip_output=clip_output,
            transforms=combinator_transforms,
        )

    # Optional rejection sampling on generated pairs (see make_beam_target_validator).
    # NOTE: a validator disables the batched-generation fast path in
    # CachedBasisPipeline (per-sample generation instead) -- slower, same math.
    validator_cfg = config.get("sgm_validator") or {}
    sample_validator = None
    retry_policy = None
    if validator_cfg.get("enabled", False):
        sample_validator = make_beam_target_validator(validator_cfg)
        retry_policy = RetryPolicy(
            max_retries=int(validator_cfg.get("max_retries", 20)),
            on_exhausted="yield_last",  # never raise mid-epoch from __getitem__
        )

    train_dataset = CachedBasisPipeline(
        train_provider,
        combinator=combinator,
        transforms=train_transforms,
        basis_position_extractor=make_centroid_position_extractor(
            method="first_moment",
            component=1,
        ),
        num_samples=config["data"]["total_train_samples"],
        seed=config["seed"],
        generation_batch_size=config["combinator"]["generation_batch_size"],
        sample_validator=sample_validator,
        retry_policy=retry_policy,
        eager=True,
    ).to_framework_dataset(dataset_ops=config["data"]["dataset_ops"])
    val_dataset = PyTorchPipeline(
        val_provider,
        eval_transforms,
    ).to_memory_dataset(config["data"]["dataset_ops"])
    test_dataset = PyTorchPipeline(
        test_provider,
        eval_transforms,
    ).to_memory_dataset(config["data"]["dataset_ops"])

    # ====================================
    # Legacy case templates kept for future reuse.
    # ====================================

    # ====================================
    # Case 1 :multiple datasets (source).
    # ====================================
    # train_provider = SqlProvider(
    #     sources={"connection": db_paths[0], "sql": config["sql"]["chromox_all"]},
    #     output_config={"list": "image_path"},
    # ).subsample(n_samples=config["data"]["total_train_samples"], seed=config["seed"])
    # eval_provider = SqlProvider(
    #     sources={"connection": db_paths[1], "sql": config["sql"]["chromox_laser"]},
    #     output_config={"list": "image_path"},
    # ).subsample(n_samples=config["data"]["total_val_samples"], seed=config["seed"])
    # val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"])
    # for t in config["data"]["transforms"]["torch"]:
    #     if t.get("name") == "add_parent_dir":
    #         t.setdefault("params", {})["parent_dir"] = str(dataset_dirs[0])
    #         break
    # transforms_1 = build_transforms_from_config(config["data"]["transforms"]["torch"])
    # try:
    #     for t in config["data"]["transforms"]["torch"]:
    #         if t.get("name") == "add_parent_dir":
    #             t.setdefault("params", {})["parent_dir"] = str(dataset_dirs[1])
    #             break
    #     transforms_2 = build_transforms_from_config(config["data"]["transforms"]["torch"])
    # except Exception as e:
    #     print("[WARNING] Failed to build transforms_2 with second dataset, falling back to transforms_1:", e)
    #     transforms_2 = transforms_1
    # train_dataset = PyTorchPipeline(train_provider, transforms_1).to_memory_dataset(config["data"]["dataset_ops"])
    # val_dataset = PyTorchPipeline(val_provider, transforms_2).to_memory_dataset(config["data"]["dataset_ops"])
    # test_dataset = PyTorchPipeline(test_provider, transforms_2).to_memory_dataset(config["data"]["dataset_ops"])

    # ====================================
    # Case 2 :single dataset (source).
    # ====================================
    # train_sql_key = config["data"].get("train_sql_key", "processed_chromox_cropped_line_scan_one_per_group")
    # eval_sql_key = config["data"].get("eval_sql_key", "processed_chromox_cropped_random_scan_eval")
    # train_provider = SqlProvider(
    #     sources={"connection": db_paths[0], "sql": config["sql"][train_sql_key]},
    #     output_config={"list": "image_path"},
    # )
    # eval_provider = SqlProvider(
    #     sources={"connection": db_paths[0], "sql": config["sql"][eval_sql_key]},
    #     output_config={"list": "image_path"},
    # )
    # val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"])
    # for t in config["data"]["transforms"]["torch"]:
    #     if t.get("name") == "add_parent_dir":
    #         t.setdefault("params", {})["parent_dir"] = str(dataset_dirs[0])
    #         break
    # transforms_1 = build_transforms_from_config(config["data"]["transforms"]["torch"])
    # transforms_2 = transforms_1.copy()
    # train_dataset = PyTorchPipeline(train_provider, transforms_1).to_memory_dataset(config["data"]["dataset_ops"])
    # val_dataset = PyTorchPipeline(val_provider, transforms_2).to_memory_dataset(config["data"]["dataset_ops"])
    # test_dataset = PyTorchPipeline(test_provider, transforms_2).to_memory_dataset(config["data"]["dataset_ops"])

    # ====================================
    # Case 3-a :Index-based combinator.
    # ====================================
    # train_provider = SqlProvider(
    #     sources={"connection": db_paths[0], "sql": config["sql"]["clear_dmd_position_basis"]},
    #     output_config={"list": "image_path"},
    # )
    # eval_provider = SqlProvider(
    #     sources={"connection": db_paths[0], "sql": config["sql"]["clear_dmd_eval"]},
    #     output_config={"list": "image_path"},
    # )
    # for t in config["data"]["transforms"]["torch"]:
    #     if t.get("name") == "add_parent_dir":
    #         t.setdefault("params", {})["parent_dir"] = str(dataset_dirs[0])
    #         break
    # transforms = build_transforms_from_config(config["data"]["transforms"]["torch"])
    # canvas = pattern_gen.DynamicPatterns(*config["simulation"]["canvas_size"])
    # canvas.set_postprocess_fns(build_transforms_from_config(config["simulation"]["process_functions"]))
    # canvas._distributions = [
    #     pattern_gen.StaticGaussianDistribution(canvas)
    #     for _ in range(config["simulation"]["total_Guassian_num"])
    # ]
    # canvas.set_threshold(config["simulation"]["minimum_pixel_threshold"])
    # stream = canvas.pattern_stream(
    #     std_1=config["simulation"]["std_1"],
    #     std_2=config["simulation"]["std_2"],
    #     max_intensity=config["simulation"]["max_intensity"],
    #     fade_rate=config["simulation"]["fade_rate"],
    #     distribution=config["simulation"]["distribution"],
    # )
    # combinator = IndexCombinator(
    #     pattern_provider=stream,
    #     transforms=build_transforms_from_config(config["combinator"]["transforms"]["torch"]),
    # )
    # val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"])
    # train_dataset = CachedBasisPipeline(
    #     train_provider,
    #     combinator=combinator,
    #     transforms=transforms,
    #     num_samples=config["data"]["total_train_samples"],
    #     seed=config["seed"],
    #     eager=True,
    # ).to_framework_dataset(dataset_ops=config["data"]["dataset_ops"])
    # val_dataset = PyTorchPipeline(val_provider, transforms).to_memory_dataset(config["data"]["dataset_ops"])
    # test_dataset = PyTorchPipeline(test_provider, transforms).to_memory_dataset(config["data"]["dataset_ops"])

    # ====================================
    # Case 4 :direct file loading pattern.
    # ====================================
    # folder_1 = dataset_sources[0]
    # train = FileProvider(root_paths=config["paths"][folder_1])
    # train_provider, eval_provider = train.split(config["data"]["train_val_split"], seed=config["seed"])
    # val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"], seed=config["seed"])
    # transforms = build_transforms_from_config(config["data"]["transforms"]["torch"])[1:]
    # transforms.insert(2, T.get("torch_to_grayscale"))
    # train_dataset = PyTorchPipeline(train_provider, transforms).to_memory_dataset(config["data"]["dataset_ops"])
    # val_dataset = PyTorchPipeline(val_provider, transforms).to_memory_dataset(config["data"]["dataset_ops"])
    # test_dataset = PyTorchPipeline(test_provider, transforms).to_memory_dataset(config["data"]["dataset_ops"])

    return {
        "train_provider": train_provider,
        "val_provider": val_provider,
        "test_provider": test_provider,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "dataset_sources": dataset_sources,
    }




# ========================================
# Shared Model Construction Helpers
# ========================================
def build_model_for_training(config: dict, train_dataset) -> dict:
    model_name = config['model']['name']
    scheduler = None
    criterion = None
    G = None
    D = None
    losses = None
    opt_g = None
    opt_d = None

    if model_name == "CAE":
        from models.CAE import Autoencoder2D
        model = instantiate(Autoencoder2D, config["model"], allow_extra_kwargs=True)
    elif model_name == "TM":
        from models.TM import TransmissionMatrix
        model = instantiate(
            TransmissionMatrix,
            config["model"],
            overrides={
                "input_height": config["data"]["input_shape"][0],
                "input_width":  config["data"]["input_shape"][1],
                "output_height": config["data"]["output_shape"][0],
                "output_width":  config["data"]["output_shape"][1],
                "initialization": "xavier",
            },
            allow_extra_kwargs=True
        )
    elif model_name == "SHL_DNN":
        from models.SHL_DNN import SHLNeuralNetwork
        model = SHLNeuralNetwork(
            input_size=config['data']['input_shape'][0] * config['data']['input_shape'][1],
            hidden_size=config['model']['hidden_size'],
            output_size=config['data']['output_shape'][0] * config['data']['output_shape'][1],
            dropout_rate=config['model']['dropout_rate'],
        )
    elif model_name == "U_Net":
        from models.U_Net import UNet
        model = instantiate(UNet, config["model"], allow_extra_kwargs=True)
    elif model_name == "SwinT":
        from models.SwinT import SwinUNet, ReconLoss
        model = instantiate(SwinUNet, config["model"], allow_extra_kwargs=True)
    elif model_name == "Pix2pix":
        from models.Pix2pix import Generator, Discriminator, Pix2PixLosses
        G = Generator(channels=config["model"]["channels"])
        D = Discriminator(channels=config["model"]["channels"])
        losses = Pix2PixLosses(lambda_l1=config["model"]["lambda_l1"])
        opt_g = torch.optim.Adam(G.parameters(), lr=config["training"]["learning_rate"], betas=config["training"]["betas"])
        opt_d = torch.optim.Adam(D.parameters(), lr=config["training"]["learning_rate"], betas=config["training"]["betas"])
        model = G
    elif model_name == "ERN":
        from models.ERN import EncoderRegressor
        model = instantiate(EncoderRegressor, config["model"], allow_extra_kwargs=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "Pix2pix":
        G = G.to(device)
        D = D.to(device)
        model = G
        optimizer = None
        info = show_model_info(G) + show_model_info(D)
    elif model_name == "SwinT":
        from torch.optim.lr_scheduler import LambdaLR

        total_steps = config['training']['epochs'] * len(train_dataset)
        warmup_steps = int(config['training']['warmup_ratio'] * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        model = model.to(device)
        criterion = ReconLoss(w_l1=config['training']['w_l1'], w_ssim=config['training']['w_ssim']) # Loss: L1 + 0.3*SSIM

        # Optimizer: AdamW with recommended params
        base_lr = 4e-4 if config['training']['batch_size'] >= 64 else 2e-4
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr, betas=config['training']['betas'],
            eps=config['training']['eps'], weight_decay=config['training']['weight_decay']
        )
        scheduler = LambdaLR(optimizer, lr_lambda)
        info = show_model_info(model)
    else:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        info = show_model_info(model)

    return {
        "model_name": model_name,
        "model": model,
        "device": device,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "info": info,
        "G": G,
        "D": D,
        "losses": losses,
        "opt_g": opt_g,
        "opt_d": opt_d,
    }


# ========================================
# Shared Metric 
# ========================================

def metric_debug(extract_fn):
    def metric(pred, target):
        # --- DEBUG: force a constant output ---
        return {"val_debug_mae": 0.0, "val_debug_rmse": 0.0}
    return metric

def debug_extract_fn(img, **kwargs):
    try:
        print("[DEBUG] extract_fn input shape:", getattr(img, "shape", None), type(img))
        return extract_beam_parameters_flat(img, **kwargs)
    except Exception as e:
        print("[WARNING] extract_fn failed with:", e)
        return None

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

# --- Param-based metric (no image extraction) ---
def make_param_metric(keys=("h_centroid","v_centroid","h_width","v_width")):
    import torch, math

    def _to_vec(x):
        # x: Tensor/np/list shaped (4,) or (B,4). Return 1D list[float].
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        x = x.reshape(-1).tolist()
        return x

    def metric(pred, target):
        p = _to_vec(pred)
        t = _to_vec(target)
        n = min(len(p), len(t))
        if n < 4:  # nothing to score
            return {}

        # take first len(keys) values
        p = p[:len(keys)]
        t = t[:len(keys)]

        out = {}
        diffs = [pi - ti for pi, ti in zip(p, t)]
        for k, d in zip(keys, diffs):
            out[f"val_{k}_mae"]  = abs(d)
            out[f"val_{k}_mse"]  = d*d
            out[f"val_{k}_rmse"] = abs(d)**0.5 if d >= 0 else (d*d)**0.5  # same as sqrt(mse)

        # simple overall
        overall_mae = sum(abs(d) for d in diffs) / len(diffs)
        overall_mse = sum(d*d for d in diffs) / len(diffs)
        out["val_overall_mae"]  = overall_mae
        out["val_overall_mse"]  = overall_mse
        out["val_overall_rmse"] = overall_mse ** 0.5
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
    
    
    
# Datapipeline validator
def max_value_validator(sample, record):
    """
    Return True if valid.
    Reject if any ndarray element is > 1.0.
    """
    # sample can be ndarray or tuple of ndarrays
    arrays = sample if isinstance(sample, tuple) else (sample,)

    for arr in arrays:
        if np.any(arr > 1.0):
            return (False, f"pixel value exceeded 1.0 (max={float(np.max(arr)):.6f})")
    return True
