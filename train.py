# pip install xflow-py
from xflow import ConfigManager, FileProvider, SqlProvider, PyTorchPipeline, build_model_report, TransformRegistry as T, instantiate, show_model_info
from xflow.data import build_transforms_from_config
from xflow.utils import save_image, resolve_resource_dir
from xflow.extensions.physics.pipeline import CachedBasisPipeline, IndexCombinator
from xflow.extensions.physics import pattern_gen

import torch
import os
from datetime import datetime  
from functools import partial
from config_utils import load_config, detect_machine
from utils import *

# Future CLI parameters
# ========================================
# Configuration
# ========================================
# Create experiment output directory  (timestamped)

experiment_name = "CLEAR25"
dataset_sources = ["processed_chromox"]  # ["dataset_1", "dataset_2"]
folder_name = f"{experiment_name}-{datetime.now():%Y%m%d%H%M%S}"

config_manager = ConfigManager(
    load_config(
        f"{experiment_name}.yaml",
        experiment_name=folder_name,
        machine=detect_machine(),
        resolve=True,
    )
)
config = config_manager.get()
config_manager.add_files(config.get("extra_files", []))
config_manager["dataset_used"] = dataset_sources

output_dir = Path(config["paths"]["output"])
output_dir.mkdir(parents=True, exist_ok=True)

# ======================================== 
# Prepare Dataset
# ========================================
def resolve_dataset_dir(path_or_key: str) -> Path:
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

dataset_dirs = [resolve_dataset_dir(src) for src in dataset_sources]
db_rel = config["dataset_structure"]["db"].lstrip("/\\")
db_paths = [d / db_rel for d in dataset_dirs]

# train_provider = SqlProvider(
#     sources={"connection": db_paths[0], "sql": config["sql"]["chromox_line_scan"]}, output_config={'list': "image_path"}
# ).subsample(n_samples=config["data"]["total_train_samples"], seed=config["seed"])
# eval_provider = SqlProvider(
#     sources={"connection": db_paths[0], "sql": config["sql"]["chromox_random_scan"]}, output_config={'list': "image_path"}
# ).subsample(n_samples=config["data"]["total_val_samples"], seed=config["seed"])
# val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"])


train_provider = SqlProvider(
    sources={"connection": db_paths[0], "sql": config["sql"]["chromox_random_scan"]}, output_config={'list': "image_path"}
)
train_provider, eval_provider = train_provider.split(config["data"]["train_val_split"])
val_provider, test_provider = eval_provider.split(config["data"]["val_test_split"])


# pad abs path to db saved relative dirs.
for t in config["data"]["transforms"]["torch"]:
    if t.get("name") == "add_parent_dir":
        t.setdefault("params", {})["parent_dir"] = str(dataset_dirs[0])
        break
    
transforms = build_transforms_from_config(config["data"]["transforms"]["torch"])

# # ========== SGM simulation pattern generator ==========
# canvas = pattern_gen.DynamicPatterns(*config["simulation"]["canvas_size"])
# canvas.set_postprocess_fns(build_transforms_from_config(config["simulation"]["process_functions"]))
# canvas._distributions = [pattern_gen.StaticGaussianDistribution(canvas) for _ in range(config["simulation"]["total_Guassian_num"])]
# canvas.set_threshold(config["simulation"]["minimum_pixel_threshold"])
# stream = canvas.pattern_stream(
#     std_1=config["simulation"]["std_1"], 
#     std_2=config["simulation"]["std_2"],
#     max_intensity=config["simulation"]["max_intensity"], 
#     fade_rate=config["simulation"]["fade_rate"], 
#     distribution=config["simulation"]["distribution"]
# ) 

# # ======== combinator using index + SGM ========
# combinator = IndexCombinator(
#     pattern_provider=stream,
#     transforms= build_transforms_from_config(config["combinator"]["transforms"]["torch"]),
# )

# train_dataset = CachedBasisPipeline(
#     train_provider, 
#     combinator=combinator, 
#     transforms=transforms, 
#     num_samples=config["data"]["total_train_samples"], 
#     seed=config["seed"],
#     eager=True
# ).to_framework_dataset(framework=config["framework"], dataset_ops=config["data"]["dataset_ops"])

train_dataset = PyTorchPipeline(
    train_provider, 
    transforms
).to_memory_dataset(config["data"]["dataset_ops"])  

val_dataset = PyTorchPipeline(
    val_provider, 
    transforms
).to_memory_dataset(config["data"]["dataset_ops"])   # testset data do not need thresholding since it is to remove stacking noise?

test_dataset = PyTorchPipeline(
    test_provider,
    transforms
).to_memory_dataset(config["data"]["dataset_ops"])



print("Samples: ",len(train_provider),len(val_provider),len(test_provider))
print("Batch: ",len(train_dataset),len(val_dataset),len(test_dataset))

model_name = config['model']['name']

# save a sample from dataset for debugging
if model_name in REGRESSION:
    for left_parts, params, right_parts in test_dataset:
        print("Sample types: ", type(left_parts[0]))
        print(f"Batch shapes: {left_parts.shape}, {right_parts.shape}")
        save_image(left_parts[0], config["paths"]["output"] + "/input.png")
        save_image(right_parts[0], config["paths"]["output"] + "/output.png")
        break
else:
    for left_parts, right_parts in test_dataset:
        # batch will be a tuple: (right_halves, left_halves) due to split_width
        print("Sample types: ", type(left_parts[0]))
        print(f"Batch shapes: {left_parts.shape}, {right_parts.shape}")
        if model_name in SAMPLE_FLATTENED:
            save_image(left_parts[0].reshape(config['data']['input_shape']), config["paths"]["output"] + "/input.png")
            save_image(right_parts[0].reshape(config['data']['output_shape']), config["paths"]["output"] + "/output.png")
        else:
            save_image(left_parts[0], config["paths"]["output"] + "/input.png")
            save_image(right_parts[0], config["paths"]["output"] + "/output.png")
        break


# ======================================== 
# Construct Model
# ======================================== 
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
elif model_name == "ERN":
    from models.ERN import EncoderRegressor
    model = instantiate(EncoderRegressor, config["model"], allow_extra_kwargs=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == "Pix2pix":
    G = G.to(device)
    D = D.to(device)
    model = G
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

print(info)
model_device = next(model.parameters()).device
report = build_model_report(
    model,
    lambda: model(torch.randn(1, 1, *config["data"]["input_shape"], device=model_device))
)
with open(f"{config['paths']['output']}/model_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
config_manager.save(output_dir=config["paths"]["output"], config_filename=config["name"])


# ======================================== 
# Training
# ======================================== 
from xflow import TorchTrainer, TorchGANTrainer
from xflow.trainers import build_callbacks_from_config
from xflow.extensions.physics.beam import extract_beam_parameters

# 1) loss/optimizer
criterion = torch.nn.MSELoss()  # Pixel wise MSE loss.

# 2) callbacks (unchanged) + any custom wiring
callbacks = build_callbacks_from_config(
    config=config["callbacks"],
    framework=config["framework"],  
) # keep dataset closure for last callback, sequence hardcoded
callbacks[-1].set_dataset(test_dataset)
callbacks[-1].set_training_dataset(train_dataset)

# Extract beam parameters closure (return as dict)
if model_name in SAMPLE_FLATTENED:
    extract_beam_parameters_dict = partial(extract_beam_parameters_flat, as_array=False)
    beam_param_metric = make_beam_param_metric(extract_beam_parameters_dict)
elif model_name in REGRESSION:   # e.g., "ERN"
    beam_param_metric = make_param_metric()
else:
    extract_beam_parameters_dict = partial(extract_beam_parameters, as_array=False)
    beam_param_metric = make_beam_param_metric(extract_beam_parameters_dict)

# 3) run training
if model_name in GAN:
    trainer = TorchGANTrainer(
        generator=G,
        discriminator=D,
        optimizer_g=opt_g,
        optimizer_d=opt_d,
        losses=losses,
        device=device,
        callbacks=callbacks,
        output_dir=config["paths"]["output"],
        data_pipeline=train_dataset,
        val_metrics=[beam_param_metric],
    )
else:
    trainer = TorchTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        output_dir=config["paths"]["output"],
        data_pipeline=train_dataset,
        val_metrics=[beam_param_metric],
        scheduler= scheduler if model_name == "SwinT" else None, 
        scheduler_step_per_batch=True,
    )

history = trainer.fit(
    train_loader=train_dataset, 
    val_loader=val_dataset,
    epochs=config['training']['epochs'],
)

# 4) save results
trainer.save_history(f"{config['paths']['output']}/history.json")
trainer.save_model(config["paths"]["output"])  # uses model.save_model(...) if available
print("Training ALL complete.")