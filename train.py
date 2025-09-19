# pip install xflow-py
from xflow import ConfigManager, FileProvider, PyTorchPipeline, show_model_info
from xflow.data import build_transforms_from_config
from xflow.utils import load_validated_config, save_image
import xflow.extensions.physics

import torch
import os
from datetime import datetime  
from config_utils import load_config
from utils import *


# ==================== 
# Configuration
# ==================== 
# Future CLI parameters
EVALUATE_ON_TEST = False  # whether to run evaluation on test set after training


# Create experiment output directory  (timestamped)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  

experiment_name = "CAE_syth"  # TM, SHL_DNN, U_Net, Pix2pix, ERN, CAE, SwinT, CAE_syth
folder_name = f"{experiment_name}-{timestamp}"  
config_manager = ConfigManager(load_config(f"{experiment_name}.yaml", experiment_name=folder_name))
config = config_manager.get()
config_manager.add_files(config["extra_files"])

experiment_output_dir = config["paths"]["output"]
os.makedirs(experiment_output_dir, exist_ok=True)

# ==================== 
# Prepare Dataset
# ====================
training_folder = os.path.join(config["paths"]["dataset"], config["data"]["training_set"])
evaluation_folder = os.path.join(config["paths"]["dataset"], config["data"]["evaluation_set"])
train_provider = FileProvider(training_folder).subsample(fraction=config["data"]["subsample_fraction"], seed=config["seed"]) 
evaluation_provider = FileProvider(evaluation_folder).subsample(fraction=config["data"]["subsample_fraction"], seed=config["seed"]) 
val_provider, test_provider = evaluation_provider.split(ratio=config["data"]["val_test_split"], seed=config["seed"])

transforms = build_transforms_from_config(config["data"]["transforms"]["torch"])
def make_dataset(provider):
    return PyTorchPipeline(provider, transforms).to_memory_dataset(config["data"]["dataset_ops"])

train_dataset = make_dataset(train_provider)
val_dataset = make_dataset(val_provider)
test_dataset = make_dataset(test_provider)

print("Samples: ",len(train_provider),len(val_provider),len(test_provider))
print("Batch: ",len(train_dataset),len(val_dataset),len(test_dataset))

# save a sample from dataset for debugging
if experiment_name in REGRESSION:
    for left_parts, params, right_parts in test_dataset:
        print(f"Batch shapes: {left_parts.shape}, {right_parts.shape}")
        save_image(left_parts[0], config["paths"]["output"] + "/input.png")
        save_image(right_parts[0], config["paths"]["output"] + "/output.png")
        break
else:
    for left_parts, right_parts in test_dataset:
        # batch will be a tuple: (right_halves, left_halves) due to split_width
        print(f"Batch shapes: {left_parts.shape}, {right_parts.shape}")
        if experiment_name in SAMPLE_FLATTENED:
            save_image(left_parts[0].reshape(config['data']['input_shape']), config["paths"]["output"] + "/input.png")
            save_image(right_parts[0].reshape(config['data']['output_shape']), config["paths"]["output"] + "/output.png")
        else:
            save_image(left_parts[0], config["paths"]["output"] + "/input.png")
            save_image(right_parts[0], config["paths"]["output"] + "/output.png")
        break

# ==================== 
# Construct Model
# ====================
if experiment_name == "CAE":
    from models.CAE import Autoencoder2D
    model = Autoencoder2D(
        in_channels=int(config['model']["in_channels"]),
        encoder=config['model']["encoder"],
        decoder=config['model']["decoder"],
        kernel_size=int(config['model']["kernel_size"]),
        apply_batchnorm=config['model']["apply_batchnorm"],
        apply_dropout=config['model']["apply_dropout"],
        final_activation=str(config['model']["final_activation"]),
    )
elif experiment_name == "TM":
    from models.TM import TransmissionMatrix
    model = TransmissionMatrix(
        input_height = config["data"]["input_shape"][0],
        input_width = config["data"]["input_shape"][1],
        output_height = config["data"]["output_shape"][0],
        output_width = config["data"]["output_shape"][1],
        initialization = "xavier",
    )
elif experiment_name == "SHL_DNN":
    from models.SHL_DNN import SHLNeuralNetwork
    model = SHLNeuralNetwork(
        input_size=config['data']['input_shape'][0] * config['data']['input_shape'][1],
        hidden_size=config['model']['hidden_size'], 
        output_size=config['data']['output_shape'][0] * config['data']['output_shape'][1],
        dropout_rate=config['model']['dropout_rate'],
    )
elif experiment_name == "U_Net":
    from models.U_Net import UNet
    model = UNet(
        in_channels=config["model"]["in_channels"],
        encoder=config["model"]["encoder"],
        decoder=config["model"]["decoder"],
        kernel_size=config["model"]["kernel_size"],
        apply_batchnorm=config["model"]["apply_batchnorm"],
        apply_dropout=config["model"]["apply_dropout"],
        out_channels=config["model"]["out_channels"],
        final_activation=config["model"]["final_activation"],
    )
elif experiment_name == "SwinT":
    from models.SwinT import SwinUNet, ReconLoss
    model = SwinUNet(
        img_size=config['model']['img_size'],
        in_chans=config['model']['in_chans'],
        out_chans=config['model']['out_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size'],
        patch_size=config['model']['patch_size'],
    )
elif experiment_name == "Pix2pix":
    from models.Pix2pix import Generator, Discriminator, Pix2PixLosses
    G = Generator(channels=config["model"]["channels"])
    D = Discriminator(channels=config["model"]["channels"])
    losses = Pix2PixLosses(lambda_l1=config["model"]["lambda_l1"])
    opt_g = torch.optim.Adam(G.parameters(), lr=config["training"]["learning_rate"], betas=config["training"]["betas"])
    opt_d = torch.optim.Adam(D.parameters(), lr=config["training"]["learning_rate"], betas=config["training"]["betas"])
elif experiment_name == "ERN":
    from models.ERN import EncoderRegressor
    model = EncoderRegressor(
            in_channels=config['model']['in_channels'],
            kernel_size=config['model']['kernel_size'],
            encoder=config['model']['encoder'],
            decoder=config['model']['decoder'],
            final_activation=config['model']['final_activation'],  
        )
elif experiment_name == "CAE_syth":
    from models.CAE import Autoencoder2D
    model = Autoencoder2D(
        in_channels=int(config['model']["in_channels"]),
        encoder=config['model']["encoder"],
        decoder=config['model']["decoder"],
        kernel_size=int(config['model']["kernel_size"]),
        apply_batchnorm=config['model']["apply_batchnorm"],
        apply_dropout=config['model']["apply_dropout"],
        final_activation=str(config['model']["final_activation"]),
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if experiment_name == "Pix2pix":
    G = G.to(device)
    D = D.to(device)
    show_model_info(G)
    show_model_info(D)
elif experiment_name == "SwinT":
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
    show_model_info(model)
else:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    show_model_info(model)
    

# ==================== 
# Training
# ====================
from functools import partial

from xflow import TorchTrainer, TorchGANTrainer
from xflow.trainers import build_callbacks_from_config
from xflow.extensions.physics.beam import extract_beam_parameters

# 1) loss/optimizer
criterion = torch.nn.MSELoss()

# 2) callbacks (unchanged) + any custom wiring
callbacks = build_callbacks_from_config(
    config=config["callbacks"],
    framework=config["framework"],  
) # keep dataset closure for last callback, sequence hardcoded
callbacks[-1].set_dataset(test_dataset)

# Extract beam parameters closure (return as dict)
if experiment_name in SAMPLE_FLATTENED:
    extract_beam_parameters_dict = partial(extract_beam_parameters_flat, as_array=False)
    beam_param_metric = make_beam_param_metric(extract_beam_parameters_dict)
elif experiment_name in REGRESSION:   # e.g., "ERN"
    beam_param_metric = make_param_metric()
else:
    extract_beam_parameters_dict = partial(extract_beam_parameters, as_array=False)
    beam_param_metric = make_beam_param_metric(extract_beam_parameters_dict)

# 3) run training
if experiment_name in GAN:
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
        scheduler= scheduler if experiment_name == "SwinT" else None, 
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
config_manager.save(output_dir=config["paths"]["output"], config_filename=config["name"])

print("Training ALL complete.")

if EVALUATE_ON_TEST:
    print("Evaluating on TEST set...")
    results = trainer.evaluate(test_dataset, metrics=[beam_param_metric])
    print(results)
    with open(f"{config['paths']['output']}/test_results.txt", "w") as f:
        f.write(str(results))
    print("Evaluation complete.")