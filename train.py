# pip install xflow-py
from xflow import ConfigManager, FileProvider, PyTorchPipeline, show_model_info
from xflow.data import build_transforms_from_config
from xflow.utils import load_validated_config, save_image

import torch
import os
from datetime import datetime  
from config_utils import load_config

SAMPLE_FLATTENED = ['SHL_DNN']
REGRESSION = ['ERN'] # Encoder-regressor


# ==================== 
# Configuration
# ==================== 
# Create experiment output directory  (timestamped)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  

experiment_name = "TM"  # TM, SHL_DNN, U_Net, CAE, SwinT
folder_name = f"{experiment_name}-{timestamp}"  
config_manager = ConfigManager(load_config(f"{experiment_name}.yaml", experiment_name=folder_name))
config = config_manager.get()
config_manager.add_files(config["extra_files"])

experiment_output_dir = config["paths"]["output"]
os.makedirs(experiment_output_dir, exist_ok=True)

# ==================== 
# Prepare Dataset
# ====================
"""The old method has a test data leak problem due to the similarity in adjacent samples"""
# provider = FileProvider(config["paths"]["dataset"]).\
#     subsample(fraction=config["data"]["subsample_fraction"], seed=config["seed"]) # 
# train_provider, temp_provider = provider.split(ratio=config["data"]["train_val_split"], seed=config["seed"])
# val_provider, test_provider = temp_provider.split(ratio=config["data"]["val_test_split"], seed=config["seed"])

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
for left_parts, right_parts in test_dataset:
    # batch will be a tuple: (right_halves, left_halves) due to split_width
    print(f"Batch shapes: {left_parts.shape}, {right_parts.shape}")
    if experiment_name in SAMPLE_FLATTENED:
        save_image(left_parts[0].reshape(config['data']['input_shape']), config["paths"]["output"] + "/left_part.png")
        save_image(right_parts[0].reshape(config['data']['output_shape']), config["paths"]["output"] + "/right_part.png")
    else:
        save_image(left_parts[0], config["paths"]["output"] + "/left_part.png")
        save_image(right_parts[0], config["paths"]["output"] + "/right_part.png")
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
        out_channels=config["model"]["out_channels"],
        enc_channels=config["model"]["enc_channels"],
        dec_channels=config["model"]["dec_channels"],
        bottleneck_channels=config["model"]["bottleneck_channels"],
        use_skips=config["model"]["use_skips"],
        use_batchnorm=config["model"]["use_batchnorm"],
        act=config["model"]["act"],
        use_sigmoid=config["model"]["use_sigmoid"],
    )
elif experiment_name == "SwinT":
    from models.SwinT import SwinUNet
    model = SwinUNet(
        in_chans=config["model"]["in_chans"],
        out_chans=config["model"]["out_chans"],
        use_skips=config["model"]["use_skips"],
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
show_model_info(model)


# ==================== 
# Training
# ====================
from utils import make_beam_param_metric, extract_beam_parameters_flat, debug_extract_fn
from functools import partial

import xflow.extensions.physics
from xflow.trainers import TorchTrainer, build_callbacks_from_config
from xflow.extensions.physics.beam import extract_beam_parameters

# 1) loss/optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

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
else:
    extract_beam_parameters_dict = partial(extract_beam_parameters, as_array=False)
    beam_param_metric = make_beam_param_metric(extract_beam_parameters_dict)

# 3) run training
trainer = TorchTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    callbacks=callbacks,
    output_dir=config["paths"]["output"],
    data_pipeline=train_dataset,
    val_metrics=[beam_param_metric]
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
