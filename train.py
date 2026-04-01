# pip install xflow-py
from xflow import ConfigManager, build_model_report
from xflow.utils import save_image

import torch
from pathlib import Path
from datetime import datetime  
from functools import partial
from config_utils import load_config, detect_machine
from utils import *

def main():
    # Future CLI parameters
    # ========================================
    # Configuration
    # ========================================
    # Create experiment output directory  (timestamped)

    experiment_name = "CLEAR25"
    dataset_sources = ["processed_yag", "processed_yag_laser"]  # ["processed_dmd", "processed_chromox", "processed_yag", "processed_chromox_laser", "processed_yag_laser"]
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
    # need to manually resolve all complexity in utils, in here just interface to build final variables needed.
    # Mainly just change providers, the pipelines are decoupled.
    # ========================================
    dataset_bundle = build_datasets(config, dataset_sources)
    train_provider = dataset_bundle["train_provider"]
    val_provider = dataset_bundle["val_provider"]
    test_provider = dataset_bundle["test_provider"]
    train_dataset = dataset_bundle["train_dataset"]
    val_dataset = dataset_bundle["val_dataset"]
    test_dataset = dataset_bundle["test_dataset"]

    print("Samples: ", len(train_provider), len(val_provider), len(test_provider))
    print("Batch: ", len(train_dataset), len(val_dataset), len(test_dataset))

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
    # also need to resolve all complexity in utils
    # ========================================
    model_bundle = build_model_for_training(config, train_dataset)
    model = model_bundle["model"]
    device = model_bundle["device"]
    optimizer = model_bundle["optimizer"]
    scheduler = model_bundle["scheduler"]
    info = model_bundle["info"]
    G = model_bundle["G"]
    D = model_bundle["D"]
    losses = model_bundle["losses"]
    opt_g = model_bundle["opt_g"]
    opt_d = model_bundle["opt_d"]

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
            scheduler=scheduler if model_name == "SwinT" else None,
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


if __name__ == "__main__":
    main()