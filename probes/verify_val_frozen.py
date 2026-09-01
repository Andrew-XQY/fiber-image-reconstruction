"""Executable check: a validation pass on the real-beam val split leaves the
model bit-identical (parameters AND buffers), on the exact trainer code path.

Stage 1  positive control: one train_step on random noise changes weights.
Stage 2  TorchTrainer.val_step over every real-beam val batch: 0 tensors change.
Stage 3  full TorchTrainer.fit() with the production callbacks (early stopping,
         progress bar, reconstruction callback on the TEST split), 3 epochs of
         one noise batch each: after every validation the state equals the state
         right after that epoch's train step, and the final weights equal the
         snapshot of the best-val epoch (restore_best).

Run from repo root on a machine with the datasets + xflow (no basis caching):
    EXPERIMENT_CONFIG=CLEAR26_sgm_cam3 python -m probes.verify_val_frozen
"""
import hashlib
import os
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from config_utils import detect_machine, load_config  # noqa: E402
from utils import (  # noqa: E402
    build_eval_datasets,
    build_model_for_training,
    make_beam_param_metric,
)
from xflow import ConfigManager, TorchTrainer  # noqa: E402
from xflow.extensions.physics.beam import extract_beam_parameters  # noqa: E402
from xflow.trainers import build_callbacks_from_config  # noqa: E402


def snapshot(model):
    """sha256 of every parameter and buffer (bit-level identity, low memory)."""
    return {
        k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for k, v in model.state_dict().items()
    }


def changed_keys(a, b):
    return [k for k in a if a[k] != b[k]]


def main():
    name = os.getenv("EXPERIMENT_CONFIG", "CLEAR26_sgm_cam3")
    config = ConfigManager(
        load_config(f"{name}.yaml", experiment_name=f"{name}-verify-val-frozen",
                    machine=detect_machine(), resolve=True)
    ).get()
    assert config["model"]["name"] == "CAE", "script written for the CAE path"
    torch.manual_seed(int(config["seed"]))

    out_dir = Path(config["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_bundle = build_eval_datasets(config)  # real-beam val/test only
    val_loader = eval_bundle["val_dataset"]
    test_loader = eval_bundle["test_dataset"]

    bundle = build_model_for_training(config, None)
    model, device, optimizer = bundle["model"], bundle["device"], bundle["optimizer"]
    metric = make_beam_param_metric(partial(extract_beam_parameters, as_array=False))

    h, w = config["data"]["input_shape"]
    noise_batch = (torch.rand(2, 1, h, w), torch.rand(2, 1, h, w))
    noise_loader = [noise_batch]

    # ---------------- Stage 1 + 2: direct val_step path ----------------
    trainer = TorchTrainer(
        model=model, optimizer=optimizer, criterion=torch.nn.MSELoss(),
        device=device, callbacks=[], output_dir=str(out_dir),
        data_pipeline=val_loader, val_metrics=[metric],
    )
    model.train()
    s0 = snapshot(model)
    trainer.train_step(noise_batch)
    s1 = snapshot(model)
    n_changed = len(changed_keys(s0, s1))
    print(f"[1] control: train_step on noise changed {n_changed}/{len(s0)} tensors (expect >0)")
    assert n_changed > 0

    model.eval()  # trainer.py: self.model.eval() before the val loop
    before = snapshot(model)
    n_batches = 0
    for batch in val_loader:
        trainer.val_step(batch)  # trainer.py: torch.no_grad() forward + metrics
        n_batches += 1
    after = snapshot(model)
    bad = changed_keys(before, after)
    print(f"[2] {n_batches} real-beam val batches via TorchTrainer.val_step: "
          f"{len(bad)} tensors changed (expect 0); model.training={model.training}")
    if bad:
        print("    CHANGED:", bad)
        sys.exit(1)

    # ---------------- Stage 3: full fit() with production callbacks ----------------
    after_train_step = {}
    after_val = {}

    class RecordingTrainer(TorchTrainer):
        def train_step(self, batch):
            logs = super().train_step(batch)
            after_train_step[self._epoch] = snapshot(self.model)
            return logs

        def fit(self, **kw):
            self._epoch = -1
            return super().fit(**kw)

    class EpochCounter:  # records epoch index and post-validation state
        def on_epoch_begin(self, epoch, trainer=None, **kw):
            trainer._epoch = epoch

        def on_epoch_end(self, epoch, model=None, **kw):
            after_val[epoch] = snapshot(model)

    callbacks = build_callbacks_from_config(config=config["callbacks"],
                                            framework=config["framework"])
    callbacks[-1].set_dataset(test_loader)
    callbacks[-1].set_training_dataset(noise_loader)
    trainer3 = RecordingTrainer(
        model=model, optimizer=optimizer, criterion=torch.nn.MSELoss(),
        device=device, callbacks=[EpochCounter()] + callbacks,
        output_dir=str(out_dir), data_pipeline=noise_loader, val_metrics=[metric],
    )
    history = trainer3.fit(epochs=3, train_loader=noise_loader, val_loader=val_loader)

    for ep in sorted(after_val):
        bad = changed_keys(after_train_step[ep], after_val[ep])
        print(f"[3] epoch {ep}: state after validation vs after train step: "
              f"{len(bad)} tensors differ (expect 0)")
        assert not bad, bad
    best_ep = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
    final = snapshot(model)
    bad = changed_keys(after_train_step[best_ep], final)
    print(f"[3] restore_best: final weights == epoch {best_ep} (lowest val_loss) weights: "
          f"{len(bad)} tensors differ (expect 0)")
    assert not bad, bad
    print("PASS: real-beam validation never modifies parameters or buffers; "
          "it only selects which trained epoch is kept.")


if __name__ == "__main__":
    main()
