# encoder_regressor.py
from __future__ import annotations
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn


class EncoderRegressor(nn.Module):
    """
    Conv encoder (downsample by 2 each block) + MLP regressor.
    Matches TF:
      - Conv2D(k=4, s=2, padding='same'), use_bias=False
      - BatchNorm except first block
      - LeakyReLU (Keras default alpha=0.3)
      - Flatten -> Dense 512 -> Dropout 0.5 -> Dense 256 -> Dropout 0.5 -> Dense 4 -> sigmoid
    L2 regularization: set via optimizer weight_decay (not inside the module).
    """

    def __init__(
        self,
        in_channels: int = 1,
        kernel_size: int = 4,
        encoder: List[int] = [64, 128, 256, 512, 512, 1024],
        decoder: List[int] = [512, 256, 4],
        final_activation: str = "sigmoid",
    ):
        super().__init__()

        self.config = dict(
            in_channels=in_channels,
            kernel_size=kernel_size,
            encoder=list(encoder),
            decoder=list(decoder),
            final_activation=final_activation,
        )

        # ---- Encoder: Conv (stride=2, "same"-like), BN (except first), LeakyReLU ----
        pad = (kernel_size - 2) // 2  # for k=4 -> 1, matches 'same' when s=2 for even sizes like 256
        blocks = []
        prev_c = in_channels
        for i, out_c in enumerate(encoder):
            layers = [nn.Conv2d(prev_c, out_c, kernel_size, stride=2, padding=pad, bias=False)]
            if i != 0:  # no BN on first, like TF
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.3, inplace=True))  # Keras default alpha=0.3
            blocks.append(nn.Sequential(*layers))
            prev_c = out_c
        self.encoder = nn.Sequential(*blocks)

        # ---- Regressor (Flatten -> [512, 256, 4]) ----
        # Use LazyLinear for the first FC so we don't need img_size.
        mlp = []
        mlp.append(nn.LazyLinear(decoder[0]))  # in_features resolved on first forward
        mlp.append(nn.ReLU(inplace=True))
        mlp.append(nn.Dropout(p=0.5))
        mlp.append(nn.Linear(decoder[0], decoder[1]))
        mlp.append(nn.ReLU(inplace=True))
        mlp.append(nn.Dropout(p=0.5))
        mlp.append(nn.Linear(decoder[1], decoder[2]))  # -> 4
        self.mlp = nn.Sequential(*mlp)

        self.final_activation = (final_activation or "linear").lower()

        # Init conv weights ~ N(0, 0.02) like TF RandomNormal(0,0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x

    # -----------------------------
    # Save / Load
    # -----------------------------
    def save_model(self, save_dir: str, model_name: str = "model") -> str:
        """
        Save state_dict + config in one .pth file. Returns the path.
        """
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}.pth")
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)
        return path

    @classmethod
    def load_model(cls, filepath: str, device: Optional[Union[str, torch.device]] = None) -> "EncoderRegressor":
        """
        Load a model saved by save_model(). Uses a 256×256 dummy pass to materialize LazyLinear.
        """
        dev = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
        ckpt = torch.load(filepath, map_location=dev)
        config = ckpt.get("config")
        if config is None:
            raise ValueError("Checkpoint missing 'config'. Use save_model() to create compatible files.")

        model = cls(**config).to(dev)
        # Materialize LazyLinear before loading weights (assumes training used 256×256 like TF):
        with torch.no_grad():
            _ = model(torch.zeros(1, config["in_channels"], 256, 256, device=dev))
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        return model
