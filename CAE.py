# autoencoder_pytorch.py
from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bools(xs: Sequence[Union[bool, int]]) -> List[bool]:
    return [bool(int(v)) for v in xs]


class SamePadConv2d(nn.Module):
    """
    2D Conv with TF-like 'padding=\"same\"' for any kernel/stride via dynamic ZeroPad2d.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, bias: bool = False):
        super().__init__()
        self.k = int(kernel_size)
        self.s = int(stride)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=self.k, stride=self.s, padding=0, bias=bias)

    def _pad(self, x):
        h, w = x.shape[-2], x.shape[-1]
        out_h = math.ceil(h / self.s)
        out_w = math.ceil(w / self.s)
        pad_h = max((out_h - 1) * self.s + (self.k - 1) + 1 - h, 0)
        pad_w = max((out_w - 1) * self.s + (self.k - 1) + 1 - w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self._pad(x))


class SamePadConvTranspose2d(nn.Module):
    """
    2D ConvTranspose with TF-like 'padding=\"same\"' for upsampling (stride>=2).
    Chooses padding/output_padding to make out = in * stride when possible.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 2, bias: bool = False):
        super().__init__()
        k, s = int(kernel_size), int(stride)
        # Solve: out = (in-1)*s - 2p + k + op  -> target out = in*s
        # => 2p = k + op - s, with 0 <= op < s
        op = (k - s) % s
        p = (k + op - s) // 2
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s,
                                         padding=p, output_padding=op, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class DownsampleBlock(nn.Module):
    """
    TF block: Conv2D(stride=2, same, no bias) -> [BatchNorm] -> LeakyReLU(alpha=0.3).
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, apply_batchnorm: bool):
        super().__init__()
        self.conv = SamePadConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if apply_batchnorm else None
        self.act = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class UpsampleBlock(nn.Module):
    """
    TF block: Conv2DTranspose(stride=2, same, no bias) -> BatchNorm -> [Dropout(0.5)] -> ReLU.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, apply_dropout: bool):
        super().__init__()
        self.deconv = SamePadConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(p=0.5) if apply_dropout else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(x)


def _init_weights_normal_002(m: nn.Module) -> None:
    """ Match Keras RandomNormal(stddev=0.02) for conv/deconv weights. """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Autoencoder2D(nn.Module):
    """
    TF-equivalent encoder-decoder without skip connections.
    """
    def __init__(
        self,
        in_channels: int,
        encoder: Sequence[int],
        decoder: Sequence[int],
        kernel_size: int = 4,
        apply_batchnorm: Optional[Sequence[Union[bool, int]]] = None,
        apply_dropout: Optional[Sequence[Union[bool, int]]] = None,
        out_channels: Optional[int] = None,
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        assert len(encoder) == len(decoder), "encoder and decoder must have same length"
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.in_channels
        self.kernel_size = int(kernel_size)

        # keep original specs for exact round-trip saving
        self._encoder_spec = [int(x) for x in encoder]
        self._decoder_spec = [int(x) for x in decoder]

        bn_flags = _to_bools(apply_batchnorm) if apply_batchnorm is not None else [True] * len(encoder)
        dp_flags = _to_bools(apply_dropout) if apply_dropout is not None else [False] * len(decoder)
        assert len(bn_flags) == len(encoder), "apply_batchnorm length mismatch"
        assert len(dp_flags) == len(decoder), "apply_dropout length mismatch"

        # Encoder
        enc_layers: List[nn.Module] = []
        prev = self.in_channels
        for out, use_bn in zip(self._encoder_spec, bn_flags):
            enc_layers.append(DownsampleBlock(prev, int(out), kernel_size=self.kernel_size, apply_batchnorm=use_bn))
            prev = int(out)
        self.encoder = nn.ModuleList(enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        prev = int(self._encoder_spec[-1])
        for out, use_dp in zip(self._decoder_spec, dp_flags):
            dec_layers.append(UpsampleBlock(prev, int(out), kernel_size=self.kernel_size, apply_dropout=use_dp))
            prev = int(out)
        self.decoder = nn.ModuleList(dec_layers)

        # Final conv + activation
        self.final_conv = SamePadConv2d(prev, self.out_channels, kernel_size=self.kernel_size, stride=1, bias=False)

        if final_activation.lower() == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_activation.lower() == "tanh":
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        # Init like TF RandomNormal(0, 0.02)
        self.apply(_init_weights_normal_002)

        # stash exact config for round-trip save/load
        self._config = {
            "model": {
                "name": self.__class__.__name__,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "encoder": list(self._encoder_spec),
                "decoder": list(self._decoder_spec),
                "apply_batchnorm": [int(isinstance(l.bn, nn.BatchNorm2d)) for l in self.encoder],
                "apply_dropout": [int(l.dropout is not None) for l in self.decoder],
                "final_activation": "sigmoid" if isinstance(self.final_act, nn.Sigmoid)
                                   else "tanh" if isinstance(self.final_act, nn.Tanh)
                                   else "none",
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        x = self.final_conv(x)
        return self.final_act(x)

    # ---------------- new APIs ----------------
    def save_model(self, save_dir: str, model_name: str = "model") -> str:
        """
        Save a single .pth with state_dict + minimal architecture info.
        Returns the file path.
        """
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}.pth")

        arch = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "encoder": list(self._encoder_spec),
            "decoder": list(self._decoder_spec),
            # infer flags from layers to avoid external config dependencies
            "apply_batchnorm": [int(isinstance(l.bn, nn.BatchNorm2d)) for l in self.encoder],
            "apply_dropout":  [int(l.dropout is not None) for l in self.decoder],
            "final_activation": (
                "sigmoid" if isinstance(self.final_act, nn.Sigmoid)
                else "tanh" if isinstance(self.final_act, nn.Tanh)
                else "none"
            ),
        }
        torch.save({"arch": arch, "state_dict": self.state_dict()}, path)
        return path

    @classmethod
    def load_model(
        cls,
        filepath: str,
        device: Optional[Union[str, torch.device]] = None,
        eval_mode: bool = True,
    ) -> "Autoencoder2D":
        """
        Load a model saved by save_model(); rebuilds arch and loads weights.
        Returns a model moved to `device` (if given) and set to eval() by default.
        """
        map_location = device if device is not None else "cpu"
        payload = torch.load(filepath, map_location=map_location)

        if "arch" not in payload or "state_dict" not in payload:
            raise ValueError("Checkpoint must contain 'arch' and 'state_dict'.")

        m = payload["arch"]
        model = cls(
            in_channels=int(m.get("in_channels", 1)),
            encoder=m["encoder"],
            decoder=m["decoder"],
            kernel_size=int(m.get("kernel_size", 4)),
            apply_batchnorm=m.get("apply_batchnorm", None),
            apply_dropout=m.get("apply_dropout", None),
            out_channels=m.get("out_channels", None),
            final_activation=str(m.get("final_activation", "sigmoid")),
        )
        model.load_state_dict(payload["state_dict"], strict=True)

        if device is not None:
            model = model.to(device)
        if eval_mode:
            model.eval()
        return model


