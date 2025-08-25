# pix2pix_pytorch.py
from __future__ import annotations
from typing import List, Optional, Sequence, Union, Dict, Any, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- Utils -------------------------

def _init_weights(m: nn.Module):
    """Match TF initializer ~ N(0, 0.02) for convs/transposed convs."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def _act_from_name(name: Optional[str]) -> nn.Module:
    if name is None or name.lower() in {"", "identity", "linear"}:
        return nn.Identity()
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported final_activation: {name}")

def _same_pad_stride2_for_k4(kernel_size: int) -> int:
    """
    TF 'padding=same' with stride=2 & k=4 corresponds to p=1.
    This model is designed for k=4 (pix2pix default). Using other k may change shapes.
    """
    if kernel_size != 4:
        # Best-effort fallback; for k=3 this gives 0 which differs from TF 'same'.
        return max((kernel_size - 2) // 2, 0)
    return 1

# ------------------------- Building Blocks -------------------------

class Downsample(nn.Module):
    """Conv → (BN) → LeakyReLU, stride=2."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 4, apply_batchnorm: bool = True):
        super().__init__()
        p = _same_pad_stride2_for_k4(k)
        layers: List[nn.Module] = [
            # nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=2, padding=p, bias=False)
            nn.Conv2d(in_ch, out_ch, k, stride=2, padding=p, bias=(not apply_batchnorm))
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, apply_dropout=False, mode="resize_conv", norm="in"):
        super().__init__()

        def _norm(ch):
            return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False) if norm=="in" else nn.BatchNorm2d(ch)

        if mode == "resize_conv":
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                _norm(out_ch),
                nn.Dropout(0.5) if apply_dropout else nn.Identity(),
                nn.ReLU(inplace=True),
            )
        else:
            p = _same_pad_stride2_for_k4(k)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=2, padding=p, bias=False),
                _norm(out_ch),
                nn.Dropout(0.5) if apply_dropout else nn.Identity(),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------- Generator (U-Net) -------------------------

class GeneratorUNet(nn.Module):
    """
    Pix2Pix U-Net generator (TF-compatible):
    - encoder: channels for each down block (first down has no BN)
    - decoder: channels for up blocks BEFORE the final output layer (see note)
    - final layer: ConvTranspose to out_channels with final_activation ('tanh' in TF code)
    - use_skips: concatenate encoder features (U-Net) if True; else plain encoder-decoder

    IMPORTANT: In the provided TF code, only len(encoder)-1 up blocks are paired with skips via zip(),
    and the final upsampling is done by the last output layer. So we *use only the first
    len(encoder)-1 entries of `decoder`*; any extra entries are ignored (matching TF behavior).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 4,
        encoder: Sequence[int] = (64, 128, 128, 256, 512, 512),
        decoder: Sequence[int] = (512, 512, 256, 128, 128, 64),
        final_activation: Optional[str] = "tanh",
        use_skips: bool = True,
        dropout_up_idx: Optional[Sequence[int]] = (0,),  # dropout on first up block by default
    ):
        super().__init__()
        self.hparams: Dict[str, Any] = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            encoder=list(encoder),
            decoder=list(decoder),
            final_activation=final_activation,
            use_skips=use_skips,
            dropout_up_idx=list(dropout_up_idx) if dropout_up_idx is not None else [],
        )

        k = kernel_size
        enc = list(encoder)
        assert len(enc) >= 2, "Need at least two encoder stages."

        # Build encoder (first down has no BN like TF when apply_batchnorm=False).
        downs: List[nn.Module] = []
        prev = in_channels
        for i, ch in enumerate(enc):
            downs.append(Downsample(prev, ch, k=k, apply_batchnorm=(i != 0)))
            prev = ch
        self.downs = nn.ModuleList(downs)

        # Compute skip channel sizes for concatenation (exclude bottleneck)
        skip_chs = list(enc[:-1])[::-1]
        num_ups = len(enc) - 1

        if len(decoder) < num_ups:
            raise ValueError(
                f"decoder must have at least {num_ups} entries; got {len(decoder)}"
            )

        # Build up blocks (use only first num_ups entries in decoder → TF zip behavior)
        ups: List[Upsample] = []
        cur_in = enc[-1]  # bottleneck channels
        for i in range(num_ups):
            out_ch = decoder[i]
            # ups.append(Upsample(cur_in, out_ch, k=k, apply_dropout=(i in (dropout_up_idx or []))))
            ups.append(Upsample(cur_in, out_ch, k=k, apply_dropout=(i in (dropout_up_idx or [])),
                    mode="resize_conv", norm="in"))
            # after upsample, if we use skips the next in_ch includes concatenation
            cur_in = out_ch + (skip_chs[i] if use_skips else 0)
        self.ups = nn.ModuleList(ups)

        # Final upsampling layer to target channels (bias=True like TF final Conv2DTranspose)
        p = _same_pad_stride2_for_k4(k)
        self.last = nn.ConvTranspose2d(cur_in, out_channels, kernel_size=k, stride=2, padding=p, bias=True)
        self.final_act = _act_from_name(final_activation)

        # Init
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward & collect skips
        skips: List[torch.Tensor] = []
        for d in self.downs:
            x = d(x)
            skips.append(x)

        # Prepare skip list (exclude bottleneck, reverse)
        use_skips = self.hparams["use_skips"]
        if use_skips:
            skips_use = skips[:-1][::-1]
            # Pair each up with a skip (matches TF zip behavior)
            for up, skip in zip(self.ups, skips_use):
                x = up(x)
                x = torch.cat([x, skip], dim=1)
        else:
            for up in self.ups:
                x = up(x)

        x = self.last(x)
        x = self.final_act(x)
        return x

    # -------------------- Persistence APIs --------------------

    def save_model(self, save_dir: str, model_name: str = "model") -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}.pth")
        payload = {
            "class": self.__class__.__name__,
            "hparams": self.hparams,
            "state_dict": self.state_dict(),
            "pytorch_version": torch.__version__,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load_model(
        cls,
        filepath: str,
        device: Optional[Union[str, torch.device]] = None,
        eval_mode: bool = True,
    ) -> "GeneratorUNet":
        map_location = torch.device(device) if isinstance(device, str) else device
        payload = torch.load(filepath, map_location=map_location)
        h = payload["hparams"]
        model = cls(**h)
        model.load_state_dict(payload["state_dict"], strict=True)
        if map_location is not None:
            model.to(map_location)
        if eval_mode:
            model.eval()
        return model

# ------------------------- Discriminator (PatchGAN) -------------------------

class PatchDiscriminator(nn.Module):
    """
    TF-compatible PatchGAN:
    concat([inp, tar]) → C64 (no BN) → C128 → C256 → ZeroPad → C512(s=1, no bias) → BN → LReLU
    → ZeroPad → C1(s=1)  → logits (B, 1, 30, 30) for 256×256 images.
    """
    def __init__(self, in_channels: int = 1, cond_channels: int = 1, kernel_size: int = 4):
        super().__init__()
        k = kernel_size
        p = _same_pad_stride2_for_k4(k)
        ch_in = in_channels + cond_channels

        self.net = nn.Sequential(
            # down1: no BN
            # nn.Conv2d(ch_in, 64, kernel_size=k, stride=2, padding=p, bias=True),
            nn.Conv2d(ch_in, 64, kernel_size=k, stride=2, padding=p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # down2
            nn.Conv2d(64, 128, kernel_size=k, stride=2, padding=p, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # down3
            nn.Conv2d(128, 256, kernel_size=k, stride=2, padding=p, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # zero pad → conv s=1
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, kernel_size=k, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, kernel_size=k, stride=1, padding=0, bias=True),
        )
        self.apply(_init_weights)

        self.hparams: Dict[str, Any] = dict(
            in_channels=in_channels, cond_channels=cond_channels, kernel_size=kernel_size
        )

    def forward(self, inp: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        x = torch.cat([inp, tar], dim=1)
        return self.net(x)  # logits

    # -------------------- Persistence APIs --------------------

    def save_model(self, save_dir: str, model_name: str = "model") -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}.pth")
        payload = {
            "class": self.__class__.__name__,
            "hparams": self.hparams,
            "state_dict": self.state_dict(),
            "pytorch_version": torch.__version__,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load_model(
        cls,
        filepath: str,
        device: Optional[Union[str, torch.device]] = None,
        eval_mode: bool = True,
    ) -> "PatchDiscriminator":
        map_location = torch.device(device) if isinstance(device, str) else device
        payload = torch.load(filepath, map_location=map_location)
        h = payload["hparams"]
        model = cls(**h)
        model.load_state_dict(payload["state_dict"], strict=True)
        if map_location is not None:
            model.to(map_location)
        if eval_mode:
            model.eval()
        return model

# ------------------------- Losses (TF-Equivalent) -------------------------

class Pix2PixLosses(nn.Module):
    """BCEWithLogits + λ * L1 (target - gen_output), like the TF code."""
    def __init__(self, lambda_l1: float = 100.0):
        super().__init__()
        self.adv = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = float(lambda_l1)

    def generator_loss(
        self, disc_generated_logits: torch.Tensor, gen_output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gan_loss = self.adv(disc_generated_logits, torch.ones_like(disc_generated_logits))
        l1_loss = self.l1(gen_output, target)
        total = gan_loss + self.lambda_l1 * l1_loss
        return total, gan_loss, l1_loss

    def discriminator_loss(
        self, disc_real_logits: torch.Tensor, disc_fake_logits: torch.Tensor
    ) -> torch.Tensor:
        real_loss = self.adv(disc_real_logits, torch.ones_like(disc_real_logits))
        fake_loss = self.adv(disc_fake_logits, torch.zeros_like(disc_fake_logits))
        return real_loss + fake_loss