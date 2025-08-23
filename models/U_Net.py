# U-Net.py
# Minimal U-Net with optional asymmetric channel widths and optional skip connections.

from typing import Sequence, Optional, Union
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import save_tensor_image_and_exit
    
def _make_act(name: str) -> nn.Module:
    """Factory for hidden activations."""
    name = (name or "relu").lower()
    if name in {"relu"}:
        return nn.ReLU(inplace=True)
    if name in {"leaky_relu", "leakyrelu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    raise ValueError("act must be 'relu' or 'leaky_relu'")

class DoubleConv(nn.Module):
    """
    Two 3x3 conv blocks.
    ### CHANGED: optional BatchNorm and selectable activation.
    """
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = False, act: str = "relu"):
        super().__init__()
        self.use_batchnorm = bool(use_batchnorm)
        self.act_name = act

        layers = []
        # conv -> [bn] -> act
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not self.use_batchnorm)]
        if self.use_batchnorm:
            layers += [nn.BatchNorm2d(out_ch)]
        layers += [_make_act(self.act_name)]

        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not self.use_batchnorm)]
        if self.use_batchnorm:
            layers += [nn.BatchNorm2d(out_ch)]
        layers += [_make_act(self.act_name)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNet(nn.Module):
    """
    Encoder:  [DoubleConv -> MaxPool] * L   (channels = enc_channels[i])
    Bottleneck: DoubleConv(bottleneck_in, bottleneck_channels)
    Decoder:  [Up(ConvTranspose2d) -> (optional concat skip) -> DoubleConv] * L
              up out-ch = dec_channels[i]; DoubleConv maps (dec_i [+ enc_i]) -> dec_i
    Final:    1x1 Conv to out_channels (optional Sigmoid)

    Args:
        in_channels:  input channels (e.g., 1 or 3)
        out_channels: output channels (e.g., 1 for regression/map)
        enc_channels: per-level encoder widths (depth = L)
        dec_channels: per-level decoder widths (must have same length as enc);
                      if None, mirrors enc_channels (symmetric U-Net)
        bottleneck_channels: channels at bottleneck; defaults to 2*enc_channels[-1]
        use_sigmoid: apply Sigmoid at output (useful if labels are in [0,1])
        use_skips: enable/disable skip connections (default True). If False, no concats.
        use_batchnorm: add BatchNorm after each conv in DoubleConv blocks
        act:           'relu' or 'leaky_relu' for all hidden layers
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        enc_channels: Sequence[int] = (64, 128, 256, 512),
        dec_channels: Optional[Sequence[int]] = None,
        bottleneck_channels: Optional[int] = None,
        use_sigmoid: bool = False,
        use_skips: bool = True,
        use_batchnorm: bool = False,
        act: str = "relu",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.enc_channels = [int(c) for c in enc_channels]
        self.dec_channels = [int(c) for c in (dec_channels or enc_channels)]
        if len(self.enc_channels) != len(self.dec_channels):
            raise ValueError("enc_channels and dec_channels must have the same length (same # of resolution steps).")
        self.use_skips = bool(use_skips)
        self.use_batchnorm = bool(use_batchnorm)
        self.act = act

        depth = len(self.enc_channels)
        self.depth = depth

        # Encoder
        downs, pools = [], []
        prev_c = self.in_channels
        for f in self.enc_channels:
            downs.append(DoubleConv(prev_c, f, use_batchnorm=self.use_batchnorm, act=self.act))
            pools.append(nn.MaxPool2d(2, 2))
            prev_c = f
        self.downs = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)

        # Bottleneck
        bottleneck_c = int(bottleneck_channels) if bottleneck_channels is not None else prev_c * 2
        self._bottleneck_channels = bottleneck_c
        self.bottleneck = DoubleConv(prev_c, bottleneck_c, use_batchnorm=self.use_batchnorm, act=self.act)

        # Decoder (build in decode order: top level first)
        ups, dec_convs = [], []
        curr_c = bottleneck_c
        for i in reversed(range(depth)):
            out_c = self.dec_channels[i]
            ups.append(nn.ConvTranspose2d(curr_c, out_c, kernel_size=2, stride=2))
            in_to_dec = out_c + self.enc_channels[i] if self.use_skips else out_c
            dec_convs.append(DoubleConv(in_to_dec, out_c, use_batchnorm=self.use_batchnorm, act=self.act))
            curr_c = out_c
        self.ups = nn.ModuleList(ups)
        self.dec_convs = nn.ModuleList(dec_convs)

        # Final 1x1 conv + optional activation
        self.final_conv = nn.Conv2d(curr_c, self.out_channels, kernel_size=1)
        self.output_act = nn.Sigmoid() if use_sigmoid else nn.Identity()
        self.use_sigmoid = bool(use_sigmoid)

        # === INIT: run robust, activation-aware initialization ===
        self._init_weights()

    # ----------------------------- INIT HELPERS -----------------------------
    def _act_for_init(self):
        """Map act string to init kwargs."""
        name = (self.act or "relu").lower()
        if name in {"relu"}:
            return dict(nonlinearity="relu", a=0.0)
        if name in {"leaky_relu", "leakyrelu", "lrelu"}:
            return dict(nonlinearity="leaky_relu", a=0.01)
        # fallback
        return dict(nonlinearity="relu", a=0.0)

    @staticmethod
    def _make_bilinear_kernel(size: int) -> torch.Tensor:
        """2D bilinear upsampling kernel."""
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = torch.arange(size, dtype=torch.float32)
        filt = (1 - torch.abs(og - center) / factor)
        kernel = filt[:, None] * filt[None, :]
        return kernel

    def _init_weights(self):
        """### INIT: robust initialization for all submodules."""
        init_kwargs = self._act_for_init()
        nonlin = init_kwargs["nonlinearity"]
        a = init_kwargs["a"]

        with torch.no_grad():
            for m in self.modules():
                # Conv2d
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlin, a=a)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                # ConvTranspose2d (upsampling)
                elif isinstance(m, nn.ConvTranspose2d):
                    k_ok = (m.kernel_size[0] == m.kernel_size[1] == 2)
                    s_ok = (m.stride[0] == m.stride[1] == 2)
                    same_channels = (m.in_channels == m.out_channels)
                    no_groups = (m.groups == 1)
                    if k_ok and s_ok and same_channels and no_groups:
                        # Bilinear upsampling init
                        kernel = self._make_bilinear_kernel(2).to(m.weight.device)
                        m.weight.zero_()
                        for c in range(m.in_channels):
                            m.weight[c, c, :, :] = kernel
                    else:
                        # Fallback to Kaiming
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlin, a=a)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                # BatchNorm
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)   # gamma = 1
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)    # beta = 0

            # Final layer: Xavier for stable small outputs
            nn.init.xavier_uniform_(self.final_conv.weight, gain=1.0)
            if self.final_conv.bias is not None:
                nn.init.zeros_(self.final_conv.bias)

    def reinit_weights(self):
        """Public helper to re-apply the robust init if needed."""
        self._init_weights()

    # ----------------------------- FORWARD -----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- DEBUG: save and stop ---
        # save_tensor_image_and_exit(x)
        # ----------------------------
    
        # Encoder with optional skip collection
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            if self.use_skips:
                skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        if self.use_skips:
            for up, dec, skip in zip(self.ups, self.dec_convs, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([skip, x], dim=1)
                x = dec(x)
        else:
            for up, dec in zip(self.ups, self.dec_convs):
                x = up(x)
                x = dec(x)

        x = self.final_conv(x)
        return self.output_act(x)

    # ----------------------------- I/O helpers -----------------------------
    def save_model(self, save_dir: str, model_name: str = "model") -> str:
        """Save weights + minimal config for reloading."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}.pt")
        ckpt = {
            "state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "enc_channels": self.enc_channels,
            "dec_channels": self.dec_channels,
            "bottleneck_channels": self._bottleneck_channels,
            "use_sigmoid": self.use_sigmoid,
            "use_skips": self.use_skips,
            "use_batchnorm": self.use_batchnorm,
            "act": self.act,
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load_model(
        cls,
        filepath: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "UNet":
        """Reload a model saved by `save_model`."""
        map_location = device if device is not None else "cpu"
        ckpt = torch.load(filepath, map_location=map_location)

        # Backward-compat for older symmetric checkpoints with `features`
        if "enc_channels" not in ckpt and "features" in ckpt:
            ckpt["enc_channels"] = list(ckpt["features"])
            ckpt["dec_channels"] = list(ckpt["features"])
            if "bottleneck_channels" not in ckpt:
                ckpt["bottleneck_channels"] = ckpt["features"][-1] * 2

        model = cls(
            in_channels=int(ckpt["in_channels"]),
            out_channels=int(ckpt["out_channels"]),
            enc_channels=ckpt["enc_channels"],
            dec_channels=ckpt.get("dec_channels", ckpt["enc_channels"]),
            bottleneck_channels=int(ckpt.get("bottleneck_channels", ckpt["enc_channels"][-1] * 2)),
            use_sigmoid=bool(ckpt.get("use_sigmoid", False)),
            use_skips=bool(ckpt.get("use_skips", True)),
            use_batchnorm=bool(ckpt.get("use_batchnorm", False)),
            act=str(ckpt.get("act", "relu")),
        )
        model.load_state_dict(ckpt["state_dict"])
        if device is not None:
            model.to(device)
        return model
