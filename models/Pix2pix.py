import torch
import torch.nn as nn

# -------------------------
# Helpers / weight init
# -------------------------
def _init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -------------------------
# Building blocks
# -------------------------
class Downsample(nn.Module):
    # Conv2d(k=4, s=2, padding='same'), optional BN, LeakyReLU
    def __init__(self, in_ch: int, out_ch: int, k: int = 4, apply_batchnorm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01) if apply_batchnorm else None
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)  # TF default alpha=0.3

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

class Upsample(nn.Module):
    # ConvTranspose2d(k=4, s=2, padding='same'), BN, optional Dropout(0.5), ReLU
    def __init__(self, in_ch: int, out_ch: int, k: int = 4, apply_dropout: bool = False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.drop = nn.Dropout(p=0.5) if apply_dropout else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.drop is not None:
            x = self.drop(x)
        return self.act(x)

# -------------------------
# Generator (U-Net-like)
# -------------------------
class Generator(nn.Module):
    """
    Inputs:  (N, C, 256, 256)
    Outputs: (N, C, 256, 256) with tanh activation
    """
    def __init__(self, channels: int):
        super().__init__()
        C = channels

        # Down path (same filters/order as TF)
        self.down_stack = nn.ModuleList([
            Downsample(C,    64,  apply_batchnorm=False),  # 256 -> 128
            Downsample(64,   128),                         # 128 -> 64
            Downsample(128,  128),                         # 64  -> 32
            Downsample(128,  256),                         # 32  -> 16
            Downsample(256,  512),                         # 16  -> 8
            Downsample(512,  512),                         # 8   -> 4  (bottleneck)
        ])

        # Up path (same filters/order as TF up_stack)
        # Note: due to skip concatenations, in_ch values reflect (prev out + skip channels).
        self.up_stack = nn.ModuleList([
            Upsample(512,        512, apply_dropout=True),   # 4  -> 8   ; cat with 512 -> 1024
            Upsample(512 + 512,  512),                       # 8  -> 16  ; cat with 256 -> 768
            Upsample(512 + 256,  256),                       # 16 -> 32  ; cat with 128 -> 384
            Upsample(256 + 128,  128),                       # 32 -> 64  ; cat with 128 -> 256
            Upsample(128 + 128,  128),                       # 64 -> 128 ; cat with 64  -> 192
            Upsample(128 + 64,   64),                        # (defined but intentionally UNUSED to mirror TF zip)
        ])

        # Final layer: ConvTranspose2d(CH=192 -> CH=channels), tanh
        self.last = nn.Sequential(
            nn.ConvTranspose2d(192, channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

        self.apply(_init_weights)

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        # Remove bottleneck from skips and reverse for up path
        skips = skips[:-1][::-1]  # [512, 256, 128, 128, 64] with spatial sizes [8,16,32,64,128]

        # Mirror TF behavior: iterate over zip(up_stack, skips) → uses only first 5 up blocks
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        # Final up to 256x256
        return self.last(x)

# -------------------------
# Discriminator (PatchGAN)
# -------------------------
class Discriminator(nn.Module):
    """
    Inputs:  inp, tar  (N, C, 256, 256)
    Output:          (N, 1, 30, 30) logits map (no sigmoid)
    """
    def __init__(self, channels: int):
        super().__init__()
        C = channels
        in_ch = C * 2  # concatenate [inp, tar] along channels

        self.down1 = Downsample(in_ch, 64,  apply_batchnorm=False)  # 256 -> 128
        self.down2 = Downsample(64,  128)                           # 128 -> 64
        self.down3 = Downsample(128, 256)                           # 64  -> 32

        self.pad1 = nn.ZeroPad2d((1, 1, 1, 1))                      # 32 -> 34
        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False)  # 34 -> 31
        self.bn   = nn.BatchNorm2d(512, eps=1e-3, momentum=0.01)
        self.act  = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.pad2 = nn.ZeroPad2d((1, 1, 1, 1))                      # 31 -> 33
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=True)     # 33 -> 30

        self.apply(_init_weights)

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)  # (N, 2C, 256, 256)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.pad1(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pad2(x)
        return self.last(x)

# -------------------------
# (Optional) Losses to mirror your TF code exactly
# -------------------------
LAMBDA = 100
_bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = _bce_logits(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss  = torch.mean(torch.abs(target - gen_output))
    total    = gan_loss + (LAMBDA * l1_loss)
    return total, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = _bce_logits(disc_real_output, torch.ones_like(disc_real_output))
    gen_loss  = _bce_logits(disc_generated_output, torch.zeros_like(disc_generated_output))
    return real_loss + gen_loss


class Pix2PixLosses:
    def __init__(self, lambda_l1: float = 100.0):
        self.lambda_l1 = float(lambda_l1)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan = self.bce(disc_generated_output, torch.ones_like(disc_generated_output))
        l1  = torch.mean(torch.abs(target - gen_output))
        total = gan + self.lambda_l1 * l1
        return total, gan, l1

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real = self.bce(disc_real_output, torch.ones_like(disc_real_output))
        fake = self.bce(disc_generated_output, torch.zeros_like(disc_generated_output))
        return real + fake