import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

# ----------------------
# Small utilities
# ----------------------

def window_partition(x, window_size: int):
    """x: (B,H,W,C) -> (B*nW, Ws, Ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x


def window_reverse(windows, window_size: int, H: int, W: int):
    """windows: (B*nW, Ws, Ws, C) -> (B,H,W,C)"""
    B = int(windows.shape[0] // (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ----------------------
# Relative Position Bias (Swin v1, discrete table)
# ----------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        # (2*Ws-1, 2*Ws-1, H)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Ws, Ws
        coords_flatten = torch.flatten(coords, 1)  # 2, Ws*Ws
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, N: int):
        # return bias: (num_heads, N, N)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # N,N,H
        return bias.permute(2, 0, 1).contiguous()


# ----------------------
# Window Attention (scaled dot-product) + optional mask for SW-MSA
# ----------------------
class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, h, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, h, N, N

        attn = attn + self.rpb(N).unsqueeze(0)
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.view(1, nW, 1, N, N)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


# ----------------------
# Swin Transformer Block (W-MSA / SW-MSA)
# ----------------------
class SwinBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        H, W = input_resolution
        self.dim = dim
        self.H, self.W = H, W
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

        # precompute mask for SW-MSA
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._create_mask(H, W, window_size, shift_size), persistent=False)
        else:
            self.attn_mask = None

    @staticmethod
    def _create_mask(H: int, W: int, window_size: int, shift_size: int):
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # window
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, attn_mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------
# Patch embed / merging / expand
# ----------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B,C,H/ps,W/ps
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return self.norm(x), (H, W)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.H, self.W = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2C)
        return x, (H // 2, W // 2)


class PatchExpand(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.H, self.W = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        x = self.expand(x)  # B, H*W, 2C
        x = x.view(B, H, W, 2 * C).permute(0, 3, 1, 2).contiguous()
        x = F.pixel_shuffle(x, 2)  # B, C/2, 2H, 2W
        B, C2, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H2 * W2, C2)
        return self.norm(x), (H2, W2)


# ----------------------
# Basic encoder/decoder layers
# ----------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, downsample=True):
        super().__init__()
        H, W = input_resolution
        self.blocks = nn.ModuleList([
            SwinBlock(dim, (H, W), num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        self.downsample = PatchMerging((H, W), dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        size = self.blocks[0].H, self.blocks[0].W
        if self.downsample is not None:
            x, size = self.downsample(x)
        return x, size


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, upsample=True):
        super().__init__()
        H, W = input_resolution
        self.blocks = nn.ModuleList([
            SwinBlock(dim, (H, W), num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        self.upsample = PatchExpand((H, W), dim) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        size = self.blocks[0].H, self.blocks[0].W
        if self.upsample is not None:
            x, size = self.upsample(x)
        return x, size


# ----------------------
# Swin-UNet (original-style, minimal)
# ----------------------
class SwinUNet(nn.Module):
    def __init__(self,
                 img_size=224, in_chans=1, out_chans=1,
                 embed_dim=96, patch_size=4, window_size=7,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24)):
        super().__init__()
        H = W = img_size // patch_size
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)

        # Encoder
        self.layer1 = BasicLayer(embed_dim, (H, W), depths[0], num_heads[0], window_size, downsample=True)
        self.layer2 = BasicLayer(embed_dim * 2, (H // 2, W // 2), depths[1], num_heads[1], window_size, downsample=True)
        self.layer3 = BasicLayer(embed_dim * 4, (H // 4, W // 4), depths[2], num_heads[2], window_size, downsample=True)
        self.layer4 = BasicLayer(embed_dim * 8, (H // 8, W // 8), depths[3], num_heads[3], window_size, downsample=False)

        # Decoder + skip fusions (Linear to reduce channels after concat)
        self.up3 = BasicLayerUp(embed_dim * 8, (H // 8, W // 8), depth=1, num_heads=num_heads[3], window_size=window_size, upsample=True)
        self.fuse3 = nn.Linear(embed_dim * 4 + embed_dim * 4, embed_dim * 4)

        self.up2 = BasicLayerUp(embed_dim * 4, (H // 4, W // 4), depth=1, num_heads=num_heads[2], window_size=window_size, upsample=True)
        self.fuse2 = nn.Linear(embed_dim * 2 + embed_dim * 2, embed_dim * 2)

        self.up1 = BasicLayerUp(embed_dim * 2, (H // 2, W // 2), depth=1, num_heads=num_heads[1], window_size=window_size, upsample=True)
        self.fuse1 = nn.Linear(embed_dim + embed_dim, embed_dim)

        # Reconstruction head
        self.norm_out = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, out_chans, kernel_size=1)

        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, in_chans, H, W)
        B = x.size(0)
        x0, (H, W) = self.patch_embed(x)              # (B, H*W, C)
        x1, s1 = self.layer1(x0)                      # (B, H/2*W/2, 2C)
        x2, s2 = self.layer2(x1)                      # (B, H/4*W/4, 4C)
        x3, s3 = self.layer3(x2)                      # (B, H/8*W/8, 8C)
        xb, sb = self.layer4(x3)                      # bottleneck (no downsample)

        y3, _ = self.up3(xb)                          # -> H/4,W/4,C=4C
        y3 = self.fuse3(torch.cat([y3, x2], dim=-1))

        y2, _ = self.up2(y3)                          # -> H/2,W/2,C=2C
        y2 = self.fuse2(torch.cat([y2, x1], dim=-1))

        y1, _ = self.up1(y2)                          # -> H,W,C=C
        y1 = self.fuse1(torch.cat([y1, x0], dim=-1))  # (B, H*W, C)

        # tokens -> logits map at patch resolution, then upsample back to img_size
        y = self.norm_out(y1).view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B,C,H/ps,W/ps
        y = F.interpolate(y, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        y = self.proj_out(y)  # (B, num_classes, img_size, img_size)
        return y


# --- SSIM (simplified single-scale) ---
class SSIM(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1, self.C2 = C1, C2

        gauss = torch.tensor([
            math.exp(-((x - (window_size - 1) / 2) ** 2) / (2 * 1.5 ** 2))
            for x in range(window_size)
        ])
        gauss = (gauss / gauss.sum()).float()
        window_1d = gauss.view(1, 1, -1)
        window_2d = (window_1d.transpose(2, 1) @ window_1d).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window_2d)

    def _filter(self, x):
        C = x.size(1)
        w = self.window.expand(C, 1, self.window_size, self.window_size)
        return F.conv2d(x, w, padding=self.window_size // 2, groups=C)

    def forward(self, x, y):
        mu_x = self._filter(x); mu_y = self._filter(y)
        sigma_x = self._filter(x * x) - mu_x ** 2
        sigma_y = self._filter(y * y) - mu_y ** 2
        sigma_xy = self._filter(x * y) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)) / (
               (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2))
        return 1 - ssim.mean()  # as a loss

# --- Composite L1 + SSIM loss ---
class ReconLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(window_size=11)
        self.w_l1 = w_l1
        self.w_ssim = w_ssim

    def forward(self, pred, target):
        return self.w_l1 * self.l1(pred, target) + self.w_ssim * self.ssim(pred, target)


def build_swin_unet_tiny(img_size=224, in_chans=1, out_chans=1):
    return SwinUNet(img_size=img_size, in_chans=in_chans, out_chans=out_chans,
                    embed_dim=96, depths=(2,2,2,2), num_heads=(3,6,12,24),
                    window_size=7, patch_size=4)
