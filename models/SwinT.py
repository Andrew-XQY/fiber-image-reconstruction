# Minimal Swin-U-Net (Swin V2-style) for 1x256x256 -> 1x256x256
# Key V2 bits included:
#  - Scaled cosine attention (L2-normalized Q,K with learnable tau)
#  - Continuous relative position bias via tiny MLP on (log-spaced) relative coords
#  - Window attention + shifted-window attention (cyclic shift + mask)
#  - Patch Merging (down) and Patch Expand (up) to build a U-Net-like encoder/decoder
#
# Notes:
#  - For simplicity, I use pre-norm inside blocks (classic Transformer style).
#    Swin V2 introduced residual-post-norm for very large scale models; you can
#    swap to post-norm if you need exact parity with the paper’s stability trick.
#  - The CRPB here is a compact “continuous” bias MLP that takes
#    sign(x)*log(1+|x|) coords and predicts per-head biases.
#  - Depths and heads are small to keep code readable; scale as needed.
#  - Input/Output are 1×256×256; change patch_size / depths carefully if you alter sizes.

import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities: window partition/reverse
# ---------------------------

def window_partition(x, window_size: int):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # -> (B, nWh, win, nWw, win, C) -> (B*nWh*nWw, win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] // (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------
# Continuous Relative Position Bias (tiny MLP)
# ---------------------------

class CRPBias(nn.Module):
    """
    Continuous relative position bias (simplified):
    - Build relative coords for a window (win x win).
    - Map (dy, dx) -> sign*log(1+|.|) -> MLP -> per-head bias.
    Output shape per window: (num_heads, win*win, win*win)
    """
    def __init__(self, window_size: int, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        # Precompute relative coordinates for a window
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))  # (2, win, win)
        coords_flat = coords.reshape(2, -1)  # (2, win*win)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, win*win, win*win) (dy, dx)
        # log-spaced transform as in SwinV2 spirit: sign(x) * log(1 + |x|)
        rel = rel.float()
        rel = torch.sign(rel) * torch.log1p(torch.abs(rel))
        # store as buffer (2, N, N)
        self.register_buffer("rel_coords", rel)  # no grad

        in_dim = 2  # (dy, dx)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads)
        )

    def forward(self):
        # rel_coords: (2, N, N) -> (N*N, 2) via stack then mlp then reshape
        # We'll map each (dy,dx) pair to a bias per head.
        # Efficient path: flatten pairs, run MLP, reshape back.
        two, N, _ = self.rel_coords.shape  # N = win*win
        rel = torch.stack([self.rel_coords[0], self.rel_coords[1]], dim=-1)  # (N, N, 2)
        rel = rel.view(-1, 2)  # (N*N, 2)
        bias = self.mlp(rel)  # (N*N, num_heads)
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()  # (heads, N, N)
        return bias  # add to attention logits per head


# ---------------------------
# Scaled Cosine Attention with (shifted) windowing + mask
# ---------------------------

class WindowAttentionV2(nn.Module):
    """
    V2-style attention in a window:
      - Q,K L2-normalized -> cosine similarity
      - multiply by learnable tau (per head)
      - add continuous relative position bias
      - optional attention mask for SW-MSA
    Input:  x_windows: (num_windows*B, win*win, C)
    Output: (num_windows*B, win*win, C)
    """
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # learnable per-head temperature (tau in paper)
        self.tau = nn.Parameter(torch.ones(num_heads))

        self.proj = nn.Linear(dim, dim)
        self.crpb = CRPBias(window_size, num_heads)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """
        x: (B_w, N, C)  where N=win*win
        attn_mask: (num_windows, N, N) or None
        """
        B_w, N, C = x.shape
        qkv = self.qkv(x)  # (B_w, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B_w, N, C)

        # split heads
        q = q.view(B_w, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B_w, h, N, d)
        k = k.view(B_w, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B_w, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # L2 normalize along feature dim for cosine sim
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # cosine attention logits: (B_w, h, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1))  # cosine in [-1,1]
        attn = attn * self.tau.view(1, -1, 1, 1)     # learnable scaling per head

        # add continuous relative position bias (h, N, N)
        bias = self.crpb().to(attn.dtype)            # same N for a given window_size
        attn = attn + bias.view(1, self.num_heads, N, N)

        if attn_mask is not None:
            # attn_mask: (num_windows, N, N)  -> expand across heads & batch of windows
            nW = attn_mask.shape[0]
            attn = attn.view(B_w // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.view(1, nW, 1, N, N)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B_w, h, N, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B_w, N, C)
        out = self.proj(out)
        return out


# ---------------------------
# Swin V2 Block (W-MSA or SW-MSA)
# ---------------------------

class SwinBlockV2(nn.Module):
    """
    One Swin block:
      - (optional) cyclic shift
      - window partition -> WindowAttentionV2 -> window reverse
      - residual + MLP residual
    """
    def __init__(self, dim: int, input_resolution: Tuple[int, int],
                 num_heads: int, window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        H, W = input_resolution
        self.dim = dim
        self.H = H
        self.W = W
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

        # precompute attention mask for shifted windows (fixed H,W for this stage)
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._create_attn_mask(H, W, window_size, shift_size), persistent=False)
        else:
            self.attn_mask = None

    @staticmethod
    def _create_attn_mask(H: int, W: int, window_size: int, shift_size: int):
        # Build an attention mask that prevents tokens from attending across
        # different originally-partitioned windows after cyclic shift.
        img_mask = torch.zeros((1, H, W, 1))  # (1,H,W,1)
        cnt = 0
        for h in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
            for w in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # (nW, win, win, 1)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask  # (num_windows, N, N)

    def forward(self, x):
        """
        x: (B, H*W, C) for a fixed stage resolution
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift if needed
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = window_partition(x, self.window_size)  # (B*nW, win, win, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B*nW, N, C)

        # window attention
        attn_windows = self.attn(x_windows, attn_mask=self.attn_mask)  # (B*nW, N, C)

        # reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # (B,H,W,C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x  # residual 1

        # MLP
        x = x + self.mlp(self.norm2(x))  # residual 2
        return x


# ---------------------------
# Patch embedding / merging / expansion
# ---------------------------

class PatchEmbed(nn.Module):
    """Image -> patch tokens"""
    def __init__(self, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B,1,H,W) -> (B, H/ps*W/ps, C)
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = self.norm(x)
        return x, (H, W)


class PatchMerging(nn.Module):
    """Downsample by 2 via concat of 2x2 neighborhood then linear reduce."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        H, W = input_resolution
        self.H, self.W = H, W
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        # x: (B, H*W, C)
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        x = x.view(B, H, W, C)

        # 2x2 concat
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # -> (B, H/2*W/2, 2C)
        return x, (H // 2, W // 2)


class PatchExpand(nn.Module):
    """Upsample by 2 (inverse of PatchMerging) using a linear map + pixel shuffle logic."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        H, W = input_resolution
        self.H, self.W = H, W
        self.expand = nn.Linear(dim, 2 * dim, bias=False)  # prepare for 2x2
        self.norm = nn.LayerNorm(dim // 2)  # after shuffle channels halve

    def forward(self, x):
        # x: (B, H*W, C)
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        x = self.expand(x)  # (B, H*W, 2C)

        # rearrange to spatial upsample by 2
        x = x.view(B, H, W, 2 * C).permute(0, 3, 1, 2).contiguous()  # (B, 2C, H, W)
        x = F.pixel_shuffle(x, upscale_factor=2)  # (B, C/2, 2H, 2W)

        B, C2, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H2 * W2, C2)
        x = self.norm(x)
        return x, (H2, W2)


# ---------------------------
# Basic Swin V2 Stage (stack W-MSA/SW-MSA)
# ---------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, downsample=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList()
        H, W = input_resolution
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinBlockV2(dim, (H, W), num_heads=num_heads, window_size=window_size, shift_size=shift)
            )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        new_size = self.input_resolution
        if self.downsample is not None:
            x, new_size = self.downsample(x)
        return x, new_size


class BasicLayerUp(nn.Module):
    """Decoder mirror: Swin blocks at current res + optional upsample."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, upsample=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList()
        H, W = input_resolution
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinBlockV2(dim, (H, W), num_heads=num_heads, window_size=window_size, shift_size=shift)
            )
        self.upsample = PatchExpand(input_resolution, dim) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        new_size = self.input_resolution
        if self.upsample is not None:
            x, new_size = self.upsample(x)
        return x, new_size


# ---------------------------
# Full model: Swin-UNet-V2-min
# ---------------------------

class SwinUNet(nn.Module):
    def __init__(self,
                 img_size=256, in_chans=1, out_chans=1,
                 embed_dim=96, patch_size=4, window_size=7,
                 depths=(2, 2, 2, 2), heads=(3, 6, 12, 24)):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        H = W = img_size // patch_size

        # Encoder stages
        self.layer1 = BasicLayer(embed_dim, (H, W), depth=depths[0], num_heads=heads[0],
                                 window_size=window_size, downsample=True)
        self.layer2 = BasicLayer(embed_dim * 2, (H // 2, W // 2), depth=depths[1], num_heads=heads[1],
                                 window_size=window_size, downsample=True)
        self.layer3 = BasicLayer(embed_dim * 4, (H // 4, W // 4), depth=depths[2], num_heads=heads[2],
                                 window_size=window_size, downsample=True)
        # bottom (no downsample)
        self.layer4 = BasicLayer(embed_dim * 8, (H // 8, W // 8), depth=depths[3], num_heads=heads[3],
                                 window_size=window_size, downsample=False)

        # Decoder stages (mirror), with skip connections
        self.up3 = BasicLayerUp(embed_dim * 8, (H // 8, W // 8), depth=1, num_heads=heads[3],
                                window_size=window_size, upsample=True)
        self.fuse3 = nn.Linear(embed_dim * 4 + embed_dim * 4, embed_dim * 4)

        self.up2 = BasicLayerUp(embed_dim * 4, (H // 4, W // 4), depth=1, num_heads=heads[2],
                                window_size=window_size, upsample=True)
        self.fuse2 = nn.Linear(embed_dim * 2 + embed_dim * 2, embed_dim * 2)

        self.up1 = BasicLayerUp(embed_dim * 2, (H // 2, W // 2), depth=1, num_heads=heads[1],
                                window_size=window_size, upsample=True)
        self.fuse1 = nn.Linear(embed_dim + embed_dim, embed_dim)

        # head: bring tokens back to image
        self.norm_out = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, out_chans, kernel_size=1)

    def forward(self, x):
        """
        x: (B,1,256,256) -> (B,1,256,256)
        """
        B = x.size(0)
        # Embed
        x, (H, W) = self.patch_embed(x)                     # (B, H*W, C)

        # Encoder + save skips (token domain)
        x1, sz1 = self.layer1(x)                            # -> (B, H/2*W/2, 2C)
        x2, sz2 = self.layer2(x1)                           # -> (B, H/4*W/4, 4C)
        x3, sz3 = self.layer3(x2)                           # -> (B, H/8*W/8, 8C)
        xb, szb = self.layer4(x3)                           # -> (B, H/8*W/8, 8C)

        # Decoder with skips (concat then fuse by Linear)
        y3, s3 = self.up3(xb)                               # up to H/4,W/4 with C=4C
        y3 = torch.cat([y3, x2], dim=-1)
        y3 = self.fuse3(y3)

        y2, s2 = self.up2(y3)                               # up to H/2,W/2 with C=2C
        y2 = torch.cat([y2, x1], dim=-1)
        y2 = self.fuse2(y2)

        y1, s1 = self.up1(y2)                               # up to H,W with C=C
        y1 = torch.cat([y1, x], dim=-1)
        y1 = self.fuse1(y1)                                 # (B, H*W, C)

        # tokens -> image
        y = self.norm_out(y1)
        y = y.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        y = F.interpolate(y, scale_factor=4, mode="nearest")  # restore to 256x256 (undo initial patch=4)
        y = self.proj_out(y)  # (B, 1, 256, 256)
        return y


# ----------------------------- loss function -----------------------------
# --- SSIM (simplified single-scale) ---
class SSIM(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1, self.C2 = C1, C2

        gauss = torch.tensor([torch.exp(-((x - (window_size-1)/2)**2) / (2*1.5**2)) for x in range(window_size)])
        gauss = (gauss / gauss.sum()).float()
        window_1d = gauss.view(1,1,-1)
        window_2d = (window_1d.transpose(2,1) @ window_1d).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window_2d)

    def _filter(self, x):
        C = x.size(1)
        w = self.window.expand(C,1,self.window_size,self.window_size)
        return F.conv2d(x, w, padding=self.window_size//2, groups=C)

    def forward(self, x, y):
        mu_x = self._filter(x); mu_y = self._filter(y)
        sigma_x = self._filter(x*x) - mu_x**2
        sigma_y = self._filter(y*y) - mu_y**2
        sigma_xy = self._filter(x*y) - mu_x*mu_y
        ssim = ((2*mu_x*mu_y + self.C1)*(2*sigma_xy + self.C2)) / ((mu_x**2 + mu_y**2 + self.C1)*(sigma_x + sigma_y + self.C2))
        return 1 - ssim.mean()   # as a loss
        

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