# swin_unet.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- Utils ----------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
    Returns:
        windows: (B*nW, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: (B*nW, window_size, window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------------- Window Attention -----------------------

class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias."""
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads)
        )

        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B*nW, N, C)  where N = window_size*window_size
        mask: (nW, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)

        # relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1],
            self.window_size[0]*self.window_size[1],
            -1
        )  # N,N,heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # 1, heads, N, N
        attn = attn + relative_position_bias

        if mask is not None:
            # mask: nW, N, N
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # broadcast
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ------------------- Swin Transformer Block --------------------

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4., drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        H, W = input_resolution
        self.dim = dim
        self.input_resolution = (H, W)
        self.num_heads = num_heads
        self.window_size = min(window_size, H, W)
        self.shift_size = 0 if min(H, W) <= self.window_size else shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (self.window_size, self.window_size), num_heads,
                                    qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

        # attention mask for SW-MSA (precompute)
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self.calculate_mask(H, W))
        else:
            self.attn_mask = None

    def calculate_mask(self, H: int, W: int) -> torch.Tensor:
        # compute attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        cnt = 0
        ws = self.window_size
        ss = self.shift_size
        for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws)  # nW, ws, ws, 1
        mask_windows = mask_windows.view(-1, ws*ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, N, N
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows and apply attention
        x_windows = window_partition(x, self.window_size)  # (B*nW, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (B*nW, N, C)

        # merge windows
        x = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)  # (B, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------- Patch Embed / Merging --------------------

class PatchEmbed(nn.Module):
    """Patch Partition + Linear Embedding via strided conv."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B, HW, C)
        x = self.norm(x)
        return x, (H, W)

class PatchMerging(nn.Module):
    """Downsample: concat 2x2 neighbors then linear proj to 2*C."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        # 2x2 gather
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        H, W = H // 2, W // 2
        x = x.view(B, H * W, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H*W, 2C)
        return x, (H, W)

class PatchExpand(nn.Module):
    """Upsample by factor=2 using linear -> pixel-shuffle style rearrangement.
       Input (B, H*W, C) -> Output (B, (2H)*(2W), C_out) with C_out = C//2."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.proj = nn.Linear(dim, 2 * dim)  # -> 2C = (2*2*C_out) when C_out = C//2
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and C == self.dim
        x = self.proj(x)  # (B, HW, 2C)
        x = x.view(B, H, W, 2 * C)

        # rearrange to (2H, 2W, C//2)
        C_out = C // 2
        x = x.view(B, H, W, 2, 2, C_out).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C_out)
        H, W = H * 2, W * 2
        x = x.view(B, H * W, C_out)
        x = self.norm(x)
        return x, (H, W)

class PatchExpandX4(nn.Module):
    """Final upsample x4 to reach input size."""
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # Produce (4*4*C_out) with C_out = dim // 4
        self.proj = nn.Linear(dim, 16 * (dim // 4))
        self.norm = nn.LayerNorm(dim // 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and C == self.dim
        x = self.proj(x)  # (B, HW, 16*C_out)
        C_out = C // 4
        x = x.view(B, H, W, 4, 4, C_out).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 4, W * 4, C_out)
        H, W = H * 4, W * 4
        x = x.view(B, H * W, C_out)
        x = self.norm(x)
        return x, (H, W)

# ------------------------ Stages (Layers) ----------------------

class BasicLayer(nn.Module):
    """Encoder stage: Swin blocks * depth, then optional PatchMerging."""
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4., downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(dim, input_resolution, num_heads,
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio)
            )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor]:
        # return x_out, reso_out, skip_before_down
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x, reso = self.downsample(x)
        else:
            B, L, C = x.shape
            reso = (int((L) ** 0.5), int((L) ** 0.5))
        return x, reso, skip

class UpLayer(nn.Module):
    """Decoder stage: PatchExpand -> (skip add) -> Swin blocks * depth."""
    def __init__(self, dim_in: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4.):
        super().__init__()
        self.expand = PatchExpand(input_resolution, dim_in)  # dim_out = dim_in//2
        dim_out = dim_in // 2
        out_reso = (input_resolution[0] * 2, input_resolution[1] * 2)

        # after skip-add, run blocks at dim_out
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(dim_out, out_reso, num_heads,
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio)
            )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x, reso = self.expand(x)  # now (B, (2H)*(2W), C//2)
        # skip and x are same dim & resolution → elementwise add
        x = x + skip
        for blk in self.blocks:
            x = blk(x)
        return x, reso

# ---------------------------- Swin-UNet ------------------------

class SwinUNet(nn.Module):
    """
    Minimal Swin-UNet for (C_in, 224, 224) -> (C_out, 224, 224).
    Set `in_chans` (e.g., 1 for grayscale) and `out_chans` as needed.
    """
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,         # <-- set to 1 for grayscale input
        out_chans: int = 3,        # <-- set as you like
        embed_dim: int = 96,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert img_size % (patch_size * 16) == 0 or img_size == 224, "Fixed 224x224 in this minimal version."

        self.config = dict(
            img_size=img_size, in_chans=in_chans, out_chans=out_chans, embed_dim=embed_dim,
            depths=depths, num_heads=num_heads, window_size=window_size, patch_size=patch_size, mlp_ratio=mlp_ratio
        )

        # ---- Encoder ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        reso0 = (img_size // patch_size, img_size // patch_size)  # 56x56

        self.layer1 = BasicLayer(embed_dim, reso0, depths[0], num_heads[0], window_size, mlp_ratio, downsample=True)
        reso1 = (reso0[0] // 2, reso0[1] // 2)  # 28x28
        self.layer2 = BasicLayer(embed_dim * 2, reso1, depths[1], num_heads[1], window_size, mlp_ratio, downsample=True)
        reso2 = (reso1[0] // 2, reso1[1] // 2)  # 14x14
        self.layer3 = BasicLayer(embed_dim * 4, reso2, depths[2], num_heads[2], window_size, mlp_ratio, downsample=True)
        reso3 = (reso2[0] // 2, reso2[1] // 2)  # 7x7
        self.layer4 = BasicLayer(embed_dim * 8, reso3, depths[3], num_heads[3], window_size, mlp_ratio, downsample=False)  # bottleneck

        # ---- Decoder ----
        self.up3 = UpLayer(embed_dim * 8, reso3, depths[2], num_heads[2], window_size, mlp_ratio)  # 7->14
        self.up2 = UpLayer(embed_dim * 4, reso2, depths[1], num_heads[1], window_size, mlp_ratio)  # 14->28
        self.up1 = UpLayer(embed_dim * 2, reso1, depths[0], num_heads[0], window_size, mlp_ratio)  # 28->56

        self.final_up = PatchExpandX4(reso0, embed_dim)  # 56->224
        self.head = nn.Linear(embed_dim // 4, out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.config["img_size"] and W == self.config["img_size"], "Expect fixed 224x224 here."
        assert C == self.config["in_chans"], f"Expected {self.config['in_chans']} input channels, got {C}"

        x, _ = self.patch_embed(x)                 # (B, 56*56, 96)
        x1, _, skip1 = self.layer1(x)              # 28x28x192, skip 56x56x96
        x2, _, skip2 = self.layer2(x1)             # 14x14x384
        x3, _, skip3 = self.layer3(x2)             # 7x7x768
        x4, _, _    = self.layer4(x3)              # 7x7x768

        y3, _ = self.up3(x4, skip3)                # 14x14x384
        y2, _ = self.up2(y3, skip2)                # 28x28x192
        y1, _ = self.up1(y2, skip1)                # 56x56x96

        y, _  = self.final_up(y1)                  # 224x224x(96//4)
        y     = self.head(y)                       # (B, 224*224, out_chans)
        y     = y.transpose(1, 2).view(B, self.config["out_chans"], H, W)
        return y

    # ---- Save / Load stay the same ----
    def save_model(self, save_dir: str, model_name: str = "model"):
        os.makedirs(save_dir, exist_ok=True)
        payload = {"state_dict": self.state_dict(), "config": self.config}
        path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save(payload, path)
        return path

    @classmethod
    def load_model(cls, filepath: str, device: Optional[Union[str, torch.device]] = None):
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        cfg = checkpoint["config"]
        model = cls(**cfg)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model

