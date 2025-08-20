# Minimal Swin-V2-style 4-stage encoder + conv decoder (256x256 → 256x256)
# - Fixes: SW-MSA mask, pre-norm, PatchMerging+LN, DropPath, asserts
# - Downs: 256→64 (patch 4), then merges: 64→32→16→8 (4 stages total)
# - Ups: 8→16→32→64→128→256

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------
def window_partition(x, window_size):
    # x: [B, H, W, C] -> [B*nW, w, w, C]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, nH, nW, w, w, C]
    windows = x.reshape(-1, window_size, window_size, C)
    return windows

def window_unpartition(windows, window_size, H, W):
    # windows: [B*nW, w, w, C] -> [B, H, W, C]
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

# ----------------------------
# DropPath (stochastic depth)
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ----------------------------
# Relative position bias (V1-style table)
                
# ----------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        ws = window_size
        num_rel = (2 * ws - 1) * (2 * ws - 1)
        self.table = nn.Parameter(torch.zeros(num_rel, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing='ij'
        ))  # [2, ws, ws]
        coords_flat = coords.reshape(2, -1)  # [2, ws*ws]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, N, N]
        rel = rel.permute(1, 2, 0)  # [N, N, 2]
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel_index = rel[:, :, 0] * (2 * ws - 1) + rel[:, :, 1]  # [N, N]
        self.register_buffer("relative_index", rel_index, persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self):
        H = self.table.shape[1]
        idx = self.relative_index.view(-1)
        N = self.relative_index.shape[0]
        bias = self.table[idx].view(N, N, H).permute(2, 0, 1)  # [heads, N, N]
        return bias

# ----------------------------
# Cosine attention (Swin-V2 style)
# ----------------------------
class CosineAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # per-head learnable temperature (clamped to avoid instabilities)
        self.tau = nn.Parameter(torch.ones(num_heads) * 0.07)

    def forward(self, x, attn_bias, attn_mask=None):
        # x: [B*nW, N, C]; attn_bias: [H, N, N]; attn_mask: [nW, N, N] or None
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [BnW, N, H, Hd]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.einsum("bnhd,bmhd->bhnm", q, k)  # [BnW, H, N, N]
        tau = self.tau.clamp(min=1e-3).view(1, -1, 1, 1)
        attn = attn / tau
        attn = attn + attn_bias.unsqueeze(0)  # broadcast over batch

        if attn_mask is not None:
            # attn: [B*nW_img, H, N, N]
            nW_img = attn_mask.shape[0]          # windows per image
            BnW, Hh, Nq, Nk = attn.shape
            assert Nq == attn_mask.shape[1] and Nk == attn_mask.shape[2]
            B = BnW // nW_img
            # reshape to [B, nW_img, H, N, N]
            attn = attn.view(B, nW_img, Hh, Nq, Nk)
            # broadcast mask over batch and heads: [1, nW_img, 1, N, N]
            attn = attn + attn_mask.view(1, nW_img, 1, Nq, Nk)
            # back to [B*nW_img, H, N, N]
            attn = attn.view(BnW, Hh, Nq, Nk)


        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(BnW, N, C)
        return self.proj(out)

# ----------------------------
# MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# ----------------------------
# Swin-V2 Block (pre-norm + SW mask)
# ----------------------------
class SwinV2Block(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio, qkv_bias, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        self.norm_attn = nn.LayerNorm(dim)
        self.attn = CosineAttention(dim, num_heads, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.rpb = RelativePositionBias(window_size, num_heads)

    def _attn_mask(self, H, W, device):
        # Build attention mask for shifted windows; None if no shift.
        if self.shift == 0:
            return None
        ws = self.window_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        # three regions along H and W due to cyclic shift
        h_slices = (slice(0, -ws), slice(-ws, -self.shift), slice(-self.shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -self.shift), slice(-self.shift, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        return attn_mask  # [nW, N, N]

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H,W must be divisible by window_size"

        shortcut = x

        # --- Attention (pre-norm) ---
        x = self.norm_attn(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # partition windows
        windows = window_partition(x, ws)  # [BnW, w, w, C]
        windows = windows.view(-1, ws * ws, C)  # [BnW, N, C]

        attn_bias = self.rpb()  # [heads, N, N]
        attn_mask = self._attn_mask(H, W, x.device)
        attn_out = self.attn(windows, attn_bias, attn_mask)  # [BnW, N, C]

        # merge windows and reverse shift
        attn_out = attn_out.view(-1, ws, ws, C)
        x = window_unpartition(attn_out, ws, H, W)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path1(x)

        # --- MLP (pre-norm) ---
        shortcut2 = x
        x = self.norm_mlp(x)
        x = shortcut2 + self.drop_path2(self.mlp(x))

        return x

# ----------------------------
# Patch embedding / merging
# ----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, HW, C_embed]
        x = self.proj(x)  # [B, C_embed, H/ps, W/ps]
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0, "H,W must be even for PatchMerging"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)
        x = self.reduction(self.norm(x))         # [B, (H/2)*(W/2), 2C]
        return x

# ----------------------------
# Encoder (4 stages)
# ----------------------------
class SwinV2Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ps = cfg["patch_size"]
        self.patch_embed = PatchEmbed(cfg["in_chans"], cfg["embed_dim"], ps)
        self.window_size = cfg["window_size"]

        # stage dims and heads
        dims = [cfg["embed_dim"], cfg["embed_dim"] * 2, cfg["embed_dim"] * 4, cfg["embed_dim"] * 8]
        depths = cfg["depths"]         # len=4
        heads = cfg["num_heads"]       # len=4
        drop_path = cfg["drop_path"]

        # create stages
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()  # distributed across blocks
        idx = 0
        for i in range(4):
            stage = nn.ModuleList()
            for d in range(depths[i]):
                shift = 0 if (d % 2 == 0) else self.window_size // 2
                stage.append(
                    SwinV2Block(
                        dim=dims[i],
                        num_heads=heads[i],
                        window_size=self.window_size,
                        shift=shift,
                        mlp_ratio=cfg["mlp_ratio"],
                        qkv_bias=cfg["qkv_bias"],
                        drop_path=dpr[idx],
                    )
                )
                idx += 1
            self.stages.append(stage)
            if i < 3:
                self.mergers.append(PatchMerging(dims[i]))

        self.out_dim = dims[-1]

    def forward(self, x):
        # x: [B, C, 256, 256] (by default)
        B, _, H0, W0 = x.shape
        ps = self.cfg["patch_size"]
        ws = self.cfg["window_size"]

        assert H0 % ps == 0 and W0 % ps == 0, "Image must be divisible by patch_size"
        H = H0 // ps
        W = W0 // ps
        assert H % ws == 0 and W % ws == 0, "H/ps and W/ps must be divisible by window_size"

        x = self.patch_embed(x)  # [B, H*W, C0]

        # Stage 0 (size H,W)
        for blk in self.stages[0]:
            x = blk(x, H, W)

        # Stage 1 (H/2, W/2)
        x = self.mergers[0](x, H, W); H //= 2; W //= 2
        assert H % ws == 0 and W % ws == 0
        for blk in self.stages[1]:
            x = blk(x, H, W)

        # Stage 2 (H/4, W/4)
        x = self.mergers[1](x, H, W); H //= 2; W //= 2
        assert H % ws == 0 and W % ws == 0
        for blk in self.stages[2]:
            x = blk(x, H, W)

        # Stage 3 (H/8, W/8)
        x = self.mergers[2](x, H, W); H //= 2; W //= 2
        assert H % ws == 0 and W % ws == 0
        for blk in self.stages[3]:
            x = blk(x, H, W)

        # reshape tokens to feature map for decoder: [B, C, H, W]
        x = x.view(B, H, W, self.out_dim).permute(0, 3, 1, 2).contiguous()
        # With defaults: ps=4 → 256/4=64; 3 merges → 64→32→16→8 ⇒ [B, C_out, 8, 8]
        return x

# ----------------------------
# Decoder (simple conv upsampler)
# ----------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, channels, final_activation=None):
        super().__init__()
        layers = []
        prev = in_ch
        for ch in channels:
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(prev, ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            prev = ch
        layers += [nn.Conv2d(prev, out_ch, kernel_size=1)]
        self.net = nn.Sequential(*layers)
        self.final_activation = final_activation
        if final_activation is not None:
            if final_activation.lower() == "sigmoid":
                self.act = nn.Sigmoid()
            elif final_activation.lower() == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("final_activation must be None|'sigmoid'|'tanh'")
        else:
            self.act = None

    def forward(self, x):
        x = self.net(x)
        if self.act is not None:
            x = self.act(x)
        return x

# ----------------------------
# Full autoencoder
# ----------------------------
class SwinV2AutoEncoder(nn.Module):
    def get_bottleneck_features(self, x):
        """Return encoder bottleneck features for input x."""
        return self.encoder(x)

    def print_decoder_weights_stats(self):
        """Print mean and std of decoder weights for inspection."""
        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                
    def __init__(self, 
                 img_size=256,
                 in_chans=1, 
                 out_chans=1,
                 patch_size=4,
                 embed_dim=64,
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=8,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_path=0.1,
                 decoder_channels=[512, 256, 128, 64, 32],
                 final_activation=None):
        super().__init__()
        # Config
        cfg = {
            "img_size": img_size,
            "in_chans": in_chans,
            "out_chans": out_chans,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depths": depths,
            "num_heads": num_heads,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "drop_path": drop_path,
        }
        self.encoder = SwinV2Encoder(cfg)
        self.decoder = SimpleDecoder(
            in_ch=self.encoder.out_dim,
            out_ch=out_chans,
            channels=decoder_channels,
            final_activation=final_activation,
        )

    def forward(self, x):
        feats = self.encoder(x)   # e.g. [B, 8*embed_dim, 8, 8] with defaults
        out = self.decoder(feats) # [B, out_chans, 256, 256]
        return out

# ----------------------------
# Model builder
# ----------------------------
def build_model(config):
    return SwinV2AutoEncoder(
        img_size=config.get("img_size", 256),
        in_chans=config.get("in_chans", 1),
        out_chans=config.get("out_chans", 1),
        patch_size=config.get("patch_size", 4),
        embed_dim=config.get("embed_dim", 64),
        depths=config.get("depths", [2, 2, 6, 2]),
        num_heads=config.get("num_heads", [2, 4, 8, 16]),
        window_size=config.get("window_size", 8),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        qkv_bias=config.get("qkv_bias", True),
        drop_path=config.get("drop_path", 0.1),
        decoder_channels=config.get("decoder_channels", [512, 256, 128, 64, 32]),
        final_activation=config.get("final_activation", None),
    )
