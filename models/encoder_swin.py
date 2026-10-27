"""Self-contained Swin-Transformer U-Net backbone.

Vendored verbatim from downscaling-temperature@1d2c3bb (models/encoder_swin.py): M2
reuses the temperature backbone unchanged so architecture is not a variable
between the two papers.

Windowed multi-head self-attention with shifted windows, hierarchical patch
merging on the encoder and pixel-shuffle patch expansion on the decoder, with
U-Net skip connections. Returns a feature map at patch-embed resolution
(H/patch_size, W/patch_size); the per-variable heads upsample back to full grid.

Input is arbitrary H x W; the patch-embedded map is padded up to a multiple of
window_size * 2**(num_stages-1) internally and cropped on output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def window_reverse(win: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    B = int(win.shape[0] / (H * W / ws / ws))
    x = win.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


# --------------------------------------------------------------------------- #
# Windowed attention with relative position bias
# --------------------------------------------------------------------------- #
class WindowAttention(nn.Module):
    def __init__(self, dim: int, ws: int, num_heads: int):
        super().__init__()
        self.dim, self.ws, self.num_heads = dim, ws, num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.rel_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += ws - 1
        rel[..., 1] += ws - 1
        rel[..., 0] *= 2 * ws - 1
        self.register_buffer("rel_index", rel.sum(-1))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


# --------------------------------------------------------------------------- #
# Swin block (W-MSA / SW-MSA)
# --------------------------------------------------------------------------- #
class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ws: int, shift: int,
                 mlp_ratio: float, drop_path: float):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)

    def _attn_mask(self, H: int, W: int, device) -> torch.Tensor | None:
        if self.shift == 0:
            return None
        img = torch.zeros((1, H, W, 1), device=device)
        hs = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
        ws_ = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
        cnt = 0
        for h in hs:
            for w in ws_:
                img[:, h, w, :] = cnt
                cnt += 1
        mask_win = window_partition(img, self.ws).squeeze(-1)
        attn_mask = mask_win.unsqueeze(1) - mask_win.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.ws)
        win = self.attn(win, self._attn_mask(H, W, x.device))
        x = window_reverse(win, self.ws, H, W)
        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor()


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, ws, mlp_ratio, dprs):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, ws, 0 if i % 2 == 0 else ws // 2,
                      mlp_ratio, dprs[i])
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class PatchMerging(nn.Module):
    """2x downsample: concat 2x2 neighborhood -> Linear(4C -> 2C)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


class PatchExpand(nn.Module):
    """2x upsample via pixel shuffle: Linear(C -> 2C) then rearrange."""

    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x).view(B, H, W, 2 * C)
        x = F.pixel_shuffle(x.permute(0, 3, 1, 2), 2)          # (B, C/2, 2H, 2W)
        x = x.permute(0, 2, 3, 1).reshape(B, 4 * L, C // 2)
        return self.norm(x), H * 2, W * 2


# --------------------------------------------------------------------------- #
# Full backbone
# --------------------------------------------------------------------------- #
class SwinUNet(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int,
                 depths: list[int], num_heads: list[int], window_size: int,
                 mlp_ratio: float = 4.0, drop_path: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.ws = window_size
        self.num_stages = len(depths)
        self.mult = window_size * (2 ** (self.num_stages - 1))
        self.out_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.pos_drop = nn.Dropout(0.0)

        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        # Encoder stages + downsample after all but the last.
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        cur = 0
        for i, d in enumerate(depths):
            self.enc.append(SwinStage(dims[i], d, num_heads[i], window_size,
                                      mlp_ratio, dpr[cur:cur + d]))
            cur += d
            self.down.append(PatchMerging(dims[i]) if i < self.num_stages - 1 else None)

        # Decoder: expand + skip-fuse + stage, mirrored.
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):
            self.up.append(PatchExpand(dims[i]))
            self.fuse.append(nn.Linear(dims[i], dims[i - 1]))       # concat skip -> dims[i-1]
            self.dec.append(SwinStage(dims[i - 1], depths[i - 1], num_heads[i - 1],
                                      window_size, mlp_ratio,
                                      [drop_path] * depths[i - 1]))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                       # (B, C, H', W')
        B, C, H, W = x.shape
        # Pad to a multiple of mult so every stage stays window-aligned.
        pad_h = (self.mult - H % self.mult) % self.mult
        pad_w = (self.mult - W % self.mult) % self.mult
        # Reflect (not zero) padding: zero-pad injects artificial edge values
        # that the windowed attention + decoder smear into a boundary artifact.
        # Reflect keeps the padded border statistically consistent with the
        # interior. Fall back to replicate if the pad exceeds the feature size.
        if pad_h or pad_w:
            mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        H, W = H + pad_h, W + pad_w
        x = x.flatten(2).transpose(1, 2)              # (B, L, C)
        x = self.pos_drop(x)

        skips = []
        res = []
        for i, stage in enumerate(self.enc):
            x = stage(x, H, W)
            skips.append((x, H, W))
            if self.down[i] is not None:
                x, H, W = self.down[i](x, H, W)
            res.append((H, W))

        # Decoder, consuming skips from the second-finest upward.
        for j, (expand, fuse, stage) in enumerate(zip(self.up, self.fuse, self.dec)):
            x, H, W = expand(x, H, W)
            skip_x, _, _ = skips[self.num_stages - 2 - j]
            x = fuse(torch.cat([x, skip_x], dim=-1))
            x = stage(x, H, W)

        x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.out_dim, H, W)
        # Crop padding back to the patch-embed resolution.
        return x[:, :, : (x.shape[2] - pad_h) if pad_h else None,
                 : (x.shape[3] - pad_w) if pad_w else None]
