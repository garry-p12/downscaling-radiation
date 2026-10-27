"""Residual diffusion (CorrDiff-style) for the part of the target a mean model
cannot place.

Why. The mean models regress to the conditional mean, so where the inputs
cannot say WHERE sub-60 km cloud texture goes (band correlation with NSRDB
0.03-0.10), they correctly draw none of it: amplitude ratios of 0.2-0.4 below
60 km. A generative model instead draws texture of the right amount and
statistics and expresses the placement uncertainty as an ensemble. That buys
realistic fields and calibrated uncertainty, not lower RMSE of a single draw.

Formulation (Mardani et al. 2024, CorrDiff; Karras et al. 2022, EDM):
    r_true = r_mean + e,      r_mean from a frozen mean model (Swin),
and a conditional denoiser D(e_noisy; sigma, cond) learns p(e | cond), with
cond = the mean model's input channels + r_mean. Everything is in the model's
index space (kt or k*), scaled by ``sigma_data``. EDM preconditioning:
    D = c_skip e_noisy + c_out F(c_in e_noisy; c_noise, cond).
Masked pixels (outside the land grid, missing target, holdout + buffer during
training) carry no loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbed(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        x = x[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([x.cos(), x.sin()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, cin // 4) or 1, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(min(32, cout // 4) or 1, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.emb(e)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + s) + b                      # FiLM on the noise level
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class UNet(nn.Module):
    """Small fully convolutional U-Net; input H, W must be divisible by 2**levels."""

    def __init__(self, c_in: int, c_out: int = 1, base: int = 64, mults=(1, 2, 4),
                 blocks: int = 2, emb: int = 256, dropout: float = 0.1):
        super().__init__()
        self.levels = len(mults)
        self.embed = nn.Sequential(FourierEmbed(emb), nn.Linear(emb, emb), nn.SiLU(),
                                   nn.Linear(emb, emb))
        self.inp = nn.Conv2d(c_in, base, 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        ch, skips = base, []
        for i, m in enumerate(mults):
            lvl = nn.ModuleList()
            for _ in range(blocks):
                lvl.append(ResBlock(ch, base * m, emb, dropout))
                ch = base * m
            self.down.append(lvl)
            skips.append(ch)
            self.pool.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                             if i < len(mults) - 1 else nn.Identity())
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb, dropout), ResBlock(ch, ch, emb, dropout)])
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            lvl = nn.ModuleList()
            for j in range(blocks):
                lvl.append(ResBlock(ch + (skips[i] if j == 0 else 0), base * m, emb, dropout))
                ch = base * m
            self.up.append(lvl)
            self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest") if i > 0
                                 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(min(32, ch // 4), ch), nn.SiLU(),
                                 nn.Conv2d(ch, c_out, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, c_noise):
        e = self.embed(c_noise)
        h = self.inp(x)
        hs = []
        for lvl, pool in zip(self.down, self.pool):
            for blk in lvl:
                h = blk(h, e)
            hs.append(h)
            h = pool(h)
        for blk in self.mid:
            h = blk(h, e)
        for lvl, up in zip(self.up, self.upsample):
            h = torch.cat([h, hs.pop()], 1)
            for blk in lvl:
                h = blk(h, e)
            h = up(h)
        return self.out(h)


class EDM(nn.Module):
    """EDM-preconditioned conditional denoiser for a 1-channel residual."""

    def __init__(self, n_cond: int, sigma_data: float = 1.0, **unet_kw):
        super().__init__()
        self.sd = sigma_data
        self.net = UNet(n_cond + 1, 1, **unet_kw)

    def forward(self, x, sigma, cond):
        sd = self.sd
        s = sigma.view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        f = self.net(torch.cat([c_in * x, cond], 1), sigma.log() / 4)
        return c_skip * x + c_out * f

    def loss(self, e, cond, mask, p_mean: float = -1.2, p_std: float = 1.2):
        """EDM training loss on residual ``e`` (B,1,H,W); ``mask`` bool, same shape."""
        sigma = (torch.randn(e.shape[0], device=e.device) * p_std + p_mean).exp()
        s = sigma.view(-1, 1, 1, 1)
        w = (s ** 2 + self.sd ** 2) / (s * self.sd) ** 2
        e0 = torch.where(mask, e, torch.zeros_like(e))
        d = self(e0 + torch.randn_like(e0) * s, sigma, cond)
        m = mask.float()
        return (w * m * (d - e0) ** 2).sum() / m.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, cond, z=None, steps: int = 18, sigma_min: float = 0.002,
               sigma_max: float = 80.0, rho: float = 7.0):
        """EDM deterministic Heun sampler (Karras et al. 2022, Alg. 1). ``z`` is
        the unit initial noise (B,1,H,W); drawn here if not given."""
        b, _, h, w = cond.shape
        i = torch.arange(steps, dtype=torch.float64)
        t = (sigma_max ** (1 / rho) + i / (steps - 1)
             * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t = [float(v) for v in t] + [0.0]
        if z is None:
            z = torch.randn((b, 1, h, w))
        x = z.to(cond.device, torch.float32) * t[0]
        for k in range(steps):
            tc, tn = t[k], t[k + 1]
            d = (x - self(x, torch.full((b,), tc, device=cond.device), cond)) / tc
            xn = x + (tn - tc) * d
            if tn > 0:
                d2 = (xn - self(xn, torch.full((b,), tn, device=cond.device), cond)) / tn
                xn = x + (tn - tc) * (d + d2) / 2
            x = xn
        return x
