"""M2 / M3: Swin U-Net residual-kt model, optionally with a quantile head.

Backbone: models/encoder_swin.SwinUNet, unchanged from the temperature work.
Head: conv + pixel shuffle back from patch-embed to full resolution (same
shape contract as the temperature TempHead). Output channels: 1 residual for
M2, or one per quantile for M3 (median used as the point prediction).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_swin import SwinUNet


class RadiationSwin(nn.Module):
    def __init__(self, in_channels: int, model_cfg: dict, n_out: int = 1):
        super().__init__()
        enc = model_cfg["encoder"]
        self.patch_size = int(model_cfg["patch_size"])
        self.backbone = SwinUNet(in_channels=in_channels, patch_size=self.patch_size,
                                 embed_dim=enc["embed_dim"], depths=enc["depths"],
                                 num_heads=enc["num_heads"],
                                 window_size=enc["window_size"],
                                 mlp_ratio=enc["mlp_ratio"], drop_path=enc["drop_path"])
        ps = self.patch_size
        self.head = nn.Sequential(
            nn.Conv2d(enc["embed_dim"], enc["embed_dim"], 3, padding=1), nn.GELU(),
            nn.Conv2d(enc["embed_dim"], n_out * ps * ps, 3, padding=1),
            nn.PixelShuffle(ps))
        nn.init.zeros_(self.head[2].weight)          # start at "bilinear is right"
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        ps = self.patch_size
        ph, pw = (-H) % ps, (-W) % ps
        if ph or pw:
            mode = "reflect" if (ph < H and pw < W) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return self.head(self.backbone(x))[..., :H, :W]


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  weight: torch.Tensor, quantiles: list[float] | None) -> torch.Tensor:
    """Masked, weighted MSE (1 channel) or pinball loss (one channel per q)."""
    m = mask & torch.isfinite(target)
    t = torch.where(m, target, torch.zeros_like(target))
    w = weight * m
    if not quantiles:
        se = (pred[:, :1] - t) ** 2
        return (se * w).sum() / w.sum().clamp_min(1e-8)
    losses = []
    for i, q in enumerate(quantiles):
        e = t - pred[:, i:i + 1]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    pin = torch.stack(losses).mean(0)
    return (pin * w).sum() / w.sum().clamp_min(1e-8)


def consistency_penalty(kt_pred: torch.Tensor, kt_ref: torch.Tensor,
                        enc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Soft coarse-cell consistency (plan 5): squared difference between the
    block mean of predicted kt and the block mean of the reference
    (POWER kt + training offset) inside each coarse cell, over valid pixels.

    kt_pred, kt_ref, mask: (B, 1, H, W); enc: (B, H, W) coarse cell ids.
    """
    B = kt_pred.shape[0]
    total = kt_pred.new_zeros(())
    for b in range(B):
        m = mask[b, 0]
        if not m.any():
            continue
        ids = enc[b][m]
        uniq, inv = torch.unique(ids, return_inverse=True)
        cnt = torch.zeros(uniq.numel(), device=kt_pred.device).index_add_(
            0, inv, torch.ones_like(inv, dtype=kt_pred.dtype))
        mp = torch.zeros_like(cnt).index_add_(0, inv, kt_pred[b, 0][m]) / cnt
        mr = torch.zeros_like(cnt).index_add_(0, inv, kt_ref[b, 0][m]) / cnt
        total = total + ((mp - mr) ** 2).mean()
    return total / B


# --------------------------------------------------------------------------- #
# Texture-preserving loss terms
# --------------------------------------------------------------------------- #
# Under MSE a scale predicted with correlation rho comes out at amplitude rho:
# the model shrinks what it cannot place. The first real run shows it (every
# model loses variance below ~300 km; eval/spectra.json). Two remedies, both
# from downscaling-temperature training/losses.py:
#
#   amse   Subich et al. (2025) spectrally adjusted MSE: per radial wavenumber,
#          (|P| - |T|)^2 + 2 max(|P|^2, |T|^2)(1 - coherence). Penalises missing
#          AMPLITUDE separately from misplaced phase, so shrinking an
#          unpredictable scale is no longer free.
#   ssim   1 - SSIM on the kt field: local means, variances, covariance.
#
# Sharpening is expected to cost a little RMSE; that is the trade being bought.

_BIN_CACHE: dict = {}


def _radial_bins(h: int, w: int, device):
    key = (h, w, str(device))
    if key not in _BIN_CACHE:
        ky = torch.fft.fftfreq(h, device=device).view(-1, 1)
        kx = torch.fft.fftfreq(w, device=device).view(1, -1)
        kr = torch.sqrt(ky ** 2 + kx ** 2)
        nb = max(2, min(h, w) // 2)
        idx = (kr / (kr.max() + 1e-12) * nb).long().clamp_(max=nb - 1)
        _BIN_CACHE[key] = (idx.reshape(-1), nb)
    return _BIN_CACHE[key]


def amse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
         eps: float = 1e-10) -> torch.Tensor:
    """Batch-pooled, Hann-windowed AMSE on (B, 1, H, W); masked cells -> 0 in
    both fields. Scaled to sit on the MSE scale (Parseval), as in the
    temperature implementation."""
    m = (mask & torch.isfinite(target)).to(pred.dtype)
    p = pred[:, :1] * m
    t = torch.nan_to_num(target) * m
    b, _, h, w = p.shape
    win = torch.outer(torch.hann_window(h, periodic=False, device=p.device, dtype=p.dtype),
                      torch.hann_window(w, periodic=False, device=p.device, dtype=p.dtype))
    p, t = p * win, t * win
    fp, ft = torch.fft.fft2(p.squeeze(1)), torch.fft.fft2(t.squeeze(1))
    idx, nb = _radial_bins(h, w, p.device)
    ix = idx.unsqueeze(0).expand(b, -1)
    z = p.new_zeros((b, nb))
    psd_p = z.scatter_add(1, ix, (fp.abs() ** 2).reshape(b, -1)).mean(0)
    psd_t = z.clone().scatter_add(1, ix, (ft.abs() ** 2).reshape(b, -1)).mean(0)
    cross = z.clone().scatter_add(
        1, ix, (fp.real * ft.real + fp.imag * ft.imag).reshape(b, -1)).mean(0)
    coh = cross / torch.sqrt(psd_p * psd_t + eps)
    loss = (torch.sqrt(psd_p + eps) - torch.sqrt(psd_t + eps)) ** 2 \
        + 2.0 * torch.maximum(psd_p, psd_t) * (1.0 - coh)
    return loss.sum() / ((h * w) ** 2 * (win ** 2).mean())


def ssim_loss(kt_pred: torch.Tensor, kt_true: torch.Tensor, mask: torch.Tensor,
              ws: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1 - mean SSIM over valid cells. data_range = each sample's target range
    over valid cells (the per-day convention of evaluation/structure.py, so the
    optimised and the reported SSIM are the same quantity)."""
    m = mask & torch.isfinite(kt_true)
    t = torch.where(m, kt_true, kt_pred.detach())        # invalid: no signal
    p = torch.where(m, kt_pred, kt_pred.detach())
    big = torch.finfo(t.dtype).max
    tmax = torch.where(m, t, torch.full_like(t, -big)).amax((1, 2, 3), keepdim=True)
    tmin = torch.where(m, t, torch.full_like(t, big)).amin((1, 2, 3), keepdim=True)
    dr = (tmax - tmin).clamp_min(1e-3)
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    g = torch.exp(-((torch.arange(ws, device=p.device, dtype=p.dtype) - ws // 2) ** 2)
                  / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :]).view(1, 1, ws, ws)

    def f(a):
        return F.conv2d(F.pad(a, (ws // 2,) * 4, mode="reflect"), k)

    mp, mt = f(p), f(t)
    sp, st, spt = f(p * p) - mp ** 2, f(t * t) - mt ** 2, f(p * t) - mp * mt
    s = ((2 * mp * mt + c1) * (2 * spt + c2)) / ((mp ** 2 + mt ** 2 + c1) * (sp + st + c2))
    mf = m.to(s.dtype)
    return 1.0 - (s * mf).sum() / mf.sum().clamp_min(1.0)
