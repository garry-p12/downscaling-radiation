"""Train M2 (Swin U-Net, residual kt) or M3 (with quantile head).

    python -m training.train_swin --track A
    python -m training.train_swin --track B --quantiles 0.1 0.5 0.9     # M3

Leakage controls (plan 6):
  * training crops are drawn only where the whole crop lies outside the
    holdout + buffer (``exclusion``), so no patch straddles the boundary; if
    the domain is too small for any such crop, crops are drawn anywhere and
    the exclusion zone is masked out of the loss (with a warning);
  * the input normaliser is fit on training days x training pixels;
  * the loss mask is valid & ~exclusion & finite target.

Model selection: dev (val years) RMSE in kWh/m2/day over training pixels.
Writes runs/<store>/swin_<track>[_q]/best.pt with everything the predictor
needs (model cfg, channel list, normaliser, quantiles).
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch

from common import load_config
from common.normalize import Normalizer
from common.tracking import Tracker
from evaluation.metrics import rmse
from features.builder import FeatureBuilder
from models.swin import RadiationSwin, amse, consistency_penalty, ssim_loss, weighted_loss
from training.common import builder, dump, load_all, run_dir


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_normalizer(fb: FeatureBuilder, names: list[str], rng, n_days: int = 150):
    days = np.nonzero(fb.train_days)[0]
    days = rng.choice(days, size=min(n_days, days.size), replace=False)
    acc = {n: [] for n in names}
    for t in days:
        f = fb.day_fields(int(t), names)
        for n in names:
            acc[n].append(f[n][fb.train_pix])
    nz = Normalizer()
    for n in names:
        nz.fit(n, np.concatenate(acc[n]))
    return nz


def crop_origins(exclusion: np.ndarray, crop: int) -> np.ndarray:
    """(k, 2) top-left corners whose crop x crop window avoids ``exclusion``."""
    ny, nx = exclusion.shape
    ii = np.pad(exclusion.astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r = np.arange(ny - crop + 1)[:, None]
    c = np.arange(nx - crop + 1)[None, :]
    s = ii[r + crop, c + crop] - ii[r, c + crop] - ii[r + crop, c] + ii[r, c]
    return np.argwhere(s == 0)


class Batcher:
    def __init__(self, fb: FeatureBuilder, names, nz: Normalizer, crop: int, rng):
        self.fb, self.names, self.nz, self.rng = fb, names, nz, rng
        self.crop = min(crop, fb.ny, fb.nx)
        self.origins = crop_origins(fb.exclusion, self.crop)
        self.strict = len(self.origins) > 0
        if not self.strict:
            warnings.warn("[swin] no crop fits outside holdout+buffer; using any "
                          "crop with the exclusion zone masked from the loss")
            self.origins = np.argwhere(np.ones((fb.ny - self.crop + 1,
                                                fb.nx - self.crop + 1), bool))
        self.mask = fb.train_pix
        self.enc = fb.enc.reshape(fb.ny, fb.nx)

    def inputs(self, t: int) -> np.ndarray:
        f = self.fb.day_fields(t, self.names)
        x = np.stack([self.nz.transform(n, f[n]) for n in self.names])
        return np.nan_to_num(x).astype("float32")

    def sample(self, days: np.ndarray, bs: int):
        X, Y, M, Wt, KB, E = [], [], [], [], [], []
        c = self.crop
        for _ in range(bs):
            t = int(self.rng.choice(days))
            r0, c0 = self.origins[self.rng.integers(len(self.origins))]
            sl = (slice(r0, r0 + c), slice(c0, c0 + c))
            X.append(self.inputs(t)[(slice(None),) + sl])
            Y.append(self.fb.residual_target(t)[sl])
            M.append(self.mask[sl])
            toa = self.fb.norm(t)[sl]
            Wt.append(toa / toa.mean())
            KB.append(self.fb.kt_bilinear(t)[sl])
            E.append(self.enc[sl])
        f = lambda a: torch.from_numpy(np.stack(a)[:, None].astype("float32"))  # noqa: E731
        return (torch.from_numpy(np.stack(X)), f(Y),
                torch.from_numpy(np.stack(M)[:, None]), f(Wt), f(KB),
                torch.from_numpy(np.stack(E).astype("int64")))


class SwinPredictor:
    """Full-field GHI predictions from a model (training) or checkpoint (eval)."""

    def __init__(self, model, names, nz: Normalizer, quantiles, fb: FeatureBuilder,
                 device: torch.device):
        self.model, self.names, self.nz, self.q = model, names, nz, quantiles
        self.fb, self.dev = fb, device
        self.point = 0
        if quantiles:
            self.point = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    @classmethod
    def from_checkpoint(cls, ckpt_path, fb: FeatureBuilder, device: str | None = None):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        q = state.get("quantiles")
        dev = pick_device(device)
        model = RadiationSwin(len(state["channels"]), state["model_cfg"],
                              n_out=len(q) if q else 1).to(dev)
        model.load_state_dict(state["model"])
        model.eval()
        return cls(model, state["channels"], Normalizer(state["norm"]), q, fb, dev)

    @torch.no_grad()
    def raw(self, t: int) -> np.ndarray:
        f = self.fb.day_fields(t, self.names)
        x = np.stack([self.nz.transform(n, f[n]) for n in self.names])
        x = torch.from_numpy(np.nan_to_num(x).astype("float32"))[None].to(self.dev)
        return self.model(x)[0].cpu().numpy()            # (n_out, H, W)

    def ghi(self, t: int, channel: int | None = None) -> np.ndarray:
        r = self.raw(t)[self.point if channel is None else channel]
        kt = self.fb.clip(self.fb.kt_bilinear(t) + r, t)
        return kt * self.fb.norm(t)

    def __call__(self, t: int) -> np.ndarray:
        return self.ghi(t)


def main(track: str, quantiles, epochs, model_cfg_name: str, device: str | None,
         steps_per_epoch: int | None, max_minutes: float | None = None,
         pixel_loss: str | None = None, ssim_weight: float | None = None,
         tag: str | None = None) -> None:
    """Resumable: ``last.pt`` is written every epoch and picked up on restart,
    so training can run as a series of short (backfill-sized) jobs.
    ``max_minutes`` stops cleanly before the next epoch would overrun; the run
    is complete when meta.json exists."""
    t_job = time.time()
    dcfg, tcfg = load_all()
    sc = dict(tcfg["swin"])
    quantiles = quantiles or sc.get("quantiles")
    mcfg = dict(load_config(model_cfg_name))
    rng = np.random.default_rng(int(sc.get("seed", 0)))
    torch.manual_seed(int(sc.get("seed", 0)))
    fb = builder(dcfg, tcfg)
    names = fb.track(track)
    pixel_loss = pixel_loss or sc.get("pixel_loss", "mse")
    ssim_w = float(sc.get("ssim_weight", 0.0) if ssim_weight is None else ssim_weight)
    if pixel_loss == "amse" and quantiles:
        raise ValueError("amse is a point loss; not combinable with --quantiles")
    out = run_dir(dcfg, tcfg, f"swin_{track.replace('+', '_')}" + ("_q" if quantiles else "")
                  + (f"_{tag}" if tag else ""))
    dev = pick_device(device)
    last = out / "last.pt"
    resume = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    nz = Normalizer(resume["norm"]) if resume else fit_normalizer(fb, names, rng)
    batcher = Batcher(fb, names, nz, int(sc["crop"]), rng)
    model = RadiationSwin(len(names), mcfg, n_out=len(quantiles) if quantiles else 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sc["lr"]),
                            weight_decay=float(sc["weight_decay"]))
    epochs = int(epochs or sc["epochs"])
    train_days = np.nonzero(fb.train_days)[0]
    val_days = np.nonzero(fb.split == "val")[0]
    val_days = val_days[np.linspace(0, val_days.size - 1, min(60, val_days.size)).astype(int)]
    bs = int(sc["batch_size"])
    spe = int(steps_per_epoch or max(1, train_days.size // bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * spe)
    loss_kind = sc.get("loss", "mse_kt")
    cw = float(sc.get("consistency_weight", 0.0))
    # Training offset for the consistency reference: mean residual on train.
    offset = float(np.nanmean([np.nanmean(fb.residual_target(int(t))[fb.train_pix])
                               for t in rng.choice(train_days, 100)]))
    tracker = Tracker(tcfg["logging"]["wandb"], dict(tcfg),
                      {"model": out.name, "channels": names})
    print(f"[swin] {out.name}: {len(names)} channels, crop {batcher.crop} "
          f"(strict={batcher.strict}), {spe} steps x {epochs} epochs on {dev}")

    best, start = np.inf, 0
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        best, start = resume["best"], resume["epoch"] + 1
        print(f"[swin] resuming at epoch {start + 1} (best dev {best:.4f})")
    pred = SwinPredictor(model, names, nz, quantiles, fb, dev)
    ep_time = 0.0
    for ep in range(start, epochs):
        if max_minutes and ep > start and \
                time.time() - t_job + 1.3 * ep_time > 60 * max_minutes:
            print(f"[swin] stopping at epoch {ep} to fit the job time; resume later")
            tracker.finish()
            return
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(spe):
            x, y, m, w, kb, enc = batcher.sample(train_days, bs)
            x, y, m, w, kb, enc = (a.to(dev) for a in (x, y, m, w, kb, enc))
            wt = {"mse_kt": torch.ones_like(w), "mse_toa": w, "mse_energy": w ** 2}[loss_kind]
            p = model(x)
            if pixel_loss == "amse":
                loss = amse(p, y, m)
            else:
                loss = weighted_loss(p, y, m, wt, quantiles)
            if ssim_w > 0:
                pt = p[:, pred.point:pred.point + 1]
                loss = loss + ssim_w * ssim_loss(kb + pt, kb + y, m)
            if cw > 0:
                kt_p = kb + p[:, pred.point:pred.point + 1]
                loss = loss + cw * consistency_penalty(kt_p, kb + offset, enc,
                                                       m & torch.isfinite(y))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        model.eval()
        errs_p, errs_o = [], []
        for t in val_days:
            g = pred.ghi(int(t))
            errs_p.append(g[fb.train_pix])
            errs_o.append(fb.ghi[t][fb.train_pix])
        score = rmse(np.concatenate(errs_p), np.concatenate(errs_o))
        tracker.log({"train/loss": tot / spe, "dev/rmse_kwh": score}, step=ep)
        print(f"[swin] epoch {ep + 1}/{epochs} loss {tot / spe:.5f} "
              f"dev RMSE {score:.4f} kWh/m2/day ({time.time() - t0:.0f}s)")
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "model_cfg": mcfg,
                        "channels": names, "norm": nz.stats, "quantiles": quantiles,
                        "track": track, "epoch": ep, "dev_rmse_kwh": score},
                       out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "norm": nz.stats}, last)
        ep_time = time.time() - t0
    dump(out / "meta.json", {"track": track, "channels": names, "best_dev_rmse_kwh": best,
                             "quantiles": quantiles, "loss": loss_kind,
                             "pixel_loss": pixel_loss, "ssim_weight": ssim_w,
                             "consistency_weight": cw, "strict_crops": batcher.strict})
    tracker.summary("dev_rmse_kwh", best)
    tracker.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="A", help="A, B or C, plus modifiers: B+T, B+cod+T")
    ap.add_argument("--quantiles", type=float, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--model-config", default="model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--pixel-loss", choices=["mse", "amse"], default=None)
    ap.add_argument("--ssim-weight", type=float, default=None)
    ap.add_argument("--tag", default=None, help="run-name suffix, e.g. amse")
    a = ap.parse_args()
    main(a.track, a.quantiles, a.epochs, a.model_config, a.device, a.steps_per_epoch,
         a.max_minutes, a.pixel_loss, a.ssim_weight, a.tag)
