"""Train the residual diffusion model (method C) on top of a frozen mean model.

    python -m training.train_diffusion --mean-run swin_B_modis --tag B
    python -m training.train_diffusion --mean-run swin_B_modis --max-minutes 25   # backfill

Same leakage controls as train_swin: crops only where the whole crop avoids
the holdout + buffer, conditioning normaliser = the mean model's (fit on
training days x pixels), loss mask valid & ~exclusion & finite target.
``sigma_data`` is the std of the residual-of-the-mean on training days x
pixels. Weights are tracked with an EMA, and the EMA weights are what is saved
and evaluated. Model selection: CRPS of a 4-member ensemble against the target
on 24 dev days x training pixels (kWh/m2/day).

Writes runs/<group>/diff_<tag>/best.pt (+ last.pt to resume, meta.json when
done). Everything the predictor needs is in best.pt; the mean model is
re-loaded from its own checkpoint, named in best.pt.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from models.diffusion import EDM
from training.common import builder, dump, load_all, run_dir, runs_root
from training.train_swin import SwinPredictor, crop_origins, pick_device


def crps_ensemble(ens: np.ndarray, obs: np.ndarray) -> float:
    """Mean CRPS of ensemble (M, N) against obs (N,), the fair (unbiased) form."""
    m = ens.shape[0]
    t1 = np.abs(ens - obs[None]).mean(0)
    srt = np.sort(ens, 0)
    k = np.arange(1, m + 1)[:, None]
    t2 = (2 * (2 * k - m - 1) * srt).sum(0) / (m * (m - 1))      # E|X - X'| (fair)
    return float(np.mean(t1 - 0.5 * t2))


class Conditioner:
    """Mean-model residual and normalised conditioning stack for every day,
    precomputed once (float16 in RAM: ~5 GB for 21 years on the real store)."""

    def __init__(self, fb, mean: SwinPredictor, days: np.ndarray, pad_to: int = 8,
                 r_scale: float | None = None, cache: Path | None = None):
        self.fb, self.mean = fb, mean
        self.ny, self.nx = fb.ny, fb.nx
        self.H = -(-fb.ny // pad_to) * pad_to
        self.W = -(-fb.nx // pad_to) * pad_to
        self.index = {int(t): i for i, t in enumerate(days)}
        if cache is not None and cache.exists():
            z = np.load(cache)
            if np.array_equal(z["days"], days):
                self.cond, self.rmean = z["cond"], z["rmean"]
                self.r_scale = float(z["r_scale"])
                print(f"[diff] conditioning loaded from {cache}", flush=True)
                return
        C = len(mean.names) + 1
        self.cond = np.zeros((len(days), C, fb.ny, fb.nx), np.float16)
        self.rmean = np.zeros((len(days), fb.ny, fb.nx), np.float32)
        t0 = time.time()
        for i, t in enumerate(days):
            f = fb.day_fields(int(t), mean.names)
            x = np.stack([mean.nz.transform(n, f[n]) for n in mean.names])
            r = mean.raw(int(t))[mean.point]
            self.rmean[i] = r
            self.cond[i, :-1] = np.nan_to_num(x)
            if i % 1000 == 0:
                print(f"[diff] conditioning {i}/{len(days)} ({time.time() - t0:.0f}s)", flush=True)
        # Fixed at training time and stored in the checkpoint, so evaluation
        # conditions on exactly the same scaling.
        self.r_scale = r_scale or float(np.nanstd(self.rmean[:, fb.train_pix])) or 1.0
        self.cond[:, -1] = (self.rmean / self.r_scale).astype(np.float16)
        if cache is not None:
            np.savez(cache, days=days, cond=self.cond, rmean=self.rmean,
                     r_scale=self.r_scale)

    def residual(self, t: int) -> np.ndarray:
        """Target residual the diffusion learns: r_true - r_mean (index units)."""
        return self.fb.residual_target(t) - self.rmean[self.index[t]]

    def full(self, t: int) -> np.ndarray:
        """(C, H, W) conditioning padded to the network's multiple."""
        c = self.cond[self.index[t]].astype(np.float32)
        out = np.zeros((c.shape[0], self.H, self.W), np.float32)
        out[:, :self.ny, :self.nx] = c
        return out


class DiffusionPredictor:
    """GHI ensembles from a trained residual-diffusion checkpoint.

    ``member(t, k)`` is reproducible (seed = f(t, k)); ``__call__`` returns
    member 0 so the deterministic evaluation harness can score one realisation.
    """

    def __init__(self, model: EDM, cond: Conditioner, sigma_data: float, steps: int,
                 device):
        self.model, self.c, self.sd, self.steps, self.dev = model, cond, sigma_data, steps, device
        self.fb = cond.fb
        self._cache: dict = {}

    @classmethod
    def from_checkpoint(cls, ckpt_path, fb, dcfg, tcfg, days, device=None, steps: int = 18):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        dev = pick_device(device)
        mean = SwinPredictor.from_checkpoint(runs_root(dcfg, tcfg) / state["mean_run"] / "best.pt",
                                             fb, device)
        cond = Conditioner(fb, mean, days, r_scale=state["r_scale"])
        model = EDM(state["n_cond"], 1.0, **state["unet"]).to(dev)     # trained on e / sd
        model.load_state_dict(state["ema"])
        model.eval()
        return cls(model, cond, state["sigma_data"], steps, dev)

    @torch.no_grad()
    def residuals(self, t: int, n: int) -> np.ndarray:
        """(n, ny, nx) sampled residual-of-the-mean, index units."""
        key = (t, n)
        if key not in self._cache:
            g = torch.Generator(device="cpu").manual_seed(1_000_003 * t + 17)
            cond = torch.from_numpy(self.c.full(t))[None].repeat(n, 1, 1, 1).to(self.dev)
            z = torch.randn((n, 1, self.c.H, self.c.W), generator=g)
            e = self.model.sample(cond, z, steps=self.steps)
            self._cache = {key: e[:, 0, :self.c.ny, :self.c.nx].cpu().numpy() * self.sd}
        return self._cache[key]

    def ensemble(self, t: int, n: int) -> np.ndarray:
        """(n, ny, nx) GHI members, kWh/m2/day."""
        fb = self.fb
        base = fb.kt_bilinear(t) + self.c.rmean[self.c.index[t]]
        kt = np.stack([fb.clip(base + e, t) for e in self.residuals(t, n)])
        return (kt * fb.norm(t)[None]).astype("float32")

    def member(self, t: int, k: int = 0, n: int = 8) -> np.ndarray:
        return self.ensemble(t, n)[k]

    def __call__(self, t: int) -> np.ndarray:
        return self.member(t, 0)


def main(mean_run: str, tag: str | None, steps: int | None, max_minutes: float | None,
         device: str | None, batch: int | None, crop: int | None) -> None:
    t_job = time.time()
    dcfg, tcfg = load_all()
    dc = dict(tcfg.get("diffusion", {}))
    steps = int(steps or dc.get("steps", 60000))
    batch = int(batch or dc.get("batch_size", 64))
    crop = int(crop or dc.get("crop", 32))
    unet = dict(dc.get("unet", {"base": 64, "mults": [1, 2, 4], "blocks": 2, "dropout": 0.1}))
    unet["mults"] = tuple(unet["mults"])
    rng = np.random.default_rng(int(dc.get("seed", 0)))
    torch.manual_seed(int(dc.get("seed", 0)))
    fb = builder(dcfg, tcfg)
    out = run_dir(dcfg, tcfg, "diff_" + (tag or mean_run.replace("swin_", "")))
    dev = pick_device(device)
    mean = SwinPredictor.from_checkpoint(runs_root(dcfg, tcfg) / mean_run / "best.pt", fb, device)

    train_days = np.nonzero(fb.train_days)[0]
    val_days = np.nonzero(fb.split == "val")[0]
    val_days = val_days[np.linspace(0, val_days.size - 1, min(24, val_days.size)).astype(int)]
    # Cached next to the run: each short backfill job would otherwise rebuild
    # ~5 GB of conditioning from scratch.
    cond = Conditioner(fb, mean, np.r_[train_days, val_days], cache=out / "cond_cache.npz")
    # sigma_data: std of the residual-of-the-mean on training pixels
    smp = rng.choice(train_days, min(300, train_days.size), replace=False)
    sd = float(np.nanstd(np.concatenate([cond.residual(int(t))[fb.train_pix] for t in smp])))
    origins = crop_origins(fb.exclusion, crop)
    if not len(origins):
        # Small (synthetic) domains: any crop, exclusion zone masked from the
        # loss (mask_all = train_pix excludes it), as train_swin does.
        print("[diff] WARNING no crop fits outside holdout+buffer; masking instead")
        origins = np.argwhere(np.ones((fb.ny - crop + 1, fb.nx - crop + 1), bool))
    n_cond = cond.cond.shape[1]
    model = EDM(n_cond, 1.0, **unet).to(dev)           # works on e / sd
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=float(dc.get("lr", 2e-4)),
                            weight_decay=float(dc.get("weight_decay", 0.0)))
    ema_decay = float(dc.get("ema", 0.999))
    last = out / "last.pt"
    step, best = 0, np.inf
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); step, best = st["step"], st["best"]
        print(f"[diff] resuming at step {step} (best dev CRPS {best:.4f})")
    print(f"[diff] {out.name}: mean={mean_run}, {n_cond} cond channels, sigma_data {sd:.4f}, "
          f"crop {crop}, batch {batch}, {steps} steps on {dev}", flush=True)

    mask_all = fb.train_pix
    eval_every = int(dc.get("eval_every", 2000))

    def batch_tensors():
        X, Y, M = [], [], []
        for _ in range(batch):
            t = int(rng.choice(train_days))
            r0, c0 = origins[rng.integers(len(origins))]
            sl = (slice(r0, r0 + crop), slice(c0, c0 + crop))
            X.append(cond.cond[cond.index[t]][(slice(None),) + sl].astype(np.float32))
            e = cond.residual(t)[sl] / sd
            m = mask_all[sl] & np.isfinite(e)
            Y.append(np.nan_to_num(e)); M.append(m)
        f = lambda a: torch.from_numpy(np.stack(a)[:, None]).to(dev)  # noqa: E731
        return torch.from_numpy(np.stack(X)).to(dev), f(Y).float(), f(M)

    def dev_crps() -> float:
        pred = DiffusionPredictor(ema, cond, sd, 18, dev)
        ens, obs = [], []
        for t in val_days:
            g = pred.ensemble(int(t), 4)
            ens.append(g[:, fb.train_pix]); obs.append(fb.ghi[t][fb.train_pix])
        ens, obs = np.concatenate(ens, 1), np.concatenate(obs)
        ok = np.isfinite(obs) & np.isfinite(ens).all(0)
        return crps_ensemble(ens[:, ok], obs[ok])

    t0, tot, ep_time = time.time(), 0.0, 0.0
    while step < steps:
        if max_minutes and time.time() - t_job + 1.3 * ep_time > 60 * max_minutes:
            print(f"[diff] stopping at step {step} to fit the job time; resume later")
            return
        tc, n_done = time.time(), 0
        for _ in range(eval_every):
            model.train()
            x, y, m = batch_tensors()
            loss = model.loss(y, x, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - ema_decay)
            tot += loss.item(); step += 1; n_done += 1
            if step >= steps:
                break
        score = dev_crps()
        print(f"[diff] step {step}/{steps} loss {tot / max(n_done, 1):.4f} dev CRPS {score:.4f} "
              f"({time.time() - tc:.0f}s)", flush=True)
        tot = 0.0
        state = {"ema": ema.state_dict(), "n_cond": n_cond, "sigma_data": sd,
                 "r_scale": cond.r_scale,
                 "unet": unet, "mean_run": mean_run, "step": step, "dev_crps": score}
        if score < best:
            best = score
            torch.save(state, out / "best.pt")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "step": step, "best": best}, last)
        ep_time = time.time() - tc
    dump(out / "meta.json", {"mean_run": mean_run, "steps": steps, "sigma_data": sd,
                             "best_dev_crps_kwh": best, "unet": unet, "crop": crop,
                             "batch": batch, "train_minutes": (time.time() - t0) / 60})
    print(f"[diff] done; best dev CRPS {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-run", default="swin_B_modis")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--crop", type=int, default=None)
    a = ap.parse_args()
    main(a.mean_run, a.tag, a.steps, a.max_minutes, a.device, a.batch, a.crop)
