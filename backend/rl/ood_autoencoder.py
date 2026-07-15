"""Learned state-density OOD / support detector (SYSTEM_SUMMARY §16 item 7).

Upgrades the bounded local-NN support proxy (test_support_constraint.py) to a
learned *state-conditional density*, mirroring IntelliLung's autoencoder OOD
signal. A small autoencoder is trained to reconstruct the (normalised) offline
states; because it only has capacity to model the data manifold, a state's
reconstruction error is a principled coverage score — low for in-distribution
states, high for out-of-distribution ones (recon error ≈ negative log-density
on the manifold). The support threshold is calibrated as a high percentile of
the *training* reconstruction errors, so "in support" = "as reconstructible as
the training data".

Artifacts: ``models/ood_ae_track_<track>.pt`` (weights + calibrated threshold and
error stats). Used at inference by the policy router to attach an ``in_support`` /
``ood_score`` coverage signal to every recommendation.

Train:  python -m backend.rl.ood_autoencoder --track a
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("ood_ae")


class StateAutoencoder(nn.Module):
    def __init__(self, dim: int, latent: int = 8, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        return self.dec(self.enc(x))


def _recon_error(model: StateAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row reconstruction MSE (chunked) → (N,)."""
    model.eval()
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            xb = torch.as_tensor(X[i:i + 65536], dtype=torch.float32)
            out[i:i + 65536] = ((model(xb) - xb) ** 2).mean(1).numpy()
    return out


def train(track: str = "a", latent: int = 8, hidden: int = 64, epochs: int = 12,
          batch_size: int = 4096, lr: float = 1e-3, percentile: float = 99.0,
          device: str = "cpu") -> dict:
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / (
        "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"))
    feats = d["feature_order"]
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    train_idx = np.where(d["split"] == "train")[0]
    if len(train_idx) == 0:
        train_idx = np.arange(len(S))
    Xtr = S[train_idx]

    dim = Xtr.shape[1]
    model = StateAutoencoder(dim, latent, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(Xtr, device=device)
    rng = np.random.default_rng(config.SPLIT_SEED)
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = torch.as_tensor(perm[i:i + batch_size], device=device)
            xb = Xt[b]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        log.info("AE epoch %d/%d | train MSE=%.5f", ep + 1, epochs, tot / n)

    err = _recon_error(model, Xtr)
    threshold = float(np.percentile(err, percentile))
    ckpt = {"state_dict": model.state_dict(), "dim": dim, "latent": latent,
            "hidden": hidden, "threshold": threshold, "percentile": percentile,
            "err_mean": float(err.mean()), "err_std": float(err.std()),
            "err_median": float(np.median(err)), "track": track,
            "n_train": int(n)}
    out = config.MODEL_PATH / f"ood_ae_track_{track}.pt"
    torch.save(ckpt, out)
    log.info("Saved OOD autoencoder → %s (threshold p%.0f=%.5f, median=%.5f)",
             out, percentile, threshold, ckpt["err_median"])
    return ckpt


class OODDetector:
    """Loaded OOD detector: reconstruction-error support score for a state."""

    def __init__(self, ckpt: dict):
        self.model = StateAutoencoder(ckpt["dim"], ckpt["latent"], ckpt["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = float(ckpt["threshold"])
        self.err_median = float(ckpt["err_median"])
        self.err_mean = float(ckpt["err_mean"])
        self.err_std = float(ckpt["err_std"])

    @classmethod
    def load(cls, track: str = "a") -> "OODDetector":
        path = config.MODEL_PATH / f"ood_ae_track_{track}.pt"
        return cls(torch.load(path, map_location="cpu"))

    def score(self, z: np.ndarray) -> float:
        """Reconstruction error for a single normalised state (higher = more OOD)."""
        return float(_recon_error(self.model, np.asarray(z, dtype=np.float32).reshape(1, -1))[0])

    def in_support(self, z: np.ndarray) -> bool:
        return self.score(z) <= self.threshold

    def support_ratio(self, z: np.ndarray) -> float:
        """Score normalised to the calibrated threshold, clipped to [0, 1]:
        1 = as reconstructible as the training median, 0 = at/over the threshold."""
        return self.evaluate(z)["support_ratio"]

    def evaluate(self, z: np.ndarray) -> dict:
        """One forward pass → {score, in_support, support_ratio}."""
        s = self.score(z)
        if s <= self.err_median:
            ratio = 1.0
        elif s >= self.threshold:
            ratio = 0.0
        else:
            ratio = 1.0 - (s - self.err_median) / (self.threshold - self.err_median)
        return {"score": s, "in_support": bool(s <= self.threshold),
                "support_ratio": float(ratio)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    train(args.track, epochs=args.epochs)


if __name__ == "__main__":
    main()
