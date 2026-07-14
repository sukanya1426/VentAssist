"""Centralised normalisation utilities (Section 5.4).

Stats format (per feature):
    {"mean": float, "std": float, "winsor_low": float, "winsor_high": float}

``transform`` applies winsorisation (clip to [winsor_low, winsor_high]) then
z-scores with (mean, std). ``inverse_transform`` reverses only the z-score (the
winsor clip is not invertible). Stats must be fit on the TRAIN split only.

This is compatible with the ``models/normaliser_stats.json`` produced by
pipeline/state_builder.py; the waveform stage extends it to all 18 features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def fit(data: pd.DataFrame, feature_cols: Sequence[str]) -> dict:
    """Compute winsor bounds (1st/99th pct) + post-winsor mean/std per feature."""
    stats: dict[str, dict[str, float]] = {}
    for feat in feature_cols:
        vals = pd.to_numeric(data[feat], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            stats[feat] = {"mean": 0.0, "std": 1.0,
                           "winsor_low": 0.0, "winsor_high": 0.0}
            continue
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        w = np.clip(vals, lo, hi)
        std = float(np.std(w))
        stats[feat] = {
            "mean": float(np.mean(w)),
            "std": std if std > 1e-8 else 1.0,
            "winsor_low": lo,
            "winsor_high": hi,
        }
    return stats


def transform(data: np.ndarray, stats: dict, feature_order: Sequence[str]) -> np.ndarray:
    """Winsorise then z-score. `data` columns must follow `feature_order`.

    TRAINING-TIME transform. The winsor clip limits the leverage of outliers when
    fitting, which is correct for building the dataset — but it must NOT be used
    to serve live inputs (see ``transform_inference``)."""
    data = np.asarray(data, dtype=float)
    out = np.empty_like(data)
    for j, feat in enumerate(feature_order):
        s = stats[feat]
        col = np.clip(data[..., j], s["winsor_low"], s["winsor_high"])
        out[..., j] = (col - s["mean"]) / s["std"]
    return out


# Bound on |z| at inference. Sized empirically: it covers the clinically important
# input range well beyond the training winsor bounds (e.g. SpO₂ down to ~78, PEEP up
# to ~32, TV up to ~1360), while staying inside the region where the Q-network still
# extrapolates *sanely*. Pure un-clipped z-scoring is NOT safe: at SpO₂ ≈ 50 (z ≈ −19)
# the network extrapolates to "cut TV, lower FiO₂" on a critically hypoxaemic patient.
# Clamping in z-space instead saturates at a clinically CORRECT extreme (raise PEEP).
INFERENCE_CLIP_SIGMA = 8.0


def transform_inference(data: np.ndarray, stats: dict, feature_order: Sequence[str],
                        clip_sigma: float = INFERENCE_CLIP_SIGMA) -> np.ndarray:
    """SERVING-TIME transform: z-score WITHOUT the raw winsor clip, bounded in z-space.

    The training winsor bounds are the 1st/99th percentile of the *training* data
    (e.g. SpO₂ ∈ [89, 100], PEEP ∈ [0, 18], TV ∈ [227, 863]). Applying them to live
    inputs silently maps every critically hypoxaemic patient (SpO₂ 55, 70, 80, 84…)
    onto the SAME vector as SpO₂ 89 — so the recommendation cannot change no matter
    what the clinician types. That is an input-saturation bug, and in a CDSS it is a
    safety bug, not just a UX one.

    Here we z-score the raw value directly (so distinct inputs stay distinct) and
    bound the result to ±``clip_sigma`` (so the network cannot extrapolate into
    nonsense). Values beyond the bound saturate at a clinically correct extreme, and
    the learned OOD/support detector flags them + tempers the reported confidence.
    """
    data = np.asarray(data, dtype=float)
    out = np.empty_like(data)
    for j, feat in enumerate(feature_order):
        s = stats[feat]
        out[..., j] = (data[..., j] - s["mean"]) / s["std"]
    return np.clip(out, -clip_sigma, clip_sigma)


def inverse_transform(data: np.ndarray, stats: dict,
                      feature_order: Sequence[str]) -> np.ndarray:
    """Reverse the z-score (winsor clip is not reversed)."""
    data = np.asarray(data, dtype=float)
    out = np.empty_like(data)
    for j, feat in enumerate(feature_order):
        s = stats[feat]
        out[..., j] = data[..., j] * s["std"] + s["mean"]
    return out


def save(stats: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
