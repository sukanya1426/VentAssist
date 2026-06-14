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
    """Winsorise then z-score. `data` columns must follow `feature_order`."""
    data = np.asarray(data, dtype=float)
    out = np.empty_like(data)
    for j, feat in enumerate(feature_order):
        s = stats[feat]
        col = np.clip(data[..., j], s["winsor_low"], s["winsor_high"])
        out[..., j] = (col - s["mean"]) / s["std"]
    return out


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
