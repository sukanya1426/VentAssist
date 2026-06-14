"""Waveform stage 4.5 — Plethysmography feature: Perfusion Index (%).

PI = mean(AC / DC) * 100, where AC is the pulse amplitude (peak - trough) and DC
is the local baseline. Returns NaN if fewer than 10 valid pulse cycles are found.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def pleth_features(pleth: np.ndarray, fs: float) -> dict[str, float]:
    if len(pleth) < int(2 * fs):
        return {"Perfusion_Index": np.nan}

    std = np.std(pleth)
    if std < 1e-9:
        return {"Perfusion_Index": np.nan}

    min_dist = max(1, int(0.4 * fs))           # >= 0.4 s between pulses
    peaks, _ = signal.find_peaks(pleth, distance=min_dist, prominence=0.05 * std)
    if len(peaks) < 11:
        return {"Perfusion_Index": np.nan}

    dc_win = max(1, int(2 * fs))               # 2 s moving-average baseline
    dc = np.convolve(pleth, np.ones(dc_win) / dc_win, mode="same")

    ratios = []
    for i in range(len(peaks) - 1):
        p0, p1 = peaks[i], peaks[i + 1]
        trough_idx = p0 + int(np.argmin(pleth[p0:p1])) if p1 > p0 else p0
        ac = pleth[p0] - pleth[trough_idx]
        dc_local = dc[p0]
        if dc_local > 1e-9 and ac > 0:
            ratios.append(ac / dc_local)

    if len(ratios) < 10:
        return {"Perfusion_Index": np.nan}
    pi = float(np.clip(np.mean(ratios) * 100.0, 0.01, 30.0))
    return {"Perfusion_Index": pi}
