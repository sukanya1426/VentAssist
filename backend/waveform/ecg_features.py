"""Waveform stage 4.4 — ECG features: HRV (SDNN) and arrhythmia rate.

Pan–Tompkins-style R-peak detection on a (filtered) ECG window, then heart-rate
variability and a simple arrhythmia-rate heuristic. Returns NaN for a feature if
fewer than 5 valid R-peaks are found.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Pan–Tompkins: derivative → square → moving-window integrate → threshold.

    Tuned for the heavily-undersampled (~62.5 Hz) mimic4wdb ECG: the signal is
    robust-scaled to reject the large baseline artifacts seen in this data, and a
    normalised height + prominence threshold suppresses T-wave / noise peaks that
    would otherwise inflate HRV.
    """
    if len(ecg) < int(fs):
        return np.array([], dtype=int)
    # robust scale to tame ±40 mV artifacts
    med = np.median(ecg)
    mad = np.median(np.abs(ecg - med)) + 1e-9
    z = np.clip((ecg - med) / (1.4826 * mad), -8, 8)

    diff = np.diff(z, prepend=z[0])
    squared = diff ** 2
    win = max(1, int(0.15 * fs))                 # ~0.15 s integration window
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    peak = integrated.max()
    if peak < 1e-9:
        return np.array([], dtype=int)
    integrated /= peak                           # normalise to [0, 1]
    min_dist = max(1, int(0.3 * fs))             # >= 0.3 s between beats (<=200 bpm)
    peaks, _ = signal.find_peaks(integrated, height=0.2, distance=min_dist,
                                 prominence=0.1)
    return peaks


def ecg_features(ecg: np.ndarray, fs: float) -> dict[str, float]:
    """Return {HRV_SDNN (ms), Arrhythmia_rate (0-1)} or NaNs if too few peaks."""
    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 5:
        return {"HRV_SDNN": np.nan, "Arrhythmia_rate": np.nan}

    rr_ms = np.diff(peaks) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms >= 333) & (rr_ms <= 1500)]   # 40–180 bpm plausibility
    if len(rr_ms) < 4:
        return {"HRV_SDNN": np.nan, "Arrhythmia_rate": np.nan}

    sdnn = float(np.std(rr_ms))

    # Arrhythmia rate: fraction of 1-minute sub-windows flagged irregular.
    peak_times_s = peaks / fs
    total_min = max(1, int(np.ceil(peak_times_s[-1] / 60.0)))
    flagged = 0
    counted = 0
    for m in range(total_min):
        sel = (peak_times_s >= m * 60) & (peak_times_s < (m + 1) * 60)
        beats = peaks[sel]
        if len(beats) < 3:
            continue
        rr = np.diff(beats) / fs * 1000.0
        rr = rr[(rr >= 333) & (rr <= 1500)]
        if len(rr) < 2:
            continue
        counted += 1
        cv = np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else 0.0
        if cv > 0.3 or np.mean(rr) > 1500:
            flagged += 1
    arr_rate = float(flagged / counted) if counted else np.nan
    return {"HRV_SDNN": sdnn, "Arrhythmia_rate": arr_rate}
