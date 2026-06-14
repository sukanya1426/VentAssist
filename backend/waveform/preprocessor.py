"""Waveform stage 4.3 — filter, baseline-correct, and quality-flag signals.

Operates on 1-hour windows. Because the data is sampled at ~62.5 Hz (Nyquist
~31 Hz), the spec's ECG 0.5–40 Hz band is capped to 0.5–29 Hz (just below
Nyquist) — documented deviation forced by the sampling rate.

Filters (scipy butter + sosfiltfilt, zero-phase):
  ECG   : band-pass 0.5–min(40, 0.95*Nyq) Hz, order 4
  Pleth : low-pass 10 Hz, order 4
  Resp  : low-pass 1 Hz, order 4, then 60 s moving-median baseline subtraction

Quality flags (per 1-hour window):
  ECG  flatline       : std of 10 s windows < 0.01 for > 30% of the hour
  Pleth motion        : kurtosis of 10 s windows > 10 for > 20% of the hour
  Resp disconnected   : any zero-variance segment > 60 s
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.stats import kurtosis


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #
def fill_nans(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate interior NaNs, edge-fill, return (filled, finite_mask)."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.all():
        return x, finite
    if not finite.any():
        return np.zeros_like(x), finite
    idx = np.arange(len(x))
    filled = x.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return filled, finite


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def _sos_filtfilt(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    # padlen must be < signal length
    if len(x) <= 18:
        return x
    return signal.sosfiltfilt(sos, x)


def bandpass_ecg(x: np.ndarray, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(40.0, 0.95 * nyq)
    sos = signal.butter(4, [0.5 / nyq, hi / nyq], btype="band", output="sos")
    return _sos_filtfilt(x, sos)


def lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    nyq = fs / 2.0
    wn = min(cutoff, 0.95 * nyq) / nyq
    sos = signal.butter(4, wn, btype="low", output="sos")
    return _sos_filtfilt(x, sos)


def resp_baseline_subtract(x: np.ndarray, fs: float) -> np.ndarray:
    """Low-pass at 1 Hz then remove 60 s moving-median baseline drift."""
    lp = lowpass(x, fs, 1.0)
    win = max(1, int(60 * fs))
    if win % 2 == 0:
        win += 1
    if win < len(lp):
        baseline = signal.medfilt(lp, kernel_size=win)
    else:
        baseline = np.full_like(lp, np.median(lp))
    return lp - baseline


# --------------------------------------------------------------------------- #
# Quality flags (per 1-hour window)
# --------------------------------------------------------------------------- #
def _window_view(x: np.ndarray, n: int):
    k = len(x) // n
    if k == 0:
        return None
    return x[:k * n].reshape(k, n)


def ecg_flatline(x: np.ndarray, fs: float) -> bool:
    w = _window_view(x, max(1, int(10 * fs)))
    if w is None:
        return True
    frac = np.mean(w.std(axis=1) < 0.01)
    return bool(frac > 0.30)


def pleth_motion(x: np.ndarray, fs: float) -> bool:
    w = _window_view(x, max(1, int(10 * fs)))
    if w is None:
        return True
    k = kurtosis(w, axis=1, fisher=False, bias=False)
    k = np.nan_to_num(k, nan=0.0)
    return bool(np.mean(k > 10) > 0.20)


def resp_disconnected(x: np.ndarray, fs: float) -> bool:
    seg = max(1, int(60 * fs))
    w = _window_view(x, seg)
    if w is None:
        return True
    return bool(np.any(w.std(axis=1) < 1e-6))


@dataclass
class ProcessedChannel:
    signal: np.ndarray
    valid_fraction: float    # fraction of the window that was finite (pre-fill)
    quality_ok: bool         # passes the channel's quality flag


def preprocess_hour(channel: str, x: np.ndarray, fs: float) -> ProcessedChannel:
    """Filter a 1-hour window of one channel and compute its quality flag."""
    filled, finite = fill_nans(x)
    valid_fraction = float(finite.mean()) if len(finite) else 0.0
    if channel == "ECG":
        out = bandpass_ecg(filled, fs)
        ok = not ecg_flatline(out, fs)
    elif channel == "Pleth":
        out = lowpass(filled, fs, 10.0)
        ok = not pleth_motion(out, fs)
    elif channel == "Resp":
        out = resp_baseline_subtract(filled, fs)
        ok = not resp_disconnected(filled, fs)
    else:
        raise ValueError(f"unknown channel {channel}")
    return ProcessedChannel(signal=out, valid_fraction=valid_fraction, quality_ok=ok)
