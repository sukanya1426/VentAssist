"""Waveform stage 4.6 — Respiration features from the Resp belt.

Zero-crossing breath detection on the baseline-corrected Resp signal, then:
  RRV                  : coefficient of variation of breath durations
  Breathing_Regularity : 1 - RRV (clipped to [0, 1])
  Asynchrony_Score     : additive heuristic in [0, 1]

Returns NaN for all features if fewer than 10 valid breaths are detected.
"""

from __future__ import annotations

import numpy as np


def _zero_crossings(x: np.ndarray, positive_to_negative: bool) -> np.ndarray:
    s = np.sign(x)
    s[s == 0] = 1
    if positive_to_negative:
        idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0]
    else:
        idx = np.where((s[:-1] < 0) & (s[1:] > 0))[0]
    return idx


def resp_features(resp: np.ndarray, fs: float) -> dict[str, float]:
    nan = {"RRV": np.nan, "Breathing_Regularity": np.nan, "Asynchrony_Score": np.nan}
    if len(resp) < int(5 * fs):
        return nan

    # breath = interval between consecutive positive→negative crossings (expiry start)
    pn = _zero_crossings(resp, positive_to_negative=True)
    if len(pn) < 11:
        return nan
    durations = np.diff(pn) / fs
    durations = durations[(durations >= 1.5) & (durations <= 12.0)]
    if len(durations) < 10:
        return nan

    mean_dur = float(np.mean(durations))
    rrv = float(np.std(durations) / mean_dur) if mean_dur > 0 else np.nan
    regularity = float(np.clip(1.0 - rrv, 0.0, 1.0))

    # I:E ratio from inspiration vs expiration durations.
    npos = _zero_crossings(resp, positive_to_negative=False)
    ie_ratio = _ie_ratio(pn, npos, fs)

    # additive asynchrony heuristic
    score = 0.0
    if rrv > 0.25:
        score += 0.4
    if _has_consecutive_short(durations, mean_dur, k=3, frac=0.6):
        score += 0.3
    if ie_ratio is not None and ie_ratio > 1.5:
        score += 0.3
    asynchrony = float(min(score, 1.0))

    return {"RRV": rrv, "Breathing_Regularity": regularity,
            "Asynchrony_Score": asynchrony}


def _has_consecutive_short(durations: np.ndarray, mean_dur: float,
                           k: int, frac: float) -> bool:
    short = durations < frac * mean_dur
    run = 0
    for s in short:
        run = run + 1 if s else 0
        if run >= k:
            return True
    return False


def _ie_ratio(pos_to_neg: np.ndarray, neg_to_pos: np.ndarray,
              fs: float) -> float | None:
    """Mean inspiration/expiration duration ratio from alternating crossings."""
    insp, exp = [], []
    # inspiration: negative→positive ... to next positive→negative
    for nz in neg_to_pos:
        nxt = pos_to_neg[pos_to_neg > nz]
        if len(nxt):
            insp.append((nxt[0] - nz) / fs)
    for pz in pos_to_neg:
        nxt = neg_to_pos[neg_to_pos > pz]
        if len(nxt):
            exp.append((nxt[0] - pz) / fs)
    if not insp or not exp:
        return None
    mi, me = np.median(insp), np.median(exp)
    return float(mi / me) if me > 0 else None
