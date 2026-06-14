"""MDP reward functions (Section 6 of the methodology).

IMPORTANT: rewards are computed on RAW (un-normalised) physiological values.
``tabular_states.parquet`` stores raw values for exactly this reason, so rewards
must be computed before any normalisation.

TIER 1 (12-dim) reward:
    r = C1 * f_SpO2(SpO2_next) - lam1 * 1[TV > 8 mL/kg] - C2 * 1[FiO2 > 0.8 or PEEP > 15]
    f_SpO2(v) = (v - 95) / 7   for 88 <= v <= 95, else 0
                (zero outside the hypoxaemic band — avoids hyperoxia incentives)

TIER 2 (18-dim) augmented reward adds waveform-instability penalties:
    r2 = r - lam2 * Asynchrony_Score - lam3 * 1[Arrhythmia_rate > 0.2]
"""

from __future__ import annotations

from typing import Mapping


def f_spo2(v: float) -> float:
    """Oxygenation shaping term, active only in the hypoxaemic band [88, 95]."""
    if 88.0 <= v <= 95.0:
        return (v - 95.0) / 7.0
    return 0.0


def tier1_reward(s_t: Mapping[str, float], s_next: Mapping[str, float],
                 patient_weight_kg: float,
                 C1: float = 1.0, C2: float = 0.5, lam1: float = 0.3) -> float:
    """TIER 1 composite reward on raw physiological values."""
    spo2_gain = f_spo2(float(s_next["SpO2"]))

    tv = float(s_t["TV"])
    tv_penalty = lam1 if tv > 8.0 * patient_weight_kg else 0.0

    agg = (float(s_t["FiO2"]) > 0.8) or (float(s_t["PEEP"]) > 15.0)
    agg_penalty = C2 if agg else 0.0

    return C1 * spo2_gain - tv_penalty - agg_penalty


def tier2_reward(s_t: Mapping[str, float], s_next: Mapping[str, float],
                 patient_weight_kg: float,
                 C1: float = 1.0, C2: float = 0.5, lam1: float = 0.3,
                 lam2: float = 0.2, lam3: float = 0.15) -> float:
    """TIER 2 reward = TIER 1 + waveform-instability penalties.

    Missing waveform features default to 0 (no penalty).
    """
    base = tier1_reward(s_t, s_next, patient_weight_kg, C1=C1, C2=C2, lam1=lam1)
    async_score = float(s_t.get("Asynchrony_Score", 0.0) or 0.0)
    arr_rate = float(s_t.get("Arrhythmia_rate", 0.0) or 0.0)
    async_penalty = lam2 * async_score
    arr_penalty = lam3 if arr_rate > 0.2 else 0.0
    return base - async_penalty - arr_penalty
