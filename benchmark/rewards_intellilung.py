"""Faithful port of IntelliLung's reward, for the reward critique (BENCHMARK_PLAN §0 trap 2).

Reference (`reward/range.py`, verbatim in BENCHMARK_HANDOFF §0.2):

    next_states = dataset.groupby(id)[keys].shift(-1).fillna(...)
    for key in keys:
        reward += (low <= next_states[key] <= high) * priority
    if normalize: reward /= sum(priorities)
    if time_penalty: reward = reward - 1

**The critical property: `reward` is computed ONLY from `next_states`.** The action
`a` never enters the expression. Conditional on the realised next state, the reward
is a constant — identical for all 125 actions. That is exactly the pathology
VentAssist diagnosed in SUMMARY §14.3 (action-independent reward → flat Q across
actions → policy collapse), and `action_causal_bonus` is our documented fix.

Range/priority table copied from `configs/pre_processing_configs.yaml`. We implement
the subset our 12-dim tabular state supports:

    blood_paco2 [28, 55]    priority 2   -> PaCO2   ✓
    blood_pao2  [55, 80]    priority 2   -> PaO2    ✓
    blood_ph    [7.3, 7.45] priority 2   -> pH      ✓
    vital_spo2  [88, 96]    priority 2   -> SpO2    ✓
    vital_hr    [70, 109]   priority 1   -> HR      ✓
    blood_sao2  [88, 96]    priority 2   -> (no SaO2 in our state)   OMITTED
    vent_etco2  [34, 45]    priority 2   -> (no EtCO2 in our state)  OMITTED
    vital_map   [60, 109]   priority 1   -> (we carry SBP, not MAP)  OMITTED

The omissions do not affect the critique: the reward's *action-independence* is
structural (it is a function of s′ alone), not a property of which features are in it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# feature -> (low, high, priority), from their config
RANGES: dict[str, tuple[float, float, int]] = {
    "PaCO2": (28.0, 55.0, 2),
    "PaO2": (55.0, 80.0, 2),
    "pH": (7.30, 7.45, 2),
    "SpO2": (88.0, 96.0, 2),
    "HR": (70.0, 109.0, 1),
}
OMITTED = {"blood_sao2": "no SaO2 in our state",
           "vent_etco2": "no EtCO2 in our state",
           "vital_map": "we carry SBP, not MAP"}


def range_reward(next_states_raw: np.ndarray, feature_order: Sequence[str],
                 normalize: bool = True, time_penalty: bool = True) -> np.ndarray:
    """IntelliLung RangeReward over RAW next-state values → (N,) in [-1, 0].

    NOTE the signature: it takes ONLY the next state. There is nowhere to pass an
    action, because the reference reward does not use one."""
    feats = list(feature_order)
    ns = np.asarray(next_states_raw, dtype=float)
    reward = np.zeros(len(ns), dtype=np.float64)
    scale = 0
    for feat, (low, high, prio) in RANGES.items():
        col = ns[:, feats.index(feat)]
        reward += np.logical_and(low <= col, col <= high) * prio
        scale += prio
    if normalize:
        reward /= scale
    if time_penalty:
        reward -= 1.0                       # every extra ventilated step costs 1
    return reward


def range_reward_scalar(s_next: dict, normalize: bool = True,
                        time_penalty: bool = True) -> float:
    """Single-transition version (dict of raw next-state values)."""
    reward, scale = 0.0, 0
    for feat, (low, high, prio) in RANGES.items():
        if low <= float(s_next[feat]) <= high:
            reward += prio
        scale += prio
    if normalize:
        reward /= scale
    if time_penalty:
        reward -= 1.0
    return float(reward)
