"""Discrete action space (Section 5.1).

Clinician actions are inferred from consecutive hourly changes in ventilator
settings:
    ΔPEEP ∈ {-2,-1,0,+1,+2} cmH2O          (5 levels, peep_idx 0–4)
    ΔTV   ∈ {-50,-25,0,+25,+50} mL          (5 levels, tv_idx   0–4)
    ΔFiO₂ ∈ {-0.10,-0.05,0,+0.05,+0.10}     (5 levels, fio2_idx 0–4)
    ⇒ |A| = 125 discrete actions, indexed 0–124,
       action = peep_idx*25 + tv_idx*5 + fio2_idx.

NOTE: the design prompt called this a "75-action" space, but 5×5×5 = 125 and the
prompt's ``*15`` stride overlaps the PEEP blocks (it only partitions cleanly with
3 TV levels). All three stated delta dimensions have 5 levels, so the correct
bijective encoding uses a stride of 25 over 125 actions.

FiO₂ steps of ±0.05/±0.10 mirror how clinicians titrate oxygen in practice
(e.g. 0.40 → 0.45 → 0.50). Rare actions (observed < 10 times in the training
set) are remapped to the nearest valid action by L1 distance in
(ΔPEEP, ΔTV, ΔFiO₂) space. The full 75-entry map is persisted to
``models/action_map.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.pipeline import config

DELTA_PEEP_BINS = [-2, -1, 0, 1, 2]
DELTA_TV_BINS = [-50, -25, 0, 25, 50]
DELTA_FIO2_BINS = [-0.10, -0.05, 0.0, 0.05, 0.10]
N_ACTIONS = len(DELTA_PEEP_BINS) * len(DELTA_TV_BINS) * len(DELTA_FIO2_BINS)  # 125

# Index strides: action = peep_idx*_PEEP_STRIDE + tv_idx*_TV_STRIDE + fio2_idx
_TV_STRIDE = len(DELTA_FIO2_BINS)                       # 5
_PEEP_STRIDE = len(DELTA_TV_BINS) * len(DELTA_FIO2_BINS)  # 25

# FiO₂ clip range when a delta is applied to a raw FiO₂ value.
FIO2_MIN, FIO2_MAX = 0.21, 1.0


def _nearest_idx(value: float, bins: list[float]) -> int:
    """Index of the bin nearest to `value` (L1)."""
    return int(np.argmin([abs(value - b) for b in bins]))


def _describe(dpeep: int, dtv: int, dfio2: float) -> str:
    def part(name, val, unit, prec=0):
        if val == 0:
            return f"{name} unchanged"
        num = f"{val:+.{prec}f}" if prec else f"{val:+d}"
        return f"{name} {num} {unit}".rstrip()
    return (f"{part('PEEP', dpeep, 'cmH2O')}, "
            f"{part('TV', dtv, 'mL')}, "
            f"{part('FiO2', dfio2, '', prec=2)}")


# Static 125-entry action map: idx -> (ΔPEEP, ΔTV, ΔFiO₂, description)
ACTION_MAP: dict[int, tuple[int, int, float, str]] = {
    p_i * _PEEP_STRIDE + t_i * _TV_STRIDE + f_i: (p, t, f, _describe(p, t, f))
    for p_i, p in enumerate(DELTA_PEEP_BINS)
    for t_i, t in enumerate(DELTA_TV_BINS)
    for f_i, f in enumerate(DELTA_FIO2_BINS)
}


def encode_action(dpeep_raw: float, dtv_raw: float, dfio2_raw: float) -> int:
    """Discretise a raw (ΔPEEP, ΔTV, ΔFiO₂) change to an action index 0–124."""
    peep_idx = _nearest_idx(dpeep_raw, DELTA_PEEP_BINS)
    tv_idx = _nearest_idx(dtv_raw, DELTA_TV_BINS)
    fio2_idx = _nearest_idx(dfio2_raw, DELTA_FIO2_BINS)
    return peep_idx * _PEEP_STRIDE + tv_idx * _TV_STRIDE + fio2_idx


def decode_action(idx: int) -> tuple[int, int, float]:
    """Action index 0–124 → (ΔPEEP cmH2O, ΔTV mL, ΔFiO₂)."""
    dpeep, dtv, dfio2, _ = ACTION_MAP[int(idx)]
    return dpeep, dtv, dfio2


def _nearest_valid(idx: int, valid: set[int]) -> int:
    """Closest valid action to `idx` by L1 distance in (ΔPEEP, ΔTV, ΔFiO₂) space."""
    if idx in valid:
        return idx
    dp, dt, df = decode_action(idx)
    best, best_d = idx, float("inf")
    for v in valid:
        vp, vt, vf = decode_action(v)
        # scale each step to unit distance before summing
        d = abs(vp - dp) + abs(vt - dt) / 25.0 + abs(vf - df) / 0.05
        if d < best_d:
            best, best_d = v, d
    return best


def remap_rare(actions: pd.Series, train_mask: np.ndarray | None = None,
               min_count: int = 10) -> tuple[pd.Series, set[int]]:
    """Replace actions seen < min_count times (in train) with nearest valid one.

    Args:
        actions: integer action indices for every transition.
        train_mask: boolean mask selecting training transitions for the frequency
            count (defaults to all transitions).
        min_count: minimum occurrences for an action to be considered valid.

    Returns:
        (remapped_actions, valid_action_set)
    """
    counts = actions[train_mask].value_counts() if train_mask is not None \
        else actions.value_counts()
    valid = set(int(a) for a, c in counts.items() if c >= min_count)
    if not valid:                      # degenerate (tiny sample) → keep all observed
        valid = set(int(a) for a in actions.unique())
    mapping = {a: _nearest_valid(int(a), valid) for a in actions.unique()}
    return actions.map(mapping), valid


def save_action_map(valid_actions: set[int] | None = None) -> None:
    """Write the 125-entry action map (with a validity flag) to models/action_map.json."""
    config.ensure_output_dirs()
    out = {
        str(idx): {
            "delta_PEEP_cmH2O": dp,
            "delta_TV_mL": dt,
            "delta_FiO2": df,
            "description": desc,
            "valid": (valid_actions is None) or (idx in valid_actions),
        }
        for idx, (dp, dt, df, desc) in ACTION_MAP.items()
    }
    (config.MODEL_PATH / "action_map.json").write_text(json.dumps(out, indent=2))


def load_action_map() -> dict:
    return json.loads((config.MODEL_PATH / "action_map.json").read_text())
