"""Vectorised lung-protective rules, shared by evaluation and model selection.

``safety_filter.py`` beside this file enforces the same clinical rules at serving
time, one recommendation at a time, on a state dict, with thresholds overridable
by environment variable. That shape is right for the API and wrong for scoring a
million transitions, so the array form lived in ``benchmark/safety_metrics.py``
instead — which left ``backend/`` unable to check a violation rate without
importing ``benchmark/``, and ``benchmark/`` is a read-only consumer of
``backend/`` by convention (it must never be a dependency of the serving path).

So the array form lives here, and ``benchmark/safety_metrics.py`` imports it. The
thresholds are module constants rather than environment reads on purpose: a
published violation rate must not change because a deployment set ``MAX_PEEP``,
and ``safety_metrics``' canonical artifact is quoted in the report.

Rules (the ones a 12-dimensional tabular state can support):
  * VT > 8 mL/kg PBW — volutrauma (ARDSnet targets 6–8)
  * PEEP > 15 cmH₂O — over-distension risk
  * FiO₂ > 0.8 — oxygen toxicity
  * FiO₂ ≥ 0.95 while SpO₂ ≥ 96% — needless hyperoxia
  * PEEP below the ARDSnet minimum for the given FiO₂ — buying oxygenation with
    FiO₂ instead of recruitment

Driving pressure > 15 cmH₂O is NOT checked: it needs the plateau pressure, which
this state does not carry. It is reported as ``None`` rather than imputed.
"""

from __future__ import annotations

import numpy as np

MAX_TV_ML_PER_KG = 8.0
MAX_PEEP = 15.0
MAX_FIO2 = 0.8
HYPEROXIA_SPO2 = 96.0
HYPEROXIA_FIO2 = 0.95

# ARDSnet lower-PEEP/higher-FiO₂ table: the MINIMUM PEEP expected at each FiO₂.
_ARDSNET = [(0.30, 5), (0.40, 5), (0.50, 8), (0.60, 10),
            (0.70, 10), (0.80, 14), (0.90, 14), (1.00, 18)]


def ardsnet_min_peep(fio2: np.ndarray) -> np.ndarray:
    """Minimum ARDSnet PEEP for each FiO₂ (step function, vectorised)."""
    out = np.full(len(fio2), 5.0)
    for f, p in _ARDSNET:
        out = np.where(fio2 >= f - 1e-9, float(p), out)
    return out


def resulting_settings(peep, tv, fio2, actions):
    """Apply the (ΔPEEP, ΔTV, ΔFiO₂) actions to the current settings."""
    from backend.mdp import action_space

    am = action_space.ACTION_MAP
    dp = np.array([am[int(a)][0] for a in actions], dtype=float)
    dt = np.array([am[int(a)][1] for a in actions], dtype=float)
    df = np.array([am[int(a)][2] for a in actions], dtype=float)
    new_fio2 = np.clip(fio2 + df, action_space.FIO2_MIN, action_space.FIO2_MAX)
    return peep + dp, tv + dt, new_fio2


def violations(peep, tv, fio2, spo2, weight_kg) -> dict:
    """Violation rate for each rule over the resulting settings."""
    tv_per_kg = tv / np.maximum(weight_kg, 1.0)
    min_peep = ardsnet_min_peep(fio2)
    rules = {
        "volutrauma_tv_gt_8ml_per_kg": tv_per_kg > MAX_TV_ML_PER_KG,
        "peep_gt_15": peep > MAX_PEEP,
        "fio2_gt_0.8": fio2 > MAX_FIO2,
        "needless_hyperoxia_fio2_ge_0.95_and_spo2_ge_96": (fio2 >= HYPEROXIA_FIO2)
                                                          & (spo2 >= HYPEROXIA_SPO2),
        "peep_below_ardsnet_min_for_fio2": peep < min_peep,
    }
    out = {k: round(float(np.mean(v)), 4) for k, v in rules.items()}
    out["any_violation"] = round(float(np.mean(np.any(np.stack(list(rules.values())), axis=0))), 4)
    # Needs airway-pressure waveform — stated, not imputed.
    out["driving_pressure_gt_15"] = None
    return out
