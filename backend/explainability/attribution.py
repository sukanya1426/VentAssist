"""Lightweight explainability (shap-free).

A permutation/perturbation attribution over the policy's Q(s, a_recommended):
each feature is nudged by ±1 std (in normalised space) and the change in the
recommended action's Q-value is the attribution. Avoids the heavy ``shap``
dependency while giving the same top-3-driver UX. A distilled CART tree provides
a human-readable decision rule.
"""

from __future__ import annotations

import numpy as np

_DISPLAY = {
    "PEEP": "PEEP", "TV": "Tidal volume", "FiO2": "FiO₂", "SpO2": "SpO₂",
    "PaO2": "PaO₂", "PaCO2": "PaCO₂", "pH": "pH", "HR": "Heart rate",
    "SBP": "Systolic BP", "RR": "Resp. rate", "RASS": "RASS", "Temp": "Temperature",
    "HRV_SDNN": "HRV (SDNN)", "Arrhythmia_rate": "Arrhythmia rate",
    "Perfusion_Index": "Perfusion index", "RRV": "Resp. rate variability",
    "Breathing_Regularity": "Breathing regularity", "Asynchrony_Score": "Asynchrony",
}


def top_features(model, state_norm: np.ndarray, action: int,
                 feature_order: list[str], k: int = 3) -> list[dict]:
    """Return the top-k features driving Q(state, action), by perturbation."""
    base_q = model.q_values(state_norm)[action]
    attrs = []
    for j, feat in enumerate(feature_order):
        up = state_norm.copy(); up[j] += 1.0
        dn = state_norm.copy(); dn[j] -= 1.0
        dq = (model.q_values(up)[action] - model.q_values(dn)[action]) / 2.0
        attrs.append((feat, float(dq)))
    attrs.sort(key=lambda x: abs(x[1]), reverse=True)
    out = []
    for feat, val in attrs[:k]:
        direction = "up" if val > 0 else "down"
        out.append({
            "feature": feat,
            "shap_value": round(val, 4),
            "direction": direction,
            "display": f"{_DISPLAY.get(feat, feat)} "
                       f"{'increases' if val > 0 else 'decreases'} the recommendation's value",
        })
    return out
