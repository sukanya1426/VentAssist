"""Distil a HybridIQL policy into a shallow CART tree for readable rules (9.2).

The tree is fit on **normalised** (z-scored) states, so its split thresholds are
in z-score units. Those are meaningless to a clinician, so ``decision_rule``
denormalises each threshold back to raw physiological units (mean + z*std, the
inverse of the z-score; the winsor clip is not reversible) and appends the
feature's unit.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from backend.mdp import action_space

# Raw-unit suffixes for the human-readable decision rule.
FEATURE_UNITS = {
    "PEEP": " cmH₂O", "TV": " mL", "FiO2": "",
    "SpO2": "%", "PaO2": " mmHg", "PaCO2": " mmHg", "pH": "",
    "HR": " bpm", "SBP": " mmHg", "RR": " /min", "RASS": "", "Temp": " °C",
    "HRV_SDNN": " ms", "Arrhythmia_rate": "", "Perfusion_Index": "%",
    "RRV": "", "Breathing_Regularity": "", "Asynchrony_Score": "",
}
# FiO₂ reads better with 2 decimals; default is 1.
_PRECISION = {"FiO2": 2, "pH": 2}


class DistilledTree:
    def __init__(self, tree: DecisionTreeClassifier, feature_order: list[str],
                 norm_stats: dict | None = None):
        self.tree = tree
        self.feature_order = feature_order
        self.norm_stats = norm_stats or {}

    @classmethod
    def fit(cls, model, states_norm: np.ndarray, feature_order: list[str],
            norm_stats: dict | None = None, max_depth: int = 4) -> "DistilledTree":
        actions = np.array([model.act(s) for s in states_norm])
        # guard: need >= 2 classes
        if len(np.unique(actions)) < 2:
            actions[0] = (actions[0] + 1) % action_space.N_ACTIONS
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(states_norm, actions)
        return cls(tree, feature_order, norm_stats)

    def _denormalise(self, feature: str, z_value: float) -> float:
        """z-score threshold → raw physiological value (inverse z-score)."""
        s = self.norm_stats.get(feature)
        if not s:
            return z_value
        return z_value * s["std"] + s["mean"]

    def _fmt_threshold(self, feature: str, z_value: float) -> str:
        raw = self._denormalise(feature, z_value)
        prec = _PRECISION.get(feature, 1)
        return f"{raw:.{prec}f}{FEATURE_UNITS.get(feature, '')}"

    def decision_rule(self, state_norm: np.ndarray) -> str:
        t = self.tree.tree_
        node, conds = 0, []
        x = state_norm.reshape(1, -1)
        while t.children_left[node] != t.children_right[node]:
            f = t.feature[node]
            thr = t.threshold[node]
            name = self.feature_order[f]
            thr_txt = self._fmt_threshold(name, thr)
            if x[0, f] <= thr:
                conds.append(f"{name} ≤ {thr_txt}"); node = t.children_left[node]
            else:
                conds.append(f"{name} > {thr_txt}"); node = t.children_right[node]
        action = int(self.tree.classes_[t.value[node].argmax()])
        dp, dt, df = action_space.decode_action(action)
        cond_txt = " AND ".join(conds[:4]) if conds else "default"
        return (f"IF {cond_txt} → ΔPEEP {dp:+d} cmH₂O, "
                f"ΔTV {dt:+d} mL, ΔFiO₂ {df:+.2f}")
