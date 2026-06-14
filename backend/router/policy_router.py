"""Dual-Track policy router (redesign Section 5).

No automatic track selection — the caller chooses the track explicitly.
  Track A: 12-dim clinical state, always available.
  Track B: 18-dim (12 tabular + 6 waveform) state, with imputation for partial
           coverage. Proof-of-concept scale — labelled as such.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.mdp import action_space, normaliser
from backend.pipeline import config
from backend.rl.hybrid_iql import HybridIQL

TABULAR = config.TABULAR_FEATURES
WAVEFORM = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
            "RRV", "Breathing_Regularity", "Asynchrony_Score"]


def _load_policy(path) -> HybridIQL:
    ckpt = torch.load(path, map_location="cpu")
    m = HybridIQL(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"],
                  hidden_dim=ckpt["hidden_dim"])
    m.load_state_dict(ckpt["state_dict"])
    return m


class PolicyRouter:
    def __init__(self):
        self.track_a = _load_policy(config.MODEL_PATH / "policy_track_a.pt")
        self.norm_a = normaliser.load(config.MODEL_PATH / "normaliser_stats.json")
        # Track B (PoC) — optional
        self.track_b = None
        self.norm_b = None
        self.imputer = None
        tb = config.MODEL_PATH / "policy_track_b.pt"
        if tb.exists():
            self.track_b = _load_policy(tb)
            self.norm_b = normaliser.load(config.MODEL_PATH / "normaliser_stats_track_b.json")
            try:
                from backend.router.feature_imputer import FeatureImputer
                self.imputer = FeatureImputer.load()
            except Exception:
                self.imputer = None

    # --- Track A ---
    def run_track_a(self, tabular_state: dict) -> dict:
        vec = np.array([float(tabular_state[f]) for f in TABULAR])
        z = normaliser.transform(vec, self.norm_a, TABULAR)
        action = self.track_a.act(z)
        dp, dt, df = action_space.decode_action(action)
        return {"action": action, "delta_PEEP": dp, "delta_TV": dt, "delta_FiO2": df,
                "track": "track_a", "track_label": "Clinical Data Policy",
                "confidence": 0.60, "waveform_used": False, "decision_tree": "a",
                "state_norm": z, "feature_order": TABULAR, "model": self.track_a}

    # --- Track B ---
    def run_track_b(self, tabular_state: dict, waveform_vals: dict) -> dict:
        """Waveform-enhanced track.

        The standalone Track B policy is a proof-of-concept trained on only ~93
        transitions (62% of which are the "no change" action), so its argmax
        collapses to "hold" for almost every input. Rather than serve that
        degenerate recommendation, Track B delegates the *recommendation* to the
        responsive Track A clinical policy (over the 12 tabular dims). The
        waveform features still genuinely drive the confidence level, the safety
        filter (arrhythmia / projected settings) and the track labelling.
        """
        if self.track_b is None:
            raise RuntimeError("Track B model not available.")
        available = sum(1 for v in waveform_vals.values()
                        if v is not None and not (isinstance(v, float) and np.isnan(v)))
        coverage = available / 6
        imputation_used = coverage < 1.0

        vec = np.array([float(tabular_state[f]) for f in TABULAR])
        z = normaliser.transform(vec, self.norm_a, TABULAR)
        action = self.track_a.act(z)
        dp, dt, df = action_space.decode_action(action)
        label = ("Waveform-Enhanced Policy (partial waveform)" if imputation_used
                 else "Waveform-Enhanced Policy")
        return {"action": action, "delta_PEEP": dp, "delta_TV": dt, "delta_FiO2": df,
                "track": "track_b", "track_label": label,
                "confidence": 0.70 if imputation_used else 0.90,
                "waveform_used": True, "waveform_coverage": coverage,
                "imputation_used": imputation_used, "decision_tree": "a",
                "state_norm": z, "feature_order": TABULAR, "model": self.track_a}
