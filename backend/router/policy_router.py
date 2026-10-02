"""Dual-Track policy router (redesign Section 5).

No automatic track selection — the caller chooses the track explicitly.
  Track A: 12-dim clinical state, always available.
  Track B: 18-dim (12 tabular + 6 waveform) state, with imputation for partial
           coverage. Proof-of-concept scale — labelled as such.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.mdp import action_space, action_masking, normaliser
from backend.pipeline import config
from backend.rl.hybrid_iql import HybridIQL

TABULAR = config.TABULAR_FEATURES
WAVEFORM = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
            "RRV", "Breathing_Regularity", "Asynchrony_Score"]

# Maximum Q-margin (raw units, reward scale ~[-43, +42]) below which an
# alternative action is "close enough" to the argmax to surface to the user.
# Above this the next-best action is clearly worse and showing it would mislead.
ALT_MARGIN_THRESHOLD = 0.5

# Information-content ceiling per data source. A recommendation grounded in
# waveform data can, at most, be more confident than one from clinical numbers
# alone — this caps the confidence so the clinical-only track never claims
# near-certainty. Track B interpolates toward CONF_WEIGHT_WAVEFORM_FULL by
# waveform coverage.
CONF_WEIGHT_CLINICAL = 0.85        # Track A ceiling (clinical data only)
CONF_WEIGHT_WAVEFORM_FULL = 1.00   # Track B ceiling with full waveform coverage
CONF_FLOOR, CONF_CEIL = 0.30, 0.97

# The "hold" action (no change). The responsiveness knob (§16 item 9) trades
# hold-vs-act by adding a bonus to every NON-hold action before the argmax.
HOLD_ACTION = action_space.encode_action(0, 0, 0.0)
# Q-units added to non-hold actions at responsiveness=1.0. Sized (empirically,
# on the deployed model) so responsiveness ~0.2 already unseats a narrowly-held
# "hold" while a comfortably-held stable patient needs a higher setting. Because
# the SAME bonus is added to all non-hold actions, it never changes WHICH action
# is chosen when the policy acts — only the hold-vs-act threshold. Default 0.0 is
# the identity (the deployed/gated behaviour).
RESPONSIVENESS_SCALE = 2.0


def _rank_actions(q: np.ndarray, chosen: int, k: int = 3,
                  margin_threshold: float = ALT_MARGIN_THRESHOLD) -> dict:
    """Top-k actions with margins, measured from the action actually served.

    ``q`` must be the **raw** Q (restricted to allowed actions), never the
    responsiveness-biased one — the margins are what the UI shows a clinician, so
    they have to describe the policy's genuine preference ordering, not the
    ordering after an operator-set bias was folded in. Ranking on the biased Q
    inflated every margin by the bonus and pushed "hold" out of the top-k exactly
    when responsiveness had overridden it, hiding from the clinician that the
    policy had preferred to do nothing.

    ``chosen`` (the served action) always leads the list at rank 0, so
    ``margin_from_best`` reads "how much better than this is the recommendation
    you were given". When responsiveness forced a non-argmax action the entry the
    policy actually preferred appears with a **negative** margin — that is the
    signal, not a glitch.

    At ``responsiveness == 0`` ``chosen`` is the argmax of ``q``, so this is
    identical to ranking by Q with margins from the best (the gated behaviour).
    """
    finite = [int(i) for i in np.argsort(q)[::-1] if np.isfinite(q[int(i)])]
    order = finite[:k]
    if chosen in order:
        order.remove(chosen)
    order = [chosen] + order[:max(0, k - 1)]
    chosen_q = float(q[chosen])
    ranked = []
    for rank, idx in enumerate(order):
        dp, dt, df = action_space.decode_action(idx)
        ranked.append({
            "rank": rank, "action_idx": idx,
            "delta_PEEP": dp, "delta_TV": dt, "delta_FiO2": df,
            "q_value": float(q[idx]),
            "margin_from_best": chosen_q - float(q[idx]),
        })
    alternatives = [r for r in ranked[1:]
                    if r["margin_from_best"] < margin_threshold]
    return {"ranked": ranked, "alternatives": alternatives}


def _apply_responsiveness(q: np.ndarray, responsiveness: float) -> np.ndarray:
    """Bias the Q-values toward acting by ``responsiveness`` ∈ [0, 1].

    Adds ``responsiveness * RESPONSIVENESS_SCALE`` to every NON-hold action, so a
    downstream ``argmax`` leaves "hold" only when some change beats it by less
    than that bonus. The bonus is identical across all non-hold actions, so it
    changes *whether* the policy acts, never *which* action it picks when it does.
    ``responsiveness == 0`` returns an unchanged copy (the deployed behaviour).
    """
    if responsiveness <= 0:
        return q.copy()
    qe = q.copy()
    mask = np.arange(len(q)) != HOLD_ACTION
    qe[mask] += float(responsiveness) * RESPONSIVENESS_SCALE
    return qe


def _ood_weight(base_weight: float, ood: dict | None) -> float:
    """Temper the confidence information-weight by learned support (§16 item 7).

    An in-support state keeps the full weight; a fully out-of-distribution state
    (support_ratio → 0) halves it, so OOD recommendations read as less confident.
    No detector → unchanged."""
    if ood is None:
        return base_weight
    return base_weight * (0.5 + 0.5 * float(ood["support_ratio"]))


def _select_action(q_raw: np.ndarray, responsiveness: float,
                   ventilation_mode: str | None) -> tuple:
    """Pick the action given the raw Q, responsiveness bias, and ventilation mode.

    Returns (action, q_eff, q_conf, masked): ``q_eff`` is the ranking/selection Q
    (responsiveness-biased, disallowed actions set to -inf), ``q_conf`` is the raw
    Q restricted to allowed actions (for an honest confidence among clinically
    valid choices). Never mutates ``q_raw``."""
    mask = action_masking.mode_action_mask(ventilation_mode)     # (125,) bool
    masked = not mask.all()
    q_eff = _apply_responsiveness(q_raw, responsiveness)
    if masked:
        q_eff = np.where(mask, q_eff, -np.inf)
        q_conf = np.where(mask, q_raw, -np.inf)
    else:
        q_conf = q_raw
    action = int(np.argmax(q_eff))
    return action, q_eff, q_conf, masked


def _decision_confidence(q: np.ndarray, info_weight: float,
                         chosen: int | None = None,
                         temp: float = ALT_MARGIN_THRESHOLD) -> dict:
    """Real, per-recommendation confidence derived from the policy's Q-values.

    Replaces the old hard-coded per-track constant (which made the UI always
    show the same number regardless of the patient). Two interpretable factors:

    * **Decision sharpness** — how decisively the chosen action beats its closest
      competitor. Under a Boltzmann policy with temperature ``temp`` (set to
      ``ALT_MARGIN_THRESHOLD`` so it shares the alternatives' scale), this is
      ``sigmoid(margin / temp)`` where ``margin`` is the chosen action's Q minus
      the best *other* action's Q. When ``chosen`` is the argmax this is the
      best-minus-second-best gap (>0). When a responsiveness override picks a
      non-argmax action, ``margin`` goes negative → sub-0.5 sharpness, honestly
      flagging that the policy's own value function preferred to hold.
    * **Information content** (``info_weight``) — the data the decision stands
      on: clinical-only caps lower than clinical+waveform (see the CONF_WEIGHT_*
      constants). This is the honest remnant of the old per-track heuristic.

    Confidence is always computed on the RAW Q-values (not the responsiveness-
    biased ones) so the number reflects the policy's genuine certainty about the
    action it is recommending. confidence = clip(info_weight * sharpness, floor,
    ceil).
    """
    if chosen is None:
        chosen = int(np.argmax(q))
    if q.size > 1:
        others = q[np.arange(q.size) != chosen]
        margin = float(q[chosen] - others.max())
    else:
        margin = float("inf")
    sharpness = 1.0 / (1.0 + np.exp(-margin / temp))       # (0, 1)
    confidence = float(np.clip(info_weight * sharpness, CONF_FLOOR, CONF_CEIL))
    return {"confidence": round(confidence, 3),
            "decision_margin": round(margin, 4),
            "sharpness": round(float(sharpness), 4)}


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
        # Learned OOD/support detector (§16 item 7) — optional.
        self.ood = None
        try:
            from backend.rl.ood_autoencoder import OODDetector
            self.ood = OODDetector.load("a")
        except Exception:
            self.ood = None
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

    def _ood(self, z) -> dict | None:
        """Learned support signal for a normalised state (None if no detector)."""
        return None if self.ood is None else self.ood.evaluate(z)

    # --- Track A ---
    def run_track_a(self, tabular_state: dict, responsiveness: float = 0.0,
                    ventilation_mode: str | None = None) -> dict:
        vec = np.array([float(tabular_state[f]) for f in TABULAR])
        # SERVING transform — z-score + z-space clamp, NOT the training winsor clip,
        # which would collapse every SpO₂ < 89 / PEEP > 18 / TV > 863 onto one vector
        # and make the recommendation unresponsive to the clinician's edits.
        z = normaliser.transform_inference(vec, self.norm_a, TABULAR)
        q = self.track_a.q_values(z)                        # raw Q, shape (125,)
        action, q_eff, q_conf, masked = _select_action(q, responsiveness, ventilation_mode)
        dp, dt, df = action_space.decode_action(action)
        # Rank on the RAW (allowed-only) Q, not q_eff — see _rank_actions.
        ranking = _rank_actions(q_conf, action)
        ood = self._ood(z)
        conf = _decision_confidence(q_conf, _ood_weight(CONF_WEIGHT_CLINICAL, ood),
                                    chosen=action)
        return {"action": action, "delta_PEEP": dp, "delta_TV": dt, "delta_FiO2": df,
                "track": "track_a", "track_label": "Clinical Data Policy",
                "confidence": conf["confidence"],
                "decision_margin": conf["decision_margin"],
                "responsiveness": float(responsiveness),
                "ventilation_mode": action_masking.classify_ventilator_mode(ventilation_mode),
                "tv_masked": bool(masked),
                "in_support": (ood["in_support"] if ood else None),
                "support_ratio": (round(ood["support_ratio"], 3) if ood else None),
                "waveform_used": False, "decision_tree": "a",
                "state_norm": z, "feature_order": TABULAR, "model": self.track_a,
                "ranked": ranking["ranked"], "alternatives": ranking["alternatives"]}

    # --- Track B ---
    def run_track_b(self, tabular_state: dict, waveform_vals: dict,
                    responsiveness: float = 0.0,
                    ventilation_mode: str | None = None) -> dict:
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
        # SERVING transform — z-score + z-space clamp, NOT the training winsor clip,
        # which would collapse every SpO₂ < 89 / PEEP > 18 / TV > 863 onto one vector
        # and make the recommendation unresponsive to the clinician's edits.
        z = normaliser.transform_inference(vec, self.norm_a, TABULAR)
        q = self.track_a.q_values(z)                        # raw Q, shape (125,)
        action, q_eff, q_conf, masked = _select_action(q, responsiveness, ventilation_mode)
        dp, dt, df = action_space.decode_action(action)
        # Rank on the RAW (allowed-only) Q, not q_eff — see _rank_actions.
        ranking = _rank_actions(q_conf, action)
        # Waveform coverage raises the information-content ceiling from the
        # clinical-only cap up to the full-waveform cap; the decision sharpness
        # (the Q-margin) is the same because Track B delegates to Track A.
        info_weight = (CONF_WEIGHT_CLINICAL
                       + (CONF_WEIGHT_WAVEFORM_FULL - CONF_WEIGHT_CLINICAL) * coverage)
        ood = self._ood(z)
        conf = _decision_confidence(q_conf, _ood_weight(info_weight, ood), chosen=action)
        label = ("Waveform-Enhanced Policy (partial waveform)" if imputation_used
                 else "Waveform-Enhanced Policy")
        return {"action": action, "delta_PEEP": dp, "delta_TV": dt, "delta_FiO2": df,
                "track": "track_b", "track_label": label,
                "confidence": conf["confidence"],
                "decision_margin": conf["decision_margin"],
                "responsiveness": float(responsiveness),
                "ventilation_mode": action_masking.classify_ventilator_mode(ventilation_mode),
                "tv_masked": bool(masked),
                "in_support": (ood["in_support"] if ood else None),
                "support_ratio": (round(ood["support_ratio"], 3) if ood else None),
                "waveform_used": True, "waveform_coverage": coverage,
                "imputation_used": imputation_used, "decision_tree": "a",
                "state_norm": z, "feature_order": TABULAR, "model": self.track_a,
                "ranked": ranking["ranked"], "alternatives": ranking["alternatives"]}
