"""MDP reward functions (Section 6 of the methodology).

IMPORTANT: rewards are computed on RAW (un-normalised) physiological values.
``tabular_states.parquet`` stores raw values for exactly this reason, so rewards
must be computed before any normalisation.

ACTION-DEPENDENT design
-----------------------
An earlier reward depended almost entirely on ``s_next`` (next-state SpO2) and
current-state penalties, so it carried no signal about which *action* was good.
The learned Q-values were then nearly flat across actions and the greedy policy
collapsed (never "hold", always the same out-of-distribution action). The reward
below is explicitly a function of the (state, action, next_state) triple:

    r = w_oxy * Δoxygenation + w_vent * Δventilation
        - action_cost(ΔPEEP, ΔTV, ΔFiO2)        # discourage needless changes
        - safety_penalty(resulting settings)     # volutrauma / O2 toxicity / high PEEP

  * Δoxygenation = f_oxy(SpO2_next) − f_oxy(SpO2_t), where f_oxy peaks in the
    safe band [92, 96] and is negative for hypoxaemia (<92) or hyperoxia (>98).
    → rewards actions (FiO2 / PEEP) that move SpO2 toward the band.
  * Δventilation = f_vent(PaCO2_next) − f_vent(PaCO2_t), f_vent peaks near 40 mmHg.
    → rewards actions (TV / rate effect) that normalise PaCO2.
  * action_cost is a small per-step penalty proportional to the change magnitude,
    so when the patient is already in range the optimal action is "hold".

TIER 2 (18-dim) augments TIER 1 with waveform-instability penalties
(asynchrony, arrhythmia).
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Target bands
SPO2_LOW, SPO2_HIGH = 92.0, 96.0      # desired oxygenation band
SPO2_HYPEROXIA = 98.0                  # above this is wasteful/harmful
PACO2_TARGET = 40.0                    # normocapnia
PACO2_TOL = 5.0                        # tolerated deviation before penalty

# Reward weights
W_OXY = 1.0
W_VENT = 0.5

# Per-step action-change cost (in "unit steps": ΔPEEP per cmH2O, ΔTV per 25 mL,
# ΔFiO2 per 0.05). Small so a worthwhile correction outweighs it.
COST_PEEP = 0.10
COST_TV = 0.10
COST_FIO2 = 0.10

# Safety penalties on the RESULTING (next-state) settings.
VOLUTRAUMA_PEN = 0.5      # TV > 8 mL/kg
HYPEROXIA_PEN = 0.3       # FiO2 > 0.8
HIGH_PEEP_PEN = 0.3       # PEEP > 15

# Action-causal bonus weight (Section 14.4 fix). Roughly the scale of the
# per-step action cost so a clinically-indicated change is worth making.
LAM_CAUSAL = 0.4


def f_oxy(spo2: float) -> float:
    """Oxygenation desirability — 0 inside the safe band, negative outside."""
    if spo2 < SPO2_LOW:
        return -(SPO2_LOW - spo2)                # hypoxaemia (1 pt per % below 92)
    if spo2 > SPO2_HYPEROXIA:
        return -0.5 * (spo2 - SPO2_HYPEROXIA)    # hyperoxia (milder)
    return 0.0


def f_vent(paco2: float) -> float:
    """Ventilation desirability — 0 within ±tol of 40 mmHg, negative beyond."""
    dev = abs(paco2 - PACO2_TARGET)
    return -max(0.0, dev - PACO2_TOL) / 5.0


def _action_cost(action_tuple: Sequence[float]) -> float:
    dpeep, dtv, dfio2 = action_tuple
    return (COST_PEEP * abs(dpeep)
            + COST_TV * abs(dtv) / 25.0
            + COST_FIO2 * abs(dfio2) / 0.05)


def _safety_penalty(s_next: Mapping[str, float], patient_weight_kg: float) -> float:
    pen = 0.0
    if float(s_next["TV"]) > 8.0 * patient_weight_kg:
        pen += VOLUTRAUMA_PEN
    if float(s_next["FiO2"]) > 0.8:
        pen += HYPEROXIA_PEN
    if float(s_next["PEEP"]) > 15.0:
        pen += HIGH_PEEP_PEN
    return pen


def action_causal_bonus(s_t: Mapping[str, float], action_tuple: Sequence[float],
                        patient_weight_kg: float,
                        lam_causal: float = LAM_CAUSAL) -> float:
    """Reward the clinically-indicated action GIVEN the current physiology.

    The existing ``f_oxy``/``f_vent`` terms depend on ``s_next``, which in the
    offline data is correlated with patient severity rather than cleanly with the
    action taken (Section 14.3 confounding: hypoxaemic states are often already on
    high FiO2 being weaned). This term ties reward directly to (state, action), so
    "raise FiO2/PEEP when SpO2 is low and not yet maxed" is rewarded regardless of
    how the (confounded) next state turned out. This is the §14.4(2) fix.
    """
    dpeep, dtv, dfio2 = action_tuple
    bonus = 0.0

    spo2 = float(s_t["SpO2"])
    fio2 = float(s_t["FiO2"])
    paco2 = float(s_t["PaCO2"]) if "PaCO2" in s_t else PACO2_TARGET
    tv_per_kg = float(s_t["TV"]) / max(patient_weight_kg, 1.0)

    # Hypoxaemic and not yet maxed on FiO2/PEEP → reward raising either.
    if spo2 < SPO2_LOW and fio2 < 0.8:
        if dfio2 > 0 or dpeep > 0:
            bonus += lam_causal

    # Hyperoxic on non-trivial FiO2 → reward weaning oxygen.
    if spo2 > SPO2_HIGH and fio2 > 0.4:
        if dfio2 < 0:
            bonus += lam_causal

    # Hypercapnic with low tidal volume → reward raising TV (within safe limits).
    if paco2 > 50.0 and tv_per_kg < 8.0:
        if dtv > 0:
            bonus += lam_causal

    # Hypocapnic with high tidal volume → reward lowering TV.
    if paco2 < 35.0 and tv_per_kg > 6.0:
        if dtv < 0:
            bonus += lam_causal

    return bonus


def tier1_reward(s_t: Mapping[str, float], s_next: Mapping[str, float],
                 action_tuple: Sequence[float], patient_weight_kg: float,
                 w_oxy: float = W_OXY, w_vent: float = W_VENT,
                 lam_causal: float = LAM_CAUSAL) -> float:
    """TIER 1 action-dependent reward on raw physiological values."""
    d_oxy = f_oxy(float(s_next["SpO2"])) - f_oxy(float(s_t["SpO2"]))
    d_vent = (f_vent(float(s_next["PaCO2"])) - f_vent(float(s_t["PaCO2"]))
              if ("PaCO2" in s_t and "PaCO2" in s_next) else 0.0)
    cost = _action_cost(action_tuple)
    safety = _safety_penalty(s_next, patient_weight_kg)
    causal = action_causal_bonus(s_t, action_tuple, patient_weight_kg, lam_causal)
    return w_oxy * d_oxy + w_vent * d_vent - cost - safety + causal


def outcome_reward(died: bool, vent_days: float,
                   horizon_days: float = 28.0,
                   w_mortality: float = 1.0, w_vfd: float = 1.0) -> float:
    """Terminal outcome reward — mortality + ventilator-free days (VFD).

    Applied ONCE at the final transition of an episode (``done``), this ties the
    per-step physiological reward to the patient-level outcome the IntelliLung
    cross-check flagged as missing (their ``mortality.py`` + ``ventilator_free_days.py``).

    Definition (Schoenfeld 2002 VFD, generalised to ``horizon_days``):
      * If the patient died within ``horizon_days`` of ventilation start → VFD = 0
        and a mortality penalty of ``-w_mortality`` is applied.
      * Otherwise the survivor gets ``+w_vfd * vfd_frac`` where
        ``vfd_frac = clip(horizon - vent_days, 0, horizon) / horizon`` ∈ [0, 1]:
        more ventilator-free days → larger positive terminal reward.

    The returned value is the UNIT terminal term (w_mortality = w_vfd = 1 by
    default, range roughly [-1, +1]); ``dataset.py`` stores it as ``outcome_unit``
    and scales it by the config ``reward.w_outcome`` so the weight can be swept at
    train time without rebuilding the dataset (same pattern as the causal bonus).
    """
    if died:
        return -float(w_mortality)
    vent = max(0.0, min(float(vent_days), float(horizon_days)))
    vfd_frac = (horizon_days - vent) / horizon_days if horizon_days > 0 else 0.0
    return float(w_vfd) * vfd_frac


def tier2_reward(s_t: Mapping[str, float], s_next: Mapping[str, float],
                 action_tuple: Sequence[float], patient_weight_kg: float,
                 w_oxy: float = W_OXY, w_vent: float = W_VENT,
                 lam2: float = 0.2, lam3: float = 0.15,
                 lam_causal: float = LAM_CAUSAL) -> float:
    """TIER 2 reward = TIER 1 + waveform-instability penalties.

    Missing waveform features default to 0 (no penalty).
    """
    base = tier1_reward(s_t, s_next, action_tuple, patient_weight_kg,
                        w_oxy=w_oxy, w_vent=w_vent, lam_causal=lam_causal)
    async_score = float(s_t.get("Asynchrony_Score", 0.0) or 0.0)
    arr_rate = float(s_t.get("Arrhythmia_rate", 0.0) or 0.0)
    async_penalty = lam2 * async_score
    arr_penalty = lam3 if arr_rate > 0.2 else 0.0
    return base - async_penalty - arr_penalty
