"""Mode-aware action masking (SYSTEM_SUMMARY §16 item 6).

IntelliLung's `actions/masking.py` removes clinically impossible actions given the
ventilation mode: tidal volume is directly settable only in **volume-control**
modes; in **pressure-control** modes the clinician sets an inspiratory pressure
and TV is a *result* of it (depends on compliance/resistance), so commanding a
ΔTV is not physically meaningful. VentAssist's action space is (ΔPEEP, ΔTV, ΔFiO₂)
— PEEP and FiO₂ are settable in both modes, so the only applicable mask is:

    pressure_control  →  disallow every action with ΔTV ≠ 0 (keep ΔTV = 0).

This is applied at inference (the 12-dim state carries no mode), driven by an
optional ``ventilation_mode`` on the request — analogous to the safety filter.
``None``/unknown ⇒ no masking (the deployed behaviour is unchanged by default).
"""

from __future__ import annotations

import numpy as np

from backend.mdp import action_space

# Ventilator-mode strings (chartevents itemid 223849 / 229314) → control paradigm.
# Best-effort curated from the invasive-mode list in IntelliLung's ventilation.sql;
# genuinely ambiguous/adaptive modes map to None (no mask) rather than guess.
VOLUME_CONTROL_MODES = {
    "CMV", "CMV/ASSIST", "CMV/ASSIST/AutoFlow", "CMV/AutoFlow", "VOL/AC", "SIMV/VOL",
    "MMV", "MMV/AutoFlow", "PRVC/AC", "PRVC/SIMV", "APV (cmv)", "APV (simv)", "VS",
}
PRESSURE_CONTROL_MODES = {
    "PCV+", "PCV+/PSV", "PCV+Assist", "P-CMV", "P-SIMV", "PRES/AC", "PSV/SBT",
    "CPAP/PSV", "CPAP/PPS", "CPAP/PSV+Apn TCPL", "CPAP/PSV+ApnPres", "CPAP/PSV+ApnVol",
    "APRV", "APRV/Biphasic+ApnPress", "APRV/Biphasic+ApnVol", "DuoPaP", "NIV", "NIV-ST",
    "MMV/PSV", "MMV/PSV/AutoFlow", "SIMV/PRES", "SIMV/PSV", "SIMV/PSV/AutoFlow",
}

_VOLUME_ALIASES = {"volume_control", "vc", "volume", "volume control"}
_PRESSURE_ALIASES = {"pressure_control", "pc", "pressure", "pressure control"}


def classify_ventilator_mode(mode: str | None) -> str | None:
    """Map a category label or raw ventilator-mode string → 'volume_control',
    'pressure_control', or None (unknown / not maskable)."""
    if mode is None:
        return None
    m = str(mode).strip()
    low = m.lower()
    if low in _VOLUME_ALIASES:
        return "volume_control"
    if low in _PRESSURE_ALIASES:
        return "pressure_control"
    if low in {"", "unknown", "none"}:
        return None
    if m in VOLUME_CONTROL_MODES:
        return "volume_control"
    if m in PRESSURE_CONTROL_MODES:
        return "pressure_control"
    return None


def mode_action_mask(mode: str | None) -> np.ndarray:
    """Boolean allow-mask over the 125 actions for a ventilation mode.

    True = clinically permissible. In pressure-control, every ΔTV ≠ 0 action is
    masked out; otherwise all actions are allowed."""
    mask = np.ones(action_space.N_ACTIONS, dtype=bool)
    if classify_ventilator_mode(mode) == "pressure_control":
        for idx, (_dp, dtv, _df, _desc) in action_space.ACTION_MAP.items():
            if dtv != 0:
                mask[idx] = False
    return mask
