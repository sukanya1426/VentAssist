"""Safety filter — every recommendation must pass before it reaches the frontend.

Rule-based guards on the resulting ventilator settings, independent of the
policy. Returns clear CRITICAL / WARNING flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MAX_PEEP = float(os.getenv("MAX_PEEP", 15))
MAX_FIO2 = float(os.getenv("MAX_FIO2", 0.8))
MAX_TV_ML_PER_KG = float(os.getenv("MAX_TV_ML_PER_KG", 8.0))


@dataclass
class SafetyResult:
    all_clear: bool
    flags: list[dict] = field(default_factory=list)


def check(state: dict, delta_PEEP: int, delta_TV: int, delta_FiO2: float,
          patient_weight: float, arrhythmia_rate: float | None = None) -> SafetyResult:
    flags: list[dict] = []
    new_peep = float(state["PEEP"]) + delta_PEEP
    new_tv = float(state["TV"]) + delta_TV
    # FiO₂ is clipped to its physiological range after applying the delta
    new_fio2 = min(1.0, max(0.21, float(state["FiO2"]) + delta_FiO2))

    if new_peep > MAX_PEEP:
        flags.append({"level": "CRITICAL",
                      "message": f"Resulting PEEP {new_peep:.0f} exceeds {MAX_PEEP:.0f} cmH₂O"})
    if new_tv > MAX_TV_ML_PER_KG * patient_weight:
        flags.append({"level": "CRITICAL",
                      "message": f"Resulting TV {new_tv:.0f} mL exceeds "
                                 f"{MAX_TV_ML_PER_KG:.0f} mL/kg "
                                 f"({MAX_TV_ML_PER_KG * patient_weight:.0f} mL)"})
    if new_fio2 > MAX_FIO2:
        flags.append({"level": "WARNING",
                      "message": f"Projected FiO₂ {new_fio2:.2f} exceeds the oxygen "
                                 f"toxicity threshold ({MAX_FIO2:.2f})"})
    if float(state["FiO2"]) > MAX_FIO2 and delta_PEEP <= 0:
        flags.append({"level": "WARNING",
                      "message": f"FiO₂ {state['FiO2']:.2f} is high; consider raising PEEP"})
    if new_peep < 0 or new_tv < 100:
        flags.append({"level": "CRITICAL",
                      "message": "Resulting setting is below a safe floor"})
    if arrhythmia_rate is not None and arrhythmia_rate > 0.2:
        flags.append({"level": "WARNING",
                      "message": f"Elevated arrhythmia rate ({arrhythmia_rate:.2f}); "
                                 "interpret waveform recommendation with caution"})

    critical = any(f["level"] == "CRITICAL" for f in flags)
    return SafetyResult(all_clear=not critical, flags=flags)
