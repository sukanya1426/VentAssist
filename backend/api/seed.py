"""The six built-in roster patients, seeded into PostgreSQL on startup.

These mirror ``frontend/src/data/patientPresets.ts``: each state exercises a
different part of the policy so the recommendation differs meaningfully between
beds. Keeping them here makes the database the single source of truth for the
roster — the frontend reads all six from ``GET /api/patients`` rather than holding
its own copy.

Seeding only bootstraps an **empty** roster. Once ``patient`` holds anything at
all it is left alone, so a deleted preset stays deleted and an edited one is not
clobbered — a delete needs no tombstone or other bookkeeping to survive a restart.
The one consequence: emptying the roster completely and restarting brings the six
back, which is the sane reading of "no patients on file" for a fresh deployment.
"""

from __future__ import annotations

import logging

from backend.api import db

log = logging.getLogger(__name__)

# (letter, weight, age, sex, bed, summary, hint, state)
_PRESETS = [
    ("A", 74, 61, "M", "ICU-01",
     "Post-op, settings and gases in range, weaning candidate",
     "settings & gases in range → expect Hold",
     {"PEEP": 8, "TV": 460, "FiO2": 0.40, "SpO2": 95, "PaO2": 88, "PaCO2": 40,
      "pH": 7.40, "HR": 84, "SBP": 120, "RR": 16, "RASS": -1, "Temp": 37.0}),
    ("B", 68, 55, "F", "ICU-02",
     "ARDS, hypoxaemic on FiO₂ 0.5, tachypnoeic",
     "SpO₂ 84 on FiO₂ 0.5 → expect ↑PEEP / ↑FiO₂",
     {"PEEP": 8, "TV": 480, "FiO2": 0.50, "SpO2": 84, "PaO2": 55, "PaCO2": 44,
      "pH": 7.34, "HR": 104, "SBP": 112, "RR": 26, "RASS": -2, "Temp": 37.6}),
    ("C", 81, 70, "M", "ICU-03",
     "Oxygen saturation 100% on high FiO₂ — over-oxygenated",
     "SpO₂ 100 on FiO₂ 0.7 → expect ↓FiO₂",
     {"PEEP": 8, "TV": 480, "FiO2": 0.70, "SpO2": 100, "PaO2": 90, "PaCO2": 44,
      "pH": 7.37, "HR": 92, "SBP": 118, "RR": 22, "RASS": -2, "Temp": 37.2}),
    ("D", 62, 48, "F", "ICU-04",
     "COPD exacerbation, hypercapnic on low tidal volume",
     "PaCO₂ 65 with TV 320 → expect ↑TV",
     {"PEEP": 8, "TV": 320, "FiO2": 0.50, "SpO2": 93, "PaO2": 72, "PaCO2": 65,
      "pH": 7.28, "HR": 98, "SBP": 118, "RR": 28, "RASS": -1, "Temp": 37.3}),
    ("E", 90, 66, "M", "ICU-05",
     "High PEEP support, over-distension risk",
     "PEEP 18 → expect ↓PEEP",
     {"PEEP": 18, "TV": 470, "FiO2": 0.50, "SpO2": 94, "PaO2": 80, "PaCO2": 43,
      "pH": 7.38, "HR": 90, "SBP": 105, "RR": 20, "RASS": -3, "Temp": 37.2}),
    ("F", 77, 52, "M", "ICU-06",
     "Large tidal volumes delivered — volutrauma risk",
     "TV 760 → expect ↓TV",
     {"PEEP": 8, "TV": 760, "FiO2": 0.50, "SpO2": 95, "PaO2": 90, "PaCO2": 38,
      "pH": 7.44, "HR": 86, "SBP": 122, "RR": 14, "RASS": -2, "Temp": 37.0}),
]


# Column order for an INSERT INTO patient — the 12 clinical fields are flat
# columns, so the composite State is spelled out once here rather than at each
# call site.
_STATE_COLUMNS = ["peep", "tv", "fio2", "spo2", "pao2", "paco2",
                  "ph", "hr", "sbp", "rr", "rass", "temp"]
# The preset dicts use the API's field names; the table uses lower-case columns.
_STATE_KEYS = ["PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2",
               "pH", "HR", "SBP", "RR", "RASS", "Temp"]


def preset_rows() -> list[tuple]:
    """The presets as ``patient`` rows, in INSERT column order."""
    rows = []
    for order, (letter, weight, age, sex, bed, summary, hint, state) in enumerate(_PRESETS):
        rows.append((
            f"patient-{letter.lower()}",      # patient_id
            f"Patient {letter}",              # name
            bed, summary, hint,
            age, sex, float(weight),
            *[float(state[k]) for k in _STATE_KEYS],
            "track_a",                        # track
            "preset",                         # source
            order,                            # display_order
        ))
    return rows


async def seed_presets() -> int:
    """Populate the roster on a fresh database. Returns how many patients were added.

    A non-empty roster is never touched — whatever is on it is what the clinician
    left there. ``LIMIT 1`` because the question is "is it empty", not "how many".
    """
    if await db.fetchval("SELECT 1 FROM patient LIMIT 1"):
        return 0
    rows = preset_rows()
    cols = (["patient_id", "name", "bed", "summary", "hint",
             "age", "sex", "weight_kg"] + _STATE_COLUMNS
            + ["track", "source", "display_order"])
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    sql = (f"INSERT INTO patient ({', '.join(cols)}) VALUES ({placeholders}) "
           "ON CONFLICT (patient_id) DO NOTHING")
    pool = await db.get_pool()
    async with pool.acquire() as con:
        # One transaction: a half-seeded roster is worse than an empty one.
        async with con.transaction():
            await con.executemany(sql, rows)
    log.info("Seeded %d preset patient(s) into an empty roster", len(rows))
    return len(rows)
