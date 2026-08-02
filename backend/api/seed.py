"""The six built-in roster patients, seeded into MongoDB on startup.

These mirror ``frontend/src/data/patientPresets.ts`` + ``patients.ts``: each state
exercises a different part of the policy so the recommendation differs meaningfully
between beds. Moving them here makes Mongo the single source of truth for the
roster — the frontend now reads all six from ``GET /api/patients`` rather than
holding its own copy.

Seeding only bootstraps an **empty** roster. Once the collection holds anything at
all it is left alone, so a deleted preset stays deleted and an edited one is not
clobbered — a delete needs no tombstone or other bookkeeping to survive a restart.
The one consequence: emptying the roster completely and restarting brings the six
back, which is the sane reading of "no patients on file" for a fresh deployment.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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


def preset_documents() -> list[dict]:
    now = datetime.now(timezone.utc)
    docs = []
    for order, (letter, weight, age, sex, bed, summary, hint, state) in enumerate(_PRESETS):
        docs.append({
            "_id": f"patient-{letter.lower()}",
            "name": f"Patient {letter}",
            "weight": weight,
            "age": age,
            "sex": sex,
            "bed": bed,
            "summary": summary,
            "hint": hint,
            "state": dict(state),
            "source": "preset",
            "waveform": None,
            "ventilation_mode": None,
            "track": "track_a",
            "order": order,
            "created_at": now,
            "updated_at": now,
        })
    return docs


async def seed_presets() -> int:
    """Populate the roster on a fresh database. Returns how many patients were added.

    A non-empty roster is never touched — whatever is on it is what the clinician
    left there.
    """
    coll = db.patients()
    if await coll.count_documents({}, limit=1):
        return 0
    docs = preset_documents()
    await coll.insert_many(docs)
    log.info("Seeded %d preset patient(s) into an empty MongoDB roster", len(docs))
    return len(docs)
