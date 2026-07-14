"""Stage — Patient-level outcomes (mortality + ventilator-free days).

Closes the outcome-reward gap flagged by the IntelliLung cross-check (their
``mortality.py`` + ``ventilator_free_days.py``): emits, per ventilation episode,
a mortality flag and a true ventilation duration so a terminal outcome reward can
be attached at the last MDP transition (see ``reward.outcome_reward`` and
``dataset.py``).

Definitions
-----------
* ``died_horizon`` — 1 if the patient's date of death (``patients.dod``) falls
  within ``horizon_days`` of ventilation start, else 0. (A 1-day grace before
  ``vent_start`` is allowed for date-only ``dod`` rounding, mirroring
  ``propensity_score.py``.)
* ``vent_days`` — TRUE total invasive-ventilation duration in days, summed across
  all invasive-vent intervals (itemid ``VENT_PROC_ITEMID``) for the stay. This is
  deliberately computed from the raw ``procedureevents`` intervals rather than the
  cohort's 72 h-truncated ``vent_end``, so VFD reflects real ventilator exposure.
* ``vfd`` — ventilator-free days within the horizon (Schoenfeld 2002):
  0 if the patient died within the horizon OR was ventilated for the whole
  horizon; otherwise ``horizon_days - vent_days``.

Only episodes with a real patient record (``patient_known == 1``) get a reliable
mortality label; for missing-patient stays (degraded mode) ``died_horizon`` is
left 0 and ``outcome_label_known = 0`` so downstream code can choose to zero the
terminal reward for them.

Inputs : cohort.csv, patients.csv(.gz), procedureevents.csv(.gz)
Output : data/processed/outcomes.csv
         [stay_id, subject_id, died_horizon, vent_days, vfd, outcome_label_known]

Run:
    python -m backend.pipeline.outcomes                 # default horizon 28 d
    python -m backend.pipeline.outcomes --horizon-days 30
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("outcomes")

DEFAULT_HORIZON_DAYS = 28.0


def _true_vent_days() -> pd.Series:
    """Total invasive-ventilation days per stay from raw procedureevents."""
    pe = pd.read_csv(
        config.PROCEDUREEVENTS,
        usecols=["stay_id", "itemid", "starttime", "endtime"],
        dtype={"stay_id": "Int64", "itemid": "Int64"},
        parse_dates=["starttime", "endtime"],
    )
    pe = pe.dropna(subset=["stay_id", "starttime", "endtime"])
    inv = pe[pe["itemid"] == config.VENT_PROC_ITEMID].copy()
    inv["stay_id"] = inv["stay_id"].astype(int)
    hours = (inv["endtime"] - inv["starttime"]) / pd.Timedelta(hours=1)
    inv = inv.assign(hours=hours.clip(lower=0.0))
    return inv.groupby("stay_id")["hours"].sum() / 24.0  # days


def build_outcomes(horizon_days: float = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    config.ensure_output_dirs()
    cohort = pd.read_csv(config.PROCESSED_PATH / "cohort.csv",
                         parse_dates=["vent_start", "vent_end"])

    patients = pd.read_csv(config.PATIENTS, usecols=["subject_id", "dod"],
                           parse_dates=["dod"])
    dod = patients.set_index("subject_id")["dod"]
    vent_days_map = _true_vent_days()

    horizon = pd.Timedelta(days=horizon_days)
    rows = []
    for _, ep in cohort.iterrows():
        sid, subj = int(ep["stay_id"]), int(ep["subject_id"])
        known = int(ep.get("patient_known", 1)) == 1
        d = dod.get(subj, pd.NaT)
        delta = (d - ep["vent_start"]) if pd.notna(d) else pd.NaT
        died = int(known and pd.notna(delta)
                   and (delta <= horizon) and (delta >= pd.Timedelta(days=-1)))

        vd = float(vent_days_map.get(sid, np.nan))
        if np.isnan(vd):
            # fall back to the (truncated) cohort window if no raw interval found
            vd = float((ep["vent_end"] - ep["vent_start"]) / pd.Timedelta(days=1))

        if died or vd >= horizon_days:
            vfd = 0.0
        else:
            vfd = horizon_days - vd

        rows.append({
            "stay_id": sid,
            "subject_id": subj,
            "died_horizon": died,
            "vent_days": round(vd, 3),
            "vfd": round(vfd, 3),
            "outcome_label_known": int(known),
        })

    out_df = pd.DataFrame(rows).sort_values("stay_id").reset_index(drop=True)
    out = config.PROCESSED_PATH / "outcomes.csv"
    out_df.to_csv(out, index=False)

    known_df = out_df[out_df["outcome_label_known"] == 1]
    log.info("Wrote outcomes (horizon=%.0f d): %d episodes → %s",
             horizon_days, len(out_df), out)
    log.info("  %d-day mortality (known patients)=%.3f | vent_days median=%.2f | "
             "VFD mean=%.2f | label_known=%.1f%%",
             horizon_days,
             known_df["died_horizon"].mean() if len(known_df) else 0.0,
             out_df["vent_days"].median(),
             out_df["vfd"].mean(),
             100 * out_df["outcome_label_known"].mean())
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-episode outcomes (mortality + VFD).")
    ap.add_argument("--horizon-days", type=float, default=DEFAULT_HORIZON_DAYS)
    args = ap.parse_args()
    build_outcomes(horizon_days=args.horizon_days)


if __name__ == "__main__":
    main()
