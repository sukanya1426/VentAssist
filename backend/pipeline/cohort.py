"""Stage 1 — Cohort construction.

Filter MIMIC-IV v3.1 to the eligible invasively-ventilated adult ICU cohort and
emit one row per ventilation episode.

Ventilation episodes come from ``procedureevents`` (itemid 225792, "Invasive
Ventilation"), which records explicit ``starttime``/``endtime`` for every vent
course. This is the ground-truth source for episode duration. The previous
implementation inferred windows from sparse ``chartevents`` markers merged with a
2 h gap, which fragmented real multi-day vent courses into ~7 h stubs and
under-counted the cohort by ~10x (2,944 vs ~28,665 eligible episodes). No
chartevents scan is needed at this stage any more.

Pipeline:
  1. Read ``procedureevents`` invasive-ventilation intervals (small table).
  2. Per stay, merge intervals separated by <= VENT_MERGE_GAP_HOURS and take the
     longest continuous episode (one episode per stay → no within-patient leak).
  3. Apply inclusion (age >= 18, vent >= 6 continuous hours) and exclusion
     (cardiac arrest, ECMO) criteria. (NIV-only is excluded implicitly: only
     stays with an invasive-vent procedure are considered.)
  4. Truncate episodes to the first 72 hours.
  5. Derive admission weight (procedureevents ``patientweight``) and a (proxy)
     sepsis flag.
  6. Write ``data/processed/cohort.csv``.

Output columns:
  [subject_id, hadm_id, stay_id, vent_start, vent_end, age, weight_kg,
   sepsis_flag, patient_known, age_imputed]

Notes / simplifications:
  * ``sepsis_flag`` uses an ICD-code proxy for sepsis rather than a full Sepsis-3
    computation. Documented here and logged.
  * ``age`` uses ``patients.anchor_age`` (the standard MIMIC-IV de-identified age).

Run directly:
    python -m backend.pipeline.cohort                # full run
    python -m backend.pipeline.cohort --max-stays 5000   # capped run
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("cohort")

# Sepsis ICD proxy codes (prefix match).
_SEPSIS_ICD9 = ("99591", "99592", "78552")
_SEPSIS_ICD10 = ("A40", "A41", "R652")


# --------------------------------------------------------------------------- #
# Interval merging — longest continuous vent episode per stay
# --------------------------------------------------------------------------- #
def _longest_episode(intervals: np.ndarray,
                     gap_h: float) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge [start, end] intervals separated by <= gap_h and return the longest.

    Args:
        intervals: array shape (n, 2) of datetime64 [start, end] pairs.
        gap_h: merge two intervals if the second starts within this many hours
            of the running end (treats brief disconnections as one episode).
    """
    if len(intervals) == 0:
        return None
    order = np.argsort(intervals[:, 0])
    intervals = intervals[order]
    gap = np.timedelta64(int(gap_h * 60), "m")
    best = None
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start - cur_end <= gap:
            if end > cur_end:
                cur_end = end
        else:
            best = _keep_longer(best, (cur_start, cur_end))
            cur_start, cur_end = start, end
    best = _keep_longer(best, (cur_start, cur_end))
    return best


def _keep_longer(a, b):
    if a is None:
        return b
    return b if (b[1] - b[0]) > (a[1] - a[0]) else a


# --------------------------------------------------------------------------- #
# ICD proxies
# --------------------------------------------------------------------------- #
def _sepsis_hadm_ids() -> set[int]:
    """hadm_ids with a sepsis ICD code (proxy for Sepsis-3)."""
    dx = pd.read_csv(
        config.DIAGNOSES,
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"hadm_id": "Int64", "icd_code": "string", "icd_version": "Int64"},
    )
    code = dx["icd_code"].str.replace(".", "", regex=False).str.upper().fillna("")
    is9 = (dx["icd_version"] == 9) & code.str.startswith(_SEPSIS_ICD9)
    is10 = (dx["icd_version"] == 10) & code.str.startswith(_SEPSIS_ICD10)
    return set(dx.loc[is9 | is10, "hadm_id"].dropna().astype(int))


def _cardiac_arrest_hadm_ids() -> set[int]:
    """hadm_ids with a cardiac-arrest ICD code (I46.x / 4275) → excluded."""
    dx = pd.read_csv(
        config.DIAGNOSES,
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"hadm_id": "Int64", "icd_code": "string", "icd_version": "Int64"},
    )
    code = dx["icd_code"].str.replace(".", "", regex=False).str.upper().fillna("")
    is9 = (dx["icd_version"] == 9) & code.str.startswith("4275")
    is10 = (dx["icd_version"] == 10) & code.str.startswith("I46")
    return set(dx.loc[is9 | is10, "hadm_id"].dropna().astype(int))


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def build_cohort(sample_chunks: Optional[int] = None,
                 max_stays: Optional[int] = None,
                 allow_missing_patients: Optional[bool] = None) -> pd.DataFrame:
    """Build and persist the eligible ventilation cohort from procedureevents.

    Args:
        sample_chunks: ignored (kept for API compatibility; procedureevents is a
            small table read in full).
        max_stays: cap the final cohort to this many stays (sample mode).
        allow_missing_patients: degraded fallback for an incomplete patients
            table. When True, ICU subjects with no patient record are KEPT
            (MIMIC-IV is adult-only) with imputed age and ``patient_known=0``.
            When None (default), AUTO-ENABLES if < 50% of ICU subjects have a
            patient record — so dropping in a complete patients.csv later
            restores normal behaviour automatically.
    """
    config.ensure_output_dirs()

    # --- reference tables (all small) ---
    icustays = pd.read_csv(
        config.ICUSTAYS,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    patients = pd.read_csv(
        config.PATIENTS, usecols=["subject_id", "anchor_age", "gender", "dod"],
        parse_dates=["dod"],
    )

    # --- patient-table coverage check / degraded-mode decision ---
    icu_subjects = set(icustays["subject_id"].unique())
    pat_subjects = set(patients["subject_id"].unique())
    coverage = len(icu_subjects & pat_subjects) / max(len(icu_subjects), 1)
    if allow_missing_patients is None:
        allow_missing_patients = coverage < 0.50
    if allow_missing_patients:
        log.warning("DEGRADED MODE: patients table covers only %.1f%% of ICU "
                    "subjects. Stays without a patient record are KEPT with "
                    "imputed age (patient_known=0). Drop in a complete "
                    "patients.csv to restore full fidelity.", 100 * coverage)
    else:
        log.info("Patient-table coverage %.1f%% — normal mode.", 100 * coverage)
    known_ages = patients.set_index("subject_id")["anchor_age"]
    median_age = int(known_ages[known_ages >= config.MIN_AGE].median()) \
        if known_ages.notna().any() else 65

    # --- procedureevents: invasive-vent intervals + ECMO markers ---
    log.info("Reading procedureevents (ventilation intervals)…")
    pe = pd.read_csv(
        config.PROCEDUREEVENTS,
        usecols=["stay_id", "itemid", "starttime", "endtime", "patientweight"],
        dtype={"stay_id": "Int64", "itemid": "Int64", "patientweight": "float64"},
        parse_dates=["starttime", "endtime"],
    )
    pe = pe.dropna(subset=["stay_id", "starttime", "endtime"])
    pe["stay_id"] = pe["stay_id"].astype(int)

    inv = pe[pe["itemid"] == config.VENT_PROC_ITEMID]
    ecmo_stays = set(pe.loc[pe["itemid"].isin(config.ECMO_PROC_ITEMIDS),
                            "stay_id"].unique())
    log.info("Invasive-vent procedure rows: %d across %d stays (ECMO stays: %d).",
             len(inv), inv["stay_id"].nunique(), len(ecmo_stays))

    # population-median weight (clipped to a plausible adult range) for fallback
    pw = inv["patientweight"].clip(20.0, 400.0)
    pop_weight = float(pw.median()) if pw.notna().any() else 80.0

    sepsis_hadm = _sepsis_hadm_ids()
    arrest_hadm = _cardiac_arrest_hadm_ids()

    icu_map = icustays.set_index("stay_id")
    pat_map = patients.set_index("subject_id")

    skipped: dict[str, int] = {}
    rows: list[dict] = []

    for stay_id, grp in inv.groupby("stay_id"):
        if stay_id not in icu_map.index:
            skipped["no_icustay"] = skipped.get("no_icustay", 0) + 1
            continue

        window = _longest_episode(
            grp[["starttime", "endtime"]].to_numpy(),
            gap_h=config.VENT_MERGE_GAP_HOURS)
        if window is None:
            skipped["no_window"] = skipped.get("no_window", 0) + 1
            continue
        vent_start, vent_end = window

        dur_h = (vent_end - vent_start) / np.timedelta64(1, "h")
        if dur_h < config.MIN_VENT_HOURS:
            skipped["short_vent"] = skipped.get("short_vent", 0) + 1
            continue

        srow = icu_map.loc[stay_id]
        subject_id = int(srow["subject_id"])
        hadm_id = int(srow["hadm_id"]) if not pd.isna(srow["hadm_id"]) else None

        # age >= 18  (or kept via degraded fallback when patient record absent)
        patient_known = subject_id in pat_map.index
        age_imputed = False
        if patient_known:
            age = pat_map.loc[subject_id, "anchor_age"]
            age = int(age) if not pd.isna(age) else None
            if age is None or age < config.MIN_AGE:
                skipped["underage"] = skipped.get("underage", 0) + 1
                continue
        else:
            if not allow_missing_patients:
                skipped["no_patient"] = skipped.get("no_patient", 0) + 1
                continue
            age = median_age
            age_imputed = True

        # exclusions
        if stay_id in ecmo_stays:
            skipped["ecmo"] = skipped.get("ecmo", 0) + 1
            continue
        if hadm_id is not None and hadm_id in arrest_hadm:
            skipped["cardiac_arrest"] = skipped.get("cardiac_arrest", 0) + 1
            continue

        # truncate to first 72 h
        max_end = vent_start + np.timedelta64(config.MAX_EPISODE_HOURS, "h")
        if vent_end > max_end:
            vent_end = max_end

        w = grp["patientweight"].clip(20.0, 400.0).median()
        weight_kg = float(w) if not pd.isna(w) else pop_weight

        rows.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": int(stay_id),
            "vent_start": vent_start,
            "vent_end": vent_end,
            "age": age,
            "weight_kg": round(weight_kg, 1),
            "sepsis_flag": int(hadm_id in sepsis_hadm) if hadm_id is not None else 0,
            "patient_known": int(patient_known),
            "age_imputed": int(age_imputed),
        })

    cohort = pd.DataFrame(rows)
    if skipped:
        log.info("Skipped stays by reason: %s", dict(sorted(skipped.items())))

    if cohort.empty:
        raise ValueError("Cohort is empty after applying inclusion/exclusion criteria.")

    cohort = cohort.sort_values("stay_id").reset_index(drop=True)
    if max_stays is not None and len(cohort) > max_stays:
        cohort = cohort.head(max_stays).reset_index(drop=True)
        log.info("Capped cohort to first %d stays (sample mode).", max_stays)

    out = config.PROCESSED_PATH / "cohort.csv"
    cohort.to_csv(out, index=False)
    log.info("Wrote cohort: %d episodes → %s", len(cohort), out)
    dur = ((pd.to_datetime(cohort["vent_end"]) - pd.to_datetime(cohort["vent_start"]))
           / pd.Timedelta(hours=1))
    log.info("  age median=%.0f, sepsis rate=%.2f, vent dur(h) median=%.1f "
             "mean=%.1f max=%.1f | patient_known=%.1f%% (age imputed for %d)",
             cohort["age"].median(), cohort["sepsis_flag"].mean(),
             dur.median(), dur.mean(), dur.max(),
             100 * cohort["patient_known"].mean(),
             int((cohort["age_imputed"] == 1).sum()))
    return cohort


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ventilation cohort.")
    ap.add_argument("--sample", action="store_true",
                    help="Fast sample run (caps cohort to --max-stays).")
    ap.add_argument("--max-stays", type=int, default=None,
                    help="Cap the cohort size.")
    ap.add_argument("--allow-missing-patients", dest="allow_missing", default=None,
                    action="store_true",
                    help="Force degraded fallback for an incomplete patients table "
                         "(auto-detected by default).")
    args = ap.parse_args()
    build_cohort(
        max_stays=args.max_stays if args.max_stays else (2000 if args.sample else None),
        allow_missing_patients=args.allow_missing,
    )


if __name__ == "__main__":
    main()
