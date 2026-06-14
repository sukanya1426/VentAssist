"""Stage 1 — Cohort construction.

Filter MIMIC-IV v3.1 to the eligible invasively-ventilated adult ICU cohort and
emit one row per ventilation episode.

Pipeline:
  1. Scan ``chartevents`` (chunked) for ventilation / ECMO / NIV / weight events.
  2. Build continuous ventilation windows per stay (merge events < 2 h apart).
  3. Apply inclusion (age >= 18, vent >= 6 continuous hours) and exclusion
     (cardiac arrest, ECMO, NIV-only) criteria.
  4. Truncate episodes to the first 72 hours.
  5. Derive admission weight and a (proxy) sepsis flag.
  6. Write ``data/processed/cohort.csv``.

Output columns:
  [subject_id, hadm_id, stay_id, vent_start, vent_end, age, weight_kg, sepsis_flag]

Notes / simplifications:
  * ``sepsis_flag`` uses an ICD-code proxy for sepsis rather than a full Sepsis-3
    (SOFA >= 2 + suspected infection) computation, which would require assembling
    SOFA component labs/vitals. Documented here and logged.
  * ``age`` uses ``patients.anchor_age`` (the standard MIMIC-IV de-identified age).

Run directly:
    python -m backend.pipeline.cohort                # full run
    python -m backend.pipeline.cohort --sample       # fast sample run
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("cohort")

# Union of itemids we need from the (huge) chartevents scan.
_SCAN_ITEMIDS = set(
    config.VENT_ITEMIDS
    + config.ECMO_ITEMIDS
    + config.NIV_ITEMIDS
    + config.WEIGHT_ITEMIDS
)
_VENT_SET = set(config.VENT_ITEMIDS)
_ECMO_SET = set(config.ECMO_ITEMIDS)
_NIV_SET = set(config.NIV_ITEMIDS)
_WEIGHT_SET = set(config.WEIGHT_ITEMIDS)

# Sepsis ICD proxy codes (prefix match).
_SEPSIS_ICD9 = ("99591", "99592", "78552")
_SEPSIS_ICD10 = ("A40", "A41", "R652")


# --------------------------------------------------------------------------- #
# Step 1 — scan chartevents for the events we care about
# --------------------------------------------------------------------------- #
def scan_chartevents(sample_chunks: Optional[int] = None) -> pd.DataFrame:
    """Chunked scan of chartevents, keeping only rows whose itemid is relevant.

    Args:
        sample_chunks: if set, stop after this many chunks (fast sample mode).

    Returns:
        DataFrame with columns [stay_id, charttime, itemid, valuenum].
    """
    log.info("Scanning chartevents (%s)…",
             f"sample: first {sample_chunks} chunks" if sample_chunks else "full")
    collected: list[pd.DataFrame] = []
    reader = pd.read_csv(
        config.CHARTEVENTS,
        chunksize=config.CHUNK_SIZE,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "float64"},
        parse_dates=["charttime"],
    )
    for i, chunk in enumerate(reader):
        hit = chunk[chunk["itemid"].isin(_SCAN_ITEMIDS)]
        if not hit.empty:
            collected.append(hit)
        if sample_chunks is not None and (i + 1) >= sample_chunks:
            break
        if (i + 1) % 50 == 0:
            log.info("  …processed %d chunks", i + 1)

    if not collected:
        raise ValueError("No relevant chartevents rows found in the scanned range.")
    events = pd.concat(collected, ignore_index=True)
    events = events.dropna(subset=["stay_id", "charttime"])
    events["stay_id"] = events["stay_id"].astype(int)
    events["itemid"] = events["itemid"].astype(int)
    log.info("Collected %d relevant chartevents rows across %d stays.",
             len(events), events["stay_id"].nunique())
    return events


# --------------------------------------------------------------------------- #
# Step 2 — continuous ventilation windows
# --------------------------------------------------------------------------- #
def _longest_vent_window(times: pd.Series) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge timestamps separated by <= VENT_MERGE_GAP_HOURS, return longest window."""
    times = times.sort_values().to_numpy()
    if len(times) == 0:
        return None
    gap = np.timedelta64(config.VENT_MERGE_GAP_HOURS, "h")
    best: Optional[tuple] = None
    start = prev = times[0]
    for t in times[1:]:
        if t - prev > gap:
            best = _keep_longer(best, (start, prev))
            start = t
        prev = t
    best = _keep_longer(best, (start, prev))
    return best


def _keep_longer(a, b):
    if a is None:
        return b
    if (b[1] - b[0]) > (a[1] - a[0]):
        return b
    return a


# --------------------------------------------------------------------------- #
# Step 5 — sepsis ICD proxy
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
    """Build and persist the eligible ventilation cohort.

    Args:
        sample_chunks: cap chartevents scan to this many chunks (sample mode).
        max_stays: cap the final cohort to this many stays (sample mode).
        allow_missing_patients: degraded fallback for an incomplete patients
            table. When True, ICU subjects with no patient record are KEPT
            (MIMIC-IV is adult-only) with imputed age and ``patient_known=0``.
            When None (default), it AUTO-ENABLES if < 50% of ICU subjects have a
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

    events = scan_chartevents(sample_chunks=sample_chunks)

    # --- weight per stay (median admission weight) ---
    weights = (
        events[events["itemid"].isin(_WEIGHT_SET)]
        .groupby("stay_id")["valuenum"].median()
    )
    pop_weight = float(weights.median()) if not weights.empty else 80.0

    # --- per-stay event sets for exclusion logic ---
    vent_ev = events[events["itemid"].isin(_VENT_SET)]
    stays_with_ecmo = set(events.loc[events["itemid"].isin(_ECMO_SET), "stay_id"].unique())
    stays_with_niv = set(events.loc[events["itemid"].isin(_NIV_SET), "stay_id"].unique())
    stays_with_invasive = set(vent_ev["stay_id"].unique())

    sepsis_hadm = _sepsis_hadm_ids()
    arrest_hadm = _cardiac_arrest_hadm_ids()

    icu_map = icustays.set_index("stay_id")
    pat_map = patients.set_index("subject_id")

    skipped: dict[str, int] = {}
    rows: list[dict] = []

    for stay_id, grp in vent_ev.groupby("stay_id"):
        window = _longest_vent_window(grp["charttime"])
        if window is None:
            skipped["no_window"] = skipped.get("no_window", 0) + 1
            continue
        vent_start, vent_end = window

        # vent duration >= 6 h
        dur_h = (vent_end - vent_start) / np.timedelta64(1, "h")
        if dur_h < config.MIN_VENT_HOURS:
            skipped["short_vent"] = skipped.get("short_vent", 0) + 1
            continue

        if stay_id not in icu_map.index:
            skipped["no_icustay"] = skipped.get("no_icustay", 0) + 1
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
            # MIMIC-IV is adult-only — keep with imputed age.
            age = median_age
            age_imputed = True

        # exclusions
        if stay_id in stays_with_ecmo:
            skipped["ecmo"] = skipped.get("ecmo", 0) + 1
            continue
        if (stay_id in stays_with_niv) and (stay_id not in stays_with_invasive):
            skipped["niv_only"] = skipped.get("niv_only", 0) + 1
            continue
        if hadm_id is not None and hadm_id in arrest_hadm:
            skipped["cardiac_arrest"] = skipped.get("cardiac_arrest", 0) + 1
            continue

        # truncate to first 72 h
        max_end = vent_start + np.timedelta64(config.MAX_EPISODE_HOURS, "h")
        if vent_end > max_end:
            vent_end = max_end

        weight_kg = float(weights.get(stay_id, np.nan))
        if np.isnan(weight_kg):
            weight_kg = pop_weight

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
    log.info("  age median=%.0f, sepsis rate=%.2f, mean vent dur(h)=%.1f, "
             "patient_known=%.1f%% (age imputed for %d)",
             cohort["age"].median(), cohort["sepsis_flag"].mean(),
             ((pd.to_datetime(cohort["vent_end"]) - pd.to_datetime(cohort["vent_start"]))
              / pd.Timedelta(hours=1)).mean(),
             100 * cohort["patient_known"].mean(), int((cohort["age_imputed"] == 1).sum()))
    return cohort


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ventilation cohort.")
    ap.add_argument("--sample", action="store_true",
                    help="Fast sample run (limited chartevents scan).")
    ap.add_argument("--sample-chunks", type=int, default=60,
                    help="Chartevents chunks to scan in sample mode.")
    ap.add_argument("--max-stays", type=int, default=None,
                    help="Cap the cohort size.")
    ap.add_argument("--allow-missing-patients", dest="allow_missing", default=None,
                    action="store_true",
                    help="Force degraded fallback for an incomplete patients table "
                         "(auto-detected by default).")
    args = ap.parse_args()
    build_cohort(
        sample_chunks=args.sample_chunks if args.sample else None,
        max_stays=args.max_stays,
        allow_missing_patients=args.allow_missing,
    )


if __name__ == "__main__":
    main()
