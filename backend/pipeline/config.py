"""Centralised configuration, dataset paths, and MIMIC-IV itemid constants.

All raw data lives at ``files/`` OUTSIDE the project and is read directly — never
copied in. Paths are resolved relative to the repository root regardless of the
current working directory, and can be overridden via environment variables
(loaded from ``backend/.env``).

IMPORTANT — itemid corrections vs. the original spec:
  * The spec listed 224687 as Tidal Volume; 224687 is actually *Minute Volume*.
    The correct tidal-volume itemids are 224685 (observed), 224684 (set),
    224686 (spontaneous).
  * The spec listed FiO2 itemid 3420; that CareVue itemid does not exist in this
    MetaVision-era MIMIC-IV v3.1 extract. Use 223835 (Inspired O2 Fraction).
  * The spec listed ECMO itemids 226873/226874; 226873 is *Inspiratory Ratio*.
    The correct ECMO itemids are 224660 (ECMO) and 228193 (Oxygenator/ECMO).
All itemids below were verified against files/mimiciv/3.1/icu/d_items.csv.gz.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Repository-root-anchored path resolution
# --------------------------------------------------------------------------- #
# This file is backend/pipeline/config.py → repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load backend/.env if present (does not override already-set env vars).
load_dotenv(REPO_ROOT / "backend" / ".env")


def _abs(env_key: str, default: str) -> Path:
    """Resolve an env-configured path against the repo root if it is relative."""
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


# Base dataset directories
MIMIC_IV_HOSP = _abs("MIMIC_IV_HOSP_PATH", "files/mimiciv/3.1/hosp")
MIMIC_IV_ICU = _abs("MIMIC_IV_ICU_PATH", "files/mimiciv/3.1/icu")
WAVEFORM_PATH = _abs("WAVEFORM_PATH", "files/mimic4wdb/0.1.0")
WAVEFORM_WAVES = _abs("WAVEFORM_WAVES_PATH", "files/mimic4wdb/0.1.0/waves")

# Output directories (inside the project)
PROCESSED_PATH = _abs("PROCESSED_PATH", "backend/data/processed")
MODEL_PATH = _abs("MODEL_PATH", "backend/models")
LOGS_PATH = REPO_ROOT / "backend" / "logs"

def resolve_table(gz_path: Path) -> Path:
    """Return a readable source for a MIMIC table.

    Some ``.csv.gz`` archives on this disk are truncated/corrupt (notably
    patients and labevents). When an uncompressed ``.csv`` sibling exists we
    prefer it, sidestepping the bad archives. pandas infers compression from the
    file extension, so either path reads transparently.
    """
    csv_sibling = gz_path.with_suffix("")  # drop ".gz" → ".csv"
    if csv_sibling.exists():
        return csv_sibling
    return gz_path


# MIMIC-IV clinical files. resolve_table() prefers the uncompressed .csv when
# present (some .gz archives on disk are corrupt).
ICUSTAYS = resolve_table(MIMIC_IV_ICU / "icustays.csv.gz")
CHARTEVENTS = resolve_table(MIMIC_IV_ICU / "chartevents.csv.gz")   # always chunked
PROCEDUREEVENTS = resolve_table(MIMIC_IV_ICU / "procedureevents.csv.gz")
DITEMS = resolve_table(MIMIC_IV_ICU / "d_items.csv.gz")
LABEVENTS = resolve_table(MIMIC_IV_HOSP / "labevents.csv.gz")      # always chunked
ADMISSIONS = resolve_table(MIMIC_IV_HOSP / "admissions.csv.gz")
PATIENTS = resolve_table(MIMIC_IV_HOSP / "patients.csv.gz")
DIAGNOSES = resolve_table(MIMIC_IV_HOSP / "diagnoses_icd.csv.gz")

# Waveform record index
WAVEFORM_RECORDS = WAVEFORM_PATH / "RECORDS"

# Chunk size for the large event tables
CHUNK_SIZE = 100_000

# --------------------------------------------------------------------------- #
# MIMIC-IV chartevents itemids  (verified against d_items.csv.gz)
# --------------------------------------------------------------------------- #

# --- Ventilation presence (used by cohort detection) ---
# Any of these appearing for a stay_id at a timestamp implies invasive ventilation.
VENT_ITEMIDS = [
    225792,  # Invasive Ventilation (procedureevents / chartevents marker)
    223849,  # Ventilator Mode
    226260,  # Mechanically Ventilated
    220339,  # PEEP set
    224684,  # Tidal Volume (set)
]

# --- procedureevents itemids (ground-truth ventilation intervals) ---
# Cohort ventilation episodes are derived from procedureevents (explicit
# start/end times), NOT from sparse chartevents markers — the latter fragment
# multi-day vent courses into ~7h stubs (verified against the data).
VENT_PROC_ITEMID = 225792                 # Invasive Ventilation
NIV_PROC_ITEMID = 225794                  # Non-invasive Ventilation
ECMO_PROC_ITEMIDS = [229529, 229530]      # ECMO Inflow / Outflow Line

# --- Exclusion markers (legacy chartevents-based; kept for reference) ---
ECMO_ITEMIDS = [224660, 228193]          # ECMO, Oxygenator/ECMO
NIV_ITEMIDS = [225794]                    # Non-invasive Ventilation
CARDIAC_ARREST_ICD = {                    # ICD-9 4275 / ICD-10 I46.x
    "9": ["4275"],
    "10": ["I46"],
}

# --- Weight ---
WEIGHT_ITEMIDS = [226512]                 # Admission Weight (Kg)

# --------------------------------------------------------------------------- #
# 12-dim TIER 1 state feature → itemid map (order is CANONICAL — do not reorder)
# Index aligns with the model input vector and normaliser_stats.json.
# PaO2 / PaCO2 / pH come from GP imputation (labevents), not chartevents.
# --------------------------------------------------------------------------- #
TABULAR_FEATURES = [
    "PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2", "pH",
    "HR", "SBP", "RR", "RASS", "Temp",
]

# chartevents-sourced features (PaO2/PaCO2/pH excluded — imputed from labs)
CHART_FEATURE_ITEMIDS = {
    "PEEP": [220339, 224700],             # PEEP set, Total PEEP Level
    "TV": [224685, 224684, 224686],       # observed, set, spontaneous (NOT 224687)
    "FiO2": [223835],                     # Inspired O2 Fraction (NOT 3420)
    "SpO2": [220277],                     # O2 saturation pulseoxymetry
    "HR": [220045],                       # Heart Rate
    "SBP": [220179, 220050],              # NIBP systolic, ABP systolic
    "RR": [220210],                       # Respiratory Rate
    "RASS": [228096],                     # Richmond-RAS Scale
    "Temp": [223761, 223762],             # Temperature F, Temperature C
}

# Forward-fill horizon per feature, in hours.
FORWARD_FILL_HOURS = {
    "PEEP": 2, "TV": 2, "FiO2": 2, "SpO2": 1,
    "HR": 1, "SBP": 1, "RR": 1, "RASS": 2, "Temp": 2,
}

# GP-imputed lab features (from labevents) and physiological clip ranges.
LAB_FEATURES = ["PaO2", "PaCO2", "pH"]
LAB_ITEMIDS = {
    "PaO2": [50821],    # pO2, Blood Gas
    "PaCO2": [50818],   # pCO2, Blood Gas
    "pH": [50820],      # pH, Blood Gas
}
LAB_CLIP_RANGES = {
    "PaO2": (40.0, 600.0),
    "PaCO2": (15.0, 100.0),
    "pH": (6.8, 7.8),
}

# Physiological clip ranges for chartevents features (applied before winsorising).
CHART_CLIP_RANGES = {
    "PEEP": (0.0, 30.0),
    "TV": (0.0, 1500.0),
    "FiO2": (0.21, 1.0),
    "SpO2": (50.0, 100.0),
    "HR": (20.0, 250.0),
    "SBP": (40.0, 250.0),
    "RR": (0.0, 60.0),
    "RASS": (-5.0, 4.0),
    "Temp": (30.0, 43.0),
}

# --------------------------------------------------------------------------- #
# Cohort inclusion / exclusion parameters
# --------------------------------------------------------------------------- #
MIN_AGE = 18
MIN_VENT_HOURS = 6           # minimum continuous ventilation duration
MAX_EPISODE_HOURS = 72       # truncate longer episodes
VENT_MERGE_GAP_HOURS = 2     # merge vent events separated by <= this gap

# Deterministic train/val/test split fractions (split by stay_id).
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLIT_SEED = 42


def ensure_output_dirs() -> None:
    """Create the processed-data, models, and logs directories if missing."""
    for d in (PROCESSED_PATH, MODEL_PATH, LOGS_PATH):
        d.mkdir(parents=True, exist_ok=True)


# Waveform signal channels of interest and the target sampling rate.
# ECG lead preference order (use the first available); Pleth and Resp are single.
WAVEFORM_FS = 62.5
ECG_LEAD_PREFERENCE = ["II", "I", "III", "V", "aVR"]
PLETH_CHANNEL = "Pleth"
RESP_CHANNEL = "Resp"
TIER2_RESP_COVERAGE_MIN = 0.50   # resp coverage fraction for tier2 eligibility


def waveform_patient_dir(subject_id: int) -> Path:
    """waves/p{subject_id[:3]}/p{subject_id}/ — e.g. 10014354 → waves/p100/p10014354."""
    group = f"p{str(subject_id)[:3]}"
    patient = f"p{subject_id}"
    return WAVEFORM_WAVES / group / patient


def waveform_record_base(subject_id: int, record_id: str) -> Path:
    """Full WFDB record base path (no extension).

    Real layout (corrects the original spec): each subject folder contains one or
    more numeric record folders, and the multi-segment record header lives at
    ``waves/pGROUP/pSUBJECT/RECORDID/RECORDID.hea``. Pass the returned path
    (without ``.hea``) to ``wfdb.rdrecord`` / ``wfdb.rdheader``.

    Example: subject 10014354, record 81739927 →
      files/mimic4wdb/0.1.0/waves/p100/p10014354/81739927/81739927
    """
    return waveform_patient_dir(subject_id) / record_id / record_id


def list_subject_records(subject_id: int) -> list[str]:
    """Return the record ids available for a subject (folder names), or []."""
    pdir = waveform_patient_dir(subject_id)
    if not pdir.exists():
        return []
    return sorted(p.name for p in pdir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.hea").exists())
