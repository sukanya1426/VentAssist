"""Track B cohort construction — waveform ∩ Track-A-episode intersection.

Inverts the join direction of the Track A cohort: instead of starting from the
ICU cohort and looking for waveforms (which yielded only ~2 time-aligned
episodes), we start from the 198 mimic4wdb waveform recordings and intersect each
recording window with the ventilation episodes Track A already derived.

Algorithm:
  1. Parse every waveform record header → (subject_id, record_id, rec_start,
     rec_end, channels). [fast, header-only, no signal read]
  2. Load the Track A cohort (``cohort.csv``) — ventilation episodes derived from
     ``procedureevents`` itemid 225792, which carries EXPLICIT start/end times.
  3. The Track B EPISODE is the intersection of a recording window with a Track A
     ventilation episode for the same subject; keep intersections with
     >= 4 h overlap and a Resp channel present.
  4. Keep the single longest-overlap episode per stay, so (stay_id, hour) stays
     unique for the aggregator and the MDP builder.

WHY THE INTERSECTION, AND NOT A CHARTEVENTS RE-DERIVATION (the 2026-10-02 fix):
this stage used to rebuild its own ventilation windows by merging sparse
``chartevents`` PEEP/TV setting events with a 2 h gap — the approach Track A
ABANDONED (see cohort.py) precisely because it fragments a continuous ventilation
course. Settings are charted roughly every 4 h, so a 2 h merge gap turns each
charting interval into its own "episode" and the >= 4 h filter then discards
almost all of them. The funnel was: 111 records with overlap → 15 survivors.
Concretely, subject 13240081 / record 87706224 came out as three separate 4.0 h
fragments, where the true procedureevents window overlapping that recording is
51.1 CONTINUOUS hours.

Intersecting the authoritative Track A episodes instead yields **37 episodes /
37 stays / ~1223 overlap-hours** (vs 15 / 13 / ~83) — about 12x the Track B
training data from the same files, with no change to the raw data.

Two further consequences of reusing the Track A cohort, both improvements:
  * Its inclusion/exclusion criteria (age >= 18, >= 6 continuous vent hours, no
    cardiac arrest, no ECMO) are inherited for free, so they are applied exactly
    once and identically on both tracks.
  * ``age`` / ``weight_kg`` / ``sepsis_flag`` are the real per-stay values from
    procedureevents ``patientweight`` and the ICD tables, replacing the
    population-default ``weight_kg = 80.0`` this stage used to stamp on every row.
No ± clock-drift buffer is applied: intersecting two real intervals is already
conservative (drift shrinks the window rather than inventing signal).

Output: data/processed/cohort_track_b.csv with columns
    [subject_id, hadm_id, stay_id, record_id, vent_start, vent_end,
     rec_start, rec_end, overlap_hours, resp_coverage_frac,
     has_ecg, has_pleth, has_resp, age, weight_kg, sepsis_flag,
     patient_known, age_imputed]

Run:
    python -m backend.pipeline.cohort_track_b
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd
import wfdb

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("cohort_track_b")

_ECG_LEADS = set(config.ECG_LEAD_PREFERENCE)
_MIN_OVERLAP_H = 4.0

# Columns carried over verbatim from the Track A cohort row.
_INHERITED = ["hadm_id", "age", "weight_kg", "sepsis_flag",
              "patient_known", "age_imputed"]


# --------------------------------------------------------------------------- #
# Step 1 — waveform recording windows
# --------------------------------------------------------------------------- #
def _all_waveform_subjects() -> list[int]:
    subs = []
    for g in glob.glob(str(config.WAVEFORM_WAVES / "p*")):
        for p in glob.glob(os.path.join(g, "p*")):
            m = re.match(r"p(\d+)$", os.path.basename(p))
            if m:
                subs.append(int(m.group(1)))
    return sorted(set(subs))


def _scan_segment_channels(base: str) -> set[str]:
    chans: set[str] = set()
    try:
        full = wfdb.rdheader(base, rd_segments=True)
        for seg in getattr(full, "segments", []) or []:
            if seg is not None and getattr(seg, "sig_name", None):
                chans.update(seg.sig_name)
    except Exception:
        pass
    return chans


def waveform_windows() -> pd.DataFrame:
    """Read every waveform record header → recording window + channel presence."""
    rows = []
    for subject in _all_waveform_subjects():
        for rid in config.list_subject_records(subject):
            base = str(config.waveform_record_base(subject, rid))
            try:
                h = wfdb.rdheader(base, rd_segments=False)
            except Exception as exc:
                log.warning("header read failed %s: %s", base, exc)
                continue
            if h.base_date is None or h.base_time is None or not h.fs or not h.sig_len:
                continue
            start = pd.Timestamp.combine(h.base_date, h.base_time)
            end = start + pd.Timedelta(seconds=h.sig_len / h.fs)
            chans = set(h.sig_name) if getattr(h, "sig_name", None) \
                else _scan_segment_channels(base)
            rows.append({
                "subject_id": subject, "record_id": rid,
                "rec_start": start, "rec_end": end,
                "has_ecg": bool(chans & _ECG_LEADS),
                "has_pleth": config.PLETH_CHANNEL in chans,
                "has_resp": config.RESP_CHANNEL in chans,
            })
    df = pd.DataFrame(rows)
    log.info("Parsed %d waveform records across %d subjects.",
             len(df), df["subject_id"].nunique() if not df.empty else 0)
    return df


# --------------------------------------------------------------------------- #
# Step 2 — the authoritative ventilation episodes
# --------------------------------------------------------------------------- #
def _track_a_episodes() -> pd.DataFrame:
    """Load the Track A cohort — procedureevents-derived, explicit start/end."""
    path = config.PROCESSED_PATH / "cohort.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Track B now intersects the Track A ventilation "
            "episodes, so the Track A cohort stage must run first "
            "(python -m backend.pipeline.cohort)."
        )
    epi = pd.read_csv(path, parse_dates=["vent_start", "vent_end"])
    log.info("Track A episodes: %d across %d subjects.",
             len(epi), epi["subject_id"].nunique())
    return epi


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    config.ensure_output_dirs()
    wins = waveform_windows()
    if wins.empty:
        raise ValueError("No waveform recording windows found.")
    epi = _track_a_episodes()

    # Every (recording, ventilation-episode) pair for the same subject.
    pairs = wins.merge(epi, on="subject_id", how="inner")
    funnel = {"records": len(wins), "subjects_with_vent": pairs["subject_id"].nunique(),
              "candidate_pairs": len(pairs)}
    if pairs.empty:
        log.warning("No waveform subject appears in the Track A cohort.")
        return _write(pd.DataFrame(), funnel)

    pairs["ov_start"] = pairs[["rec_start", "vent_start"]].max(axis=1)
    pairs["ov_end"] = pairs[["rec_end", "vent_end"]].min(axis=1)
    pairs["overlap_hours"] = (pairs["ov_end"] - pairs["ov_start"]) / pd.Timedelta(hours=1)

    pairs = pairs[pairs["overlap_hours"] > 0]
    funnel["pairs_with_overlap"] = len(pairs)
    pairs = pairs[pairs["overlap_hours"] >= _MIN_OVERLAP_H]
    funnel[f"passed_overlap_{int(_MIN_OVERLAP_H)}h"] = len(pairs)
    pairs = pairs[pairs["has_resp"]]
    funnel["passed_resp"] = len(pairs)

    # Truncate to the episode-length cap (Track A episodes are already capped, so
    # this only binds if that cap is ever raised).
    pairs["vent_start"] = pairs["ov_start"]
    pairs["vent_end"] = pairs["ov_end"].where(
        pairs["ov_end"] <= pairs["ov_start"] + pd.Timedelta(hours=config.MAX_EPISODE_HOURS),
        pairs["ov_start"] + pd.Timedelta(hours=config.MAX_EPISODE_HOURS))

    # One episode per stay — the longest overlap — so that the aggregator and the
    # MDP builder see a unique (stay_id, hour) key.
    pairs = (pairs.sort_values("overlap_hours", ascending=False)
                  .drop_duplicates(subset=["stay_id"], keep="first"))
    funnel["unique_stays"] = len(pairs)

    cohort = pairs[["subject_id", "stay_id", "record_id",
                    "vent_start", "vent_end", "rec_start", "rec_end",
                    "overlap_hours", "has_ecg", "has_pleth", "has_resp",
                    *_INHERITED]].copy()
    cohort["overlap_hours"] = cohort["overlap_hours"].round(2)
    cohort["resp_coverage_frac"] = 1.0      # refined per-hour by the aggregator
    cohort = cohort[["subject_id", "hadm_id", "stay_id", "record_id",
                     "vent_start", "vent_end", "rec_start", "rec_end",
                     "overlap_hours", "resp_coverage_frac",
                     "has_ecg", "has_pleth", "has_resp",
                     "age", "weight_kg", "sepsis_flag",
                     "patient_known", "age_imputed"]]
    return _write(cohort.sort_values("stay_id").reset_index(drop=True), funnel)


def _write(cohort: pd.DataFrame, funnel: dict) -> pd.DataFrame:
    out = config.PROCESSED_PATH / "cohort_track_b.csv"
    cohort.to_csv(out, index=False)
    log.info("Track B funnel: %s", funnel)
    if cohort.empty:
        log.warning("Track B cohort is EMPTY → %s", out)
        return cohort
    log.info("Track B cohort: %d episodes across %d subjects, "
             "%.0f overlap-hours (median %.1f h) → %s",
             len(cohort), cohort["subject_id"].nunique(),
             cohort["overlap_hours"].sum(), cohort["overlap_hours"].median(), out)
    if len(cohort) < 10:
        log.warning("Track B cohort < 10 episodes — waveform/vent overlap is "
                    "scarce in this data. Note Track B as a proof-of-concept.")
    return cohort


def main() -> None:
    build()


if __name__ == "__main__":
    main()
