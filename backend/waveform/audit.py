"""Waveform stage 4.1 — audit which cohort episodes have usable waveform data.

Maps each cohort ventilation episode to the subject's available WFDB records,
reads each record's master header (start time, duration, channels), and computes
the temporal overlap between the waveform recording and the ventilation window.

CORRECTS THE SPEC PATH LAYOUT: records live at
    waves/pGROUP/pSUBJECT/RECORDID/RECORDID.hea
(multi-segment, numeric RECORDID), not ``pSUBJECT-date``.

Output: data/processed/waveform_audit.csv with columns
    [stay_id, subject_id, record_id, wf_start, wf_end,
     has_ecg, has_pleth, has_resp,
     ecg_coverage_frac, pleth_coverage_frac, resp_coverage_frac,
     overlap_hours, episode_hours, tier2_eligible]

Run:
    python -m backend.waveform.audit
"""

from __future__ import annotations

import pandas as pd
import wfdb

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("waveform_audit")

_ECG_LEADS = set(config.ECG_LEAD_PREFERENCE)


def _record_window(subject_id: int, record_id: str):
    """Return (start_ts, end_ts, channel_set) for a record, or None on failure."""
    base = str(config.waveform_record_base(subject_id, record_id))
    try:
        h = wfdb.rdheader(base, rd_segments=False)
    except Exception as exc:  # pragma: no cover - corrupt/missing header
        log.warning("Failed to read header %s: %s", base, exc)
        return None
    if h.base_date is None or h.base_time is None or not h.fs or not h.sig_len:
        return None
    start = pd.Timestamp.combine(h.base_date, h.base_time)
    end = start + pd.Timedelta(seconds=h.sig_len / h.fs)
    # Channels: master header of a multi-segment record may not list sig_name;
    # fall back to scanning segment headers' signal names.
    chans = set(h.sig_name) if getattr(h, "sig_name", None) else _scan_segment_channels(base)
    return start, end, chans


def _scan_segment_channels(base: str) -> set[str]:
    """Union of signal names across a multi-segment record's segments."""
    chans: set[str] = set()
    try:
        full = wfdb.rdheader(base, rd_segments=True)
        for seg in getattr(full, "segments", []) or []:
            if seg is not None and getattr(seg, "sig_name", None):
                chans.update(seg.sig_name)
    except Exception:
        pass
    return chans


def audit() -> pd.DataFrame:
    config.ensure_output_dirs()
    cohort = pd.read_csv(config.PROCESSED_PATH / "cohort.csv",
                         parse_dates=["vent_start", "vent_end"])

    rows: list[dict] = []
    n_with_records = 0
    for _, ep in cohort.iterrows():
        sid, subject = int(ep["stay_id"]), int(ep["subject_id"])
        v_start, v_end = ep["vent_start"], ep["vent_end"]
        ep_hours = (v_end - v_start) / pd.Timedelta(hours=1)
        records = config.list_subject_records(subject)
        if not records:
            continue
        n_with_records += 1
        for rid in records:
            win = _record_window(subject, rid)
            if win is None:
                continue
            wf_start, wf_end, chans = win
            # temporal overlap between waveform and vent window
            ov_start = max(v_start, wf_start)
            ov_end = min(v_end, wf_end)
            overlap_h = max(0.0, (ov_end - ov_start) / pd.Timedelta(hours=1))
            has_ecg = bool(chans & _ECG_LEADS)
            has_pleth = config.PLETH_CHANNEL in chans
            has_resp = config.RESP_CHANNEL in chans
            cov = (overlap_h / ep_hours) if ep_hours > 0 else 0.0
            rows.append({
                "stay_id": sid, "subject_id": subject, "record_id": rid,
                "wf_start": wf_start, "wf_end": wf_end,
                "has_ecg": has_ecg, "has_pleth": has_pleth, "has_resp": has_resp,
                "ecg_coverage_frac": round(cov if has_ecg else 0.0, 3),
                "pleth_coverage_frac": round(cov if has_pleth else 0.0, 3),
                "resp_coverage_frac": round(cov if has_resp else 0.0, 3),
                "overlap_hours": round(overlap_h, 2),
                "episode_hours": round(ep_hours, 2),
                "tier2_eligible": bool(has_resp and cov >= config.TIER2_RESP_COVERAGE_MIN),
            })

    audit_df = pd.DataFrame(rows)
    out = config.PROCESSED_PATH / "waveform_audit.csv"
    audit_df.to_csv(out, index=False)

    n_eligible = int(audit_df["tier2_eligible"].sum()) if not audit_df.empty else 0
    n_overlap = int((audit_df["overlap_hours"] > 0).sum()) if not audit_df.empty else 0
    log.info("Cohort episodes with any waveform record: %d", n_with_records)
    log.info("Waveform records audited: %d | with vent-window overlap > 0: %d | "
             "tier2_eligible: %d", len(audit_df), n_overlap, n_eligible)
    log.info("Wrote %s", out)
    return audit_df


def main() -> None:
    audit()


if __name__ == "__main__":
    main()
