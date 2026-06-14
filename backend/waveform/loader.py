"""Waveform stage 4.2 — load WFDB signals aligned to a ventilation episode.

Reads a (multi-segment) WFDB record, slices it to the episode's
[vent_start, vent_end] window using the header's base_time, and returns the ECG
(preferred lead), Pleth and Resp channels as numpy arrays.

Real record layout (corrects the spec): waves/pGROUP/pSUBJECT/RECORDID/RECORDID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("waveform_loader")


@dataclass
class LoadedWaveform:
    ECG: Optional[np.ndarray]
    Pleth: Optional[np.ndarray]
    Resp: Optional[np.ndarray]
    fs: float
    t_start_hours: float          # offset of the slice from vent_start, in hours
    ecg_lead: Optional[str]
    rec_start: pd.Timestamp


def _pick_ecg(sig_name: list[str]) -> Optional[str]:
    for lead in config.ECG_LEAD_PREFERENCE:
        if lead in sig_name:
            return lead
    return None


def load_episode_waveform(subject_id: int, record_id: str,
                          vent_start: pd.Timestamp,
                          vent_end: pd.Timestamp) -> Optional[LoadedWaveform]:
    """Load and window a record to [vent_start, vent_end]. None if unavailable."""
    base = str(config.waveform_record_base(subject_id, record_id))
    try:
        hdr = wfdb.rdheader(base, rd_segments=False)
    except Exception as exc:
        log.warning("header read failed %s: %s", base, exc)
        return None
    if hdr.base_date is None or hdr.base_time is None or not hdr.fs:
        return None

    fs = float(hdr.fs)
    rec_start = pd.Timestamp.combine(hdr.base_date, hdr.base_time)
    # sample offsets of the episode window within the record
    a = int(max(0.0, (vent_start - rec_start).total_seconds()) * fs)
    b = int(max(0.0, (vent_end - rec_start).total_seconds()) * fs)
    b = min(b, int(hdr.sig_len))
    if b <= a:
        return None

    try:
        rec = wfdb.rdrecord(base, sampfrom=a, sampto=b)
    except Exception as exc:
        log.warning("signal read failed %s [%d:%d]: %s", base, a, b, exc)
        return None

    sig = rec.p_signal
    names = list(rec.sig_name)

    def col(name: Optional[str]) -> Optional[np.ndarray]:
        if name is None or name not in names:
            return None
        arr = sig[:, names.index(name)].astype(np.float64)
        return arr if np.isfinite(arr).any() else None

    ecg_lead = _pick_ecg(names)
    t_start_hours = max(0.0, (vent_start - rec_start).total_seconds()) / 3600.0
    return LoadedWaveform(
        ECG=col(ecg_lead),
        Pleth=col(config.PLETH_CHANNEL),
        Resp=col(config.RESP_CHANNEL),
        fs=fs,
        t_start_hours=t_start_hours,
        ecg_lead=ecg_lead,
        rec_start=rec_start,
    )
