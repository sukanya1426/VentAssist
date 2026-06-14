"""Waveform stage 4.7 — aggregate per-hour waveform features for Track B.

For each Track B episode and each 1-hour epoch, load the waveform window,
preprocess each channel, and extract the 6 features. A feature is accepted only
if its channel's quality flag passes AND >= 80% of the hour has finite signal;
otherwise it is set to NaN (imputed later at inference, never at training).

Output: data/processed/waveform_features_track_b.parquet
    [stay_id, hour, HRV_SDNN, Arrhythmia_rate, Perfusion_Index,
     RRV, Breathing_Regularity, Asynchrony_Score]

Run:
    python -m backend.waveform.aggregator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.waveform import ecg_features as E
from backend.waveform import pleth_features as PL
from backend.waveform import preprocessor as P
from backend.waveform import resp_features as R
from backend.waveform.loader import load_episode_waveform

log = get_logger("waveform_aggregator")

WAVEFORM_FEATURES = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
                     "RRV", "Breathing_Regularity", "Asynchrony_Score"]
_MIN_VALID_FRACTION = 0.80


def _hour_features(channel: str, x: np.ndarray, fs: float, extractor) -> dict:
    """Preprocess one hour of a channel, gate on quality+coverage, extract."""
    pc = P.preprocess_hour(channel, x, fs)
    feats = extractor(pc.signal, fs)
    if not pc.quality_ok or pc.valid_fraction < _MIN_VALID_FRACTION:
        return {k: np.nan for k in feats}
    return feats


def aggregate() -> pd.DataFrame:
    config.ensure_output_dirs()
    cohort = pd.read_csv(config.PROCESSED_PATH / "cohort_track_b.csv",
                         parse_dates=["vent_start", "vent_end"])

    rows: list[dict] = []
    for _, ep in cohort.iterrows():
        sid = int(ep["stay_id"])
        w = load_episode_waveform(int(ep["subject_id"]), str(ep["record_id"]),
                                  ep["vent_start"], ep["vent_end"])
        if w is None:
            continue
        fs = w.fs
        n = int(round(3600 * fs))
        ep_hours = int(np.floor((ep["vent_end"] - ep["vent_start"]) / pd.Timedelta(hours=1)))
        for hour in range(max(1, ep_hours)):
            sl = slice(hour * n, (hour + 1) * n)
            feat = {"stay_id": sid, "hour": hour}
            # ECG
            ecg = w.ECG[sl] if w.ECG is not None else None
            feat.update(_hour_features("ECG", ecg, fs, E.ecg_features)
                        if ecg is not None and len(ecg) > fs
                        else {"HRV_SDNN": np.nan, "Arrhythmia_rate": np.nan})
            # Pleth
            pleth = w.Pleth[sl] if w.Pleth is not None else None
            feat.update(_hour_features("Pleth", pleth, fs, PL.pleth_features)
                        if pleth is not None and len(pleth) > fs
                        else {"Perfusion_Index": np.nan})
            # Resp
            resp = w.Resp[sl] if w.Resp is not None else None
            feat.update(_hour_features("Resp", resp, fs, R.resp_features)
                        if resp is not None and len(resp) > fs
                        else {"RRV": np.nan, "Breathing_Regularity": np.nan,
                              "Asynchrony_Score": np.nan})
            rows.append(feat)
        log.info("  stay %d: %d hours processed", sid, max(1, ep_hours))

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No waveform features produced.")
        return df
    df = df[["stay_id", "hour", *WAVEFORM_FEATURES]]
    out = config.PROCESSED_PATH / "waveform_features_track_b.parquet"
    df.to_parquet(out, index=False)

    cov = {f: f"{df[f].notna().mean():.0%}" for f in WAVEFORM_FEATURES}
    log.info("Wrote %d (stay,hour) waveform-feature rows → %s", len(df), out)
    log.info("Per-feature non-NaN coverage: %s", cov)
    return df


def main() -> None:
    aggregate()


if __name__ == "__main__":
    main()
