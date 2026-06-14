"""Stage 3 — Build the 12-dimensional TIER 1 hourly state vector per episode.

For every episode and every 1-hour epoch we assemble the canonical 12-feature
state:
    [PEEP, TV, FiO2, SpO2, PaO2, PaCO2, pH, HR, SBP, RR, RASS, Temp]

Processing:
  1. Hourly grid from vent_start..vent_end.
  2. Aggregate chartevents within each hour (median of valuenum per feature),
     with FiO2 and Temperature unit normalisation.
  3. Forward-fill per feature up to its configured gap horizon.
  4. Merge GP-imputed PaO2/PaCO2/pH on (stay_id, hour).
  5. Physiological clipping.
  6. Compute winsor bounds (1st/99th pct) and z-score stats on the TRAIN split
     only, and persist them to models/normaliser_stats.json.

DESIGN NOTE: ``tabular_states.parquet`` stores RAW (physiologically-clipped,
un-normalised) values. Normalisation is applied on the fly at train/inference
time via the saved stats. This is deliberate — the reward function
(mdp/reward.py) requires raw physiological values, so baking normalisation into
the stored states would break reward computation.

Inputs : cohort.csv, gp_imputed_labs.parquet, chartevents.csv.gz (chunked)
Outputs: data/processed/tabular_states.parquet
         models/normaliser_stats.json
         data/processed/train_val_test_split.json

Run:
    python -m backend.pipeline.state_builder [--sample]
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd

from backend.pipeline import config, splits
from backend.pipeline.logging_utils import get_logger

log = get_logger("state_builder")

# chartevents itemid → feature (PaO2/PaCO2/pH excluded — they come from GP labs)
_ITEMID_TO_FEATURE = {
    iid: feat for feat, ids in config.CHART_FEATURE_ITEMIDS.items() for iid in ids
}
_CHART_ITEMIDS = set(_ITEMID_TO_FEATURE)
_CHART_FEATURES = list(config.CHART_FEATURE_ITEMIDS)   # 9 chartevents features


def scan_feature_chartevents(stay_ids: set[int],
                             sample_chunks: Optional[int] = None) -> pd.DataFrame:
    """Chunked scan keeping chartevents rows for our features and cohort stays."""
    log.info("Scanning chartevents for state features (%d stays)…", len(stay_ids))
    keep: list[pd.DataFrame] = []
    reader = pd.read_csv(
        config.CHARTEVENTS,
        chunksize=config.CHUNK_SIZE,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "float64"},
        parse_dates=["charttime"],
    )
    for i, chunk in enumerate(reader):
        hit = chunk[chunk["itemid"].isin(_CHART_ITEMIDS)
                    & chunk["stay_id"].isin(stay_ids)]
        if not hit.empty:
            keep.append(hit)
        if sample_chunks is not None and (i + 1) >= sample_chunks:
            break
        if (i + 1) % 50 == 0:
            log.info("  …processed %d chunks", i + 1)
    if not keep:
        raise ValueError("No state-feature chartevents found for the cohort.")
    ev = pd.concat(keep, ignore_index=True)
    ev["charttime"] = pd.to_datetime(ev["charttime"], errors="coerce")
    ev = ev.dropna(subset=["valuenum", "charttime"])
    ev["stay_id"] = ev["stay_id"].astype(int)
    ev["feature"] = ev["itemid"].map(_ITEMID_TO_FEATURE)
    log.info("Collected %d state-feature rows.", len(ev))
    return ev


def _normalise_units(feature: str, values: pd.Series) -> pd.Series:
    """FiO2 percent→fraction; Temperature Fahrenheit→Celsius."""
    if feature == "FiO2":
        values = values.where(values <= 1.0, values / 100.0)
    elif feature == "Temp":
        values = values.where(values <= 45.0, (values - 32.0) * 5.0 / 9.0)
    return values


def _build_episode_states(ep: pd.Series, ev_stay: pd.DataFrame,
                          labs_stay: pd.DataFrame) -> pd.DataFrame:
    """Assemble the hourly raw state grid for a single episode."""
    v_start, v_end = ep["vent_start"], ep["vent_end"]
    n_hours = int(np.floor((v_end - v_start) / pd.Timedelta(hours=1))) + 1
    grid = pd.DataFrame({"stay_id": int(ep["stay_id"]), "hour": np.arange(n_hours)})

    if not ev_stay.empty:
        ev_stay = ev_stay.copy()
        ev_stay["hour"] = np.floor(
            (ev_stay["charttime"] - v_start) / pd.Timedelta(hours=1)).astype("Int64")
        ev_stay = ev_stay[(ev_stay["hour"] >= 0) & (ev_stay["hour"] < n_hours)]

    for feat in _CHART_FEATURES:
        col = pd.Series(np.nan, index=grid.index)
        if not ev_stay.empty:
            sub = ev_stay[ev_stay["feature"] == feat]
            if not sub.empty:
                vals = _normalise_units(feat, sub["valuenum"])
                agg = vals.groupby(sub["hour"]).median()
                col.loc[agg.index.astype(int)] = agg.values
        # physiological clip
        lo, hi = config.CHART_CLIP_RANGES[feat]
        col = col.clip(lo, hi)
        # forward-fill up to the per-feature horizon (hours == rows)
        limit = config.FORWARD_FILL_HOURS[feat]
        col = col.ffill(limit=limit)
        grid[feat] = col.values

    # merge GP-imputed labs
    grid = grid.merge(labs_stay, on=["stay_id", "hour"], how="left")
    return grid


def _fit_normaliser(train_states: pd.DataFrame) -> dict:
    """Winsor bounds (1st/99th pct) + post-winsor mean/std per feature, on train."""
    stats: dict[str, dict[str, float]] = {}
    for feat in config.TABULAR_FEATURES:
        vals = train_states[feat].dropna().to_numpy()
        if len(vals) == 0:
            stats[feat] = {"mean": 0.0, "std": 1.0, "winsor_low": 0.0, "winsor_high": 0.0}
            continue
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        w = np.clip(vals, lo, hi)
        std = float(np.std(w))
        stats[feat] = {
            "mean": float(np.mean(w)),
            "std": std if std > 1e-8 else 1.0,
            "winsor_low": lo,
            "winsor_high": hi,
        }
    return stats


def build_states(sample_chunks: Optional[int] = None,
                 cohort_file: str = "cohort.csv",
                 labs_file: str = "gp_imputed_labs.parquet",
                 states_file: str = "tabular_states.parquet",
                 normaliser_file: str = "normaliser_stats.json",
                 split_file: str = "train_val_test_split.json") -> pd.DataFrame:
    """Build hourly states, fit+save the normaliser, write the states parquet.

    File names are parameterised so the same module serves both Track A
    (defaults) and Track B (``*_track_b`` files).
    """
    config.ensure_output_dirs()
    cohort = pd.read_csv(config.PROCESSED_PATH / cohort_file,
                         parse_dates=["vent_start", "vent_end"])
    labs = pd.read_parquet(config.PROCESSED_PATH / labs_file)
    stay_ids = set(cohort["stay_id"].astype(int))

    ev = scan_feature_chartevents(stay_ids, sample_chunks=sample_chunks)
    ev_by_stay = dict(tuple(ev.groupby("stay_id")))
    labs_by_stay = dict(tuple(labs.groupby("stay_id")))

    episodes = []
    for _, ep in cohort.iterrows():
        sid = int(ep["stay_id"])
        grid = _build_episode_states(
            ep,
            ev_by_stay.get(sid, pd.DataFrame(columns=ev.columns)),
            labs_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "hour", *config.LAB_FEATURES])),
        )
        episodes.append(grid)

    states = pd.concat(episodes, ignore_index=True)
    # order columns canonically
    states = states[["stay_id", "hour", *config.TABULAR_FEATURES]]

    # deterministic split + fit normaliser on TRAIN only.
    # Always (re)generate so the split matches the CURRENT cohort — it is
    # seeded, so re-running on an unchanged cohort reproduces the same split,
    # but a changed cohort correctly gets a fresh partition.
    split = splits.make_split(stay_ids, name=split_file)
    train_states = states[states["stay_id"].isin(set(split["train"]))]
    if train_states.empty:   # tiny sample where train slice is empty
        log.warning("Train split empty; fitting normaliser on all states.")
        train_states = states
    stats = _fit_normaliser(train_states)
    norm_path = config.MODEL_PATH / normaliser_file
    norm_path.write_text(json.dumps(stats, indent=2))
    log.info("Wrote normaliser stats (%d features) → %s", len(stats), norm_path)

    out = config.PROCESSED_PATH / states_file
    states.to_parquet(out, index=False)
    miss = states[config.TABULAR_FEATURES].isna().mean().round(3).to_dict()
    log.info("Wrote %d hourly states (%d stays) → %s",
             len(states), states["stay_id"].nunique(), out)
    log.info("Per-feature NaN fraction (raw, pre-impute): %s", miss)
    return states


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 12-dim TIER 1 hourly states.")
    ap.add_argument("--sample", action="store_true",
                    help="Limit chartevents scan for a fast run.")
    ap.add_argument("--sample-chunks", type=int, default=60)
    args = ap.parse_args()
    build_states(sample_chunks=args.sample_chunks if args.sample else None)


if __name__ == "__main__":
    main()
