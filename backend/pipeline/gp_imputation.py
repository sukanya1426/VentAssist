"""Stage 2 — Gaussian-Process imputation of sparse blood-gas labs.

PaO2, PaCO2 and pH are measured every 4–6 h but the MDP needs hourly values.
For each episode and each feature we fit a GP with a Matérn-3/2 kernel
(lengthscale = 2.0 h, observation noise = 0.1) and predict at every integer hour
of the ventilation window, clipping to physiologically plausible ranges.

Fallback: episodes with < 2 observed values for a feature use population-median
imputation (logged as a warning).

Inputs : data/processed/cohort.csv, files/.../labevents.csv.gz (chunked)
Output : data/processed/gp_imputed_labs.parquet
         columns [stay_id, hour, PaO2, PaCO2, pH]

Run:
    python -m backend.pipeline.gp_imputation [--sample]
"""

from __future__ import annotations

import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("gp_imputation")

_ITEMID_TO_FEATURE = {iid: feat for feat, ids in config.LAB_ITEMIDS.items() for iid in ids}
_ALL_LAB_ITEMIDS = set(_ITEMID_TO_FEATURE)


# --------------------------------------------------------------------------- #
# GP posterior with fixed hyperparameters (lazy torch/gpytorch import)
# --------------------------------------------------------------------------- #
def _gp_predict(t_obs: np.ndarray, y_obs: np.ndarray, t_query: np.ndarray,
                lengthscale: float = 2.0, noise: float = 0.1) -> np.ndarray:
    """Posterior mean of a Matérn-3/2 GP with fixed hyperparameters.

    Uses GPyTorch's MaternKernel(nu=1.5). No marginal-likelihood optimisation —
    hyperparameters are fixed per the methodology.
    """
    import gpytorch
    import torch

    tx = torch.as_tensor(t_obs, dtype=torch.float64).reshape(-1, 1)
    ty = torch.as_tensor(y_obs, dtype=torch.float64)
    qx = torch.as_tensor(t_query, dtype=torch.float64).reshape(-1, 1)

    kernel = gpytorch.kernels.MaternKernel(nu=1.5)
    kernel.lengthscale = lengthscale
    with torch.no_grad():
        K = kernel(tx).to_dense().double() + noise * torch.eye(len(tx), dtype=torch.float64)
        Ks = kernel(qx, tx).to_dense().double()
        mean_y = ty.mean()
        alpha = torch.linalg.solve(K, (ty - mean_y))
        mu = mean_y + Ks @ alpha
    return mu.cpu().numpy()


# --------------------------------------------------------------------------- #
# Load relevant labevents for the cohort subjects
# --------------------------------------------------------------------------- #
def load_labs(subject_ids: set[int], sample: bool = False) -> pd.DataFrame:
    """Chunked scan of labevents → blood-gas rows for the cohort subjects."""
    log.info("Scanning labevents for %d cohort subjects…", len(subject_ids))
    keep: list[pd.DataFrame] = []
    reader = pd.read_csv(
        config.LABEVENTS,
        chunksize=config.CHUNK_SIZE,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        dtype={"subject_id": "Int64", "hadm_id": "Int64",
               "itemid": "Int64", "valuenum": "float64"},
        parse_dates=["charttime"],
    )
    for i, chunk in enumerate(reader):
        hit = chunk[chunk["itemid"].isin(_ALL_LAB_ITEMIDS)
                    & chunk["subject_id"].isin(subject_ids)]
        if not hit.empty:
            keep.append(hit)
        if sample and (i + 1) >= 200:   # cap labevents scan in sample mode
            break
        if (i + 1) % 100 == 0:
            log.info("  …processed %d labevents chunks", i + 1)
    if not keep:
        log.warning("No matching lab rows found; all imputation will use fallbacks.")
        return pd.DataFrame(columns=["subject_id", "itemid", "charttime", "valuenum"])
    labs = pd.concat(keep, ignore_index=True)
    # Force datetime dtype: concatenating chunks can coerce charttime to object
    # if any chunk had empty/unparseable times, which breaks later arithmetic.
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs = labs.dropna(subset=["valuenum", "charttime"])
    labs["feature"] = labs["itemid"].map(_ITEMID_TO_FEATURE)
    log.info("Collected %d blood-gas lab measurements.", len(labs))
    return labs


def _population_medians(labs: pd.DataFrame) -> dict[str, float]:
    med = {}
    defaults = {"PaO2": 90.0, "PaCO2": 40.0, "pH": 7.4}
    for feat in config.LAB_FEATURES:
        vals = labs.loc[labs.get("feature") == feat, "valuenum"] if not labs.empty else []
        med[feat] = float(np.median(vals)) if len(vals) else defaults[feat]
    return med


def impute(sample_chunks: Optional[int] = None,
           cohort_file: str = "cohort.csv",
           out_file: str = "gp_imputed_labs.parquet") -> pd.DataFrame:
    """Run GP imputation over the cohort and write the parquet output."""
    config.ensure_output_dirs()
    cohort = pd.read_csv(config.PROCESSED_PATH / cohort_file,
                         parse_dates=["vent_start", "vent_end"])
    subject_ids = set(cohort["subject_id"].astype(int))
    labs = load_labs(subject_ids, sample=sample_chunks is not None)
    pop_med = _population_medians(labs)
    log.info("Population medians: %s", {k: round(v, 2) for k, v in pop_med.items()})

    labs_by_subject = dict(tuple(labs.groupby("subject_id"))) if not labs.empty else {}

    out_rows: list[dict] = []
    fallback_count = {f: 0 for f in config.LAB_FEATURES}

    for _, ep in cohort.iterrows():
        stay_id = int(ep["stay_id"])
        subject_id = int(ep["subject_id"])
        v_start, v_end = ep["vent_start"], ep["vent_end"]
        n_hours = int(np.floor((v_end - v_start) / pd.Timedelta(hours=1))) + 1
        hours = np.arange(n_hours, dtype=float)

        per_feat = {"stay_id": stay_id, "hour": hours}
        subj_labs = labs_by_subject.get(subject_id, pd.DataFrame())

        for feat in config.LAB_FEATURES:
            lo, hi = config.LAB_CLIP_RANGES[feat]
            if not subj_labs.empty:
                fl = subj_labs[(subj_labs["feature"] == feat)
                               & (subj_labs["charttime"] >= v_start)
                               & (subj_labs["charttime"] <= v_end)]
            else:
                fl = pd.DataFrame()

            if len(fl) >= 2:
                t_obs = ((fl["charttime"] - v_start) / pd.Timedelta(hours=1)).to_numpy()
                y_obs = np.clip(fl["valuenum"].to_numpy(), lo, hi)
                # collapse duplicate timestamps (GP needs distinct inputs)
                order = np.argsort(t_obs)
                t_obs, y_obs = t_obs[order], y_obs[order]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred = _gp_predict(t_obs, y_obs, hours)
                    pred = np.clip(pred, lo, hi)
                except Exception as exc:  # numerical issues → fallback
                    log.warning("GP failed (stay %d, %s): %s — using median.",
                                stay_id, feat, exc)
                    pred = np.full(n_hours, pop_med[feat])
                    fallback_count[feat] += 1
            else:
                pred = np.full(n_hours, pop_med[feat])
                fallback_count[feat] += 1

            per_feat[feat] = pred

        out_rows.append(pd.DataFrame(per_feat))

    result = pd.concat(out_rows, ignore_index=True)
    result["hour"] = result["hour"].astype(int)
    out = config.PROCESSED_PATH / out_file
    result.to_parquet(out, index=False)
    log.info("Wrote %d (stay, hour) imputed lab rows → %s", len(result), out)
    log.info("Median-fallback episode counts per feature: %s", fallback_count)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="GP-impute sparse blood-gas labs.")
    ap.add_argument("--sample", action="store_true",
                    help="Cap the labevents scan for a fast run.")
    args = ap.parse_args()
    impute(sample_chunks=1 if args.sample else None)


if __name__ == "__main__":
    main()
