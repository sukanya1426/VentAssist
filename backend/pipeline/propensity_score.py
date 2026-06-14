"""Stage 4 — Patient propensity score (estimated 90-day mortality risk).

Fits a logistic-regression model predicting 90-day mortality and emits a scalar
``propensity_z`` in [0, 1] per episode, used as the confounder adjustment in the
Nadaraya–Watson model-based OPE.

Feature vector (per episode, assembled from cohort + first-epoch state):
    age, weight_kg, sex (F=1), sepsis_flag, icu_readmission,
    SIRS_score*, SOFA_resp*, shock_index, temp_on_admission, pf_ratio_on_admission

*SIRS_score and SOFA_resp are SIMPLIFIED proxies computed from available vitals/
labs at the first ventilation epoch — full SIRS/SOFA require additional labs
(WBC, platelets, bilirubin, creatinine, GCS) and are deferred to a dedicated
severity-score module. Documented and logged.

Target: died within 90 days of vent_start (from patients.dod / admissions).

Inputs : cohort.csv, tabular_states.parquet, patients.csv.gz, admissions.csv.gz,
         icustays.csv.gz, train_val_test_split.json
Outputs: data/processed/propensity_scores.csv  [stay_id, propensity_z]
         models/propensity_model.pkl

Run:
    python -m backend.pipeline.propensity_score
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.pipeline import config, splits
from backend.pipeline.logging_utils import get_logger

log = get_logger("propensity_score")

_FEATURES = [
    "age", "weight_kg", "sex_female", "sepsis_flag", "icu_readmission",
    "sirs_score", "sofa_resp", "shock_index", "temp_admit", "pf_ratio_admit",
]


def _first_epoch_states() -> pd.DataFrame:
    """First-hour (hour==0) raw state per stay, for admission-time features."""
    states = pd.read_parquet(config.PROCESSED_PATH / "tabular_states.parquet")
    first = (states.sort_values(["stay_id", "hour"])
             .groupby("stay_id").first().reset_index())
    return first


def _simplified_sirs(row: pd.Series) -> int:
    """0–3 proxy: temp, HR, RR criteria (WBC unavailable)."""
    s = 0
    t = row.get("Temp")
    if pd.notna(t) and (t > 38.0 or t < 36.0):
        s += 1
    if pd.notna(row.get("HR")) and row["HR"] > 90:
        s += 1
    if pd.notna(row.get("RR")) and row["RR"] > 20:
        s += 1
    return s


def _sofa_resp(row: pd.Series) -> int:
    """0–4 SOFA respiratory sub-score from P/F ratio."""
    pao2, fio2 = row.get("PaO2"), row.get("FiO2")
    if pd.isna(pao2) or pd.isna(fio2) or fio2 <= 0:
        return 0
    pf = pao2 / fio2
    if pf < 100:
        return 4
    if pf < 200:
        return 3
    if pf < 300:
        return 2
    if pf < 400:
        return 1
    return 0


def _build_features(cohort: pd.DataFrame) -> pd.DataFrame:
    patients = pd.read_csv(config.PATIENTS,
                           usecols=["subject_id", "gender", "dod"],
                           parse_dates=["dod"])
    icustays = pd.read_csv(config.ICUSTAYS,
                           usecols=["subject_id", "stay_id", "intime"],
                           parse_dates=["intime"])

    # ICU readmission: not the subject's first ICU stay (by intime).
    icustays = icustays.sort_values(["subject_id", "intime"])
    icustays["stay_rank"] = icustays.groupby("subject_id").cumcount()
    readmit = icustays.set_index("stay_id")["stay_rank"]

    first = _first_epoch_states().set_index("stay_id")
    pat = patients.set_index("subject_id")

    rows = []
    for _, ep in cohort.iterrows():
        sid, subj = int(ep["stay_id"]), int(ep["subject_id"])
        fs = first.loc[sid] if sid in first.index else pd.Series(dtype=float)
        gender = pat.loc[subj, "gender"] if subj in pat.index else None
        hr = fs.get("HR"); sbp = fs.get("SBP")
        shock = (hr / sbp) if (pd.notna(hr) and pd.notna(sbp) and sbp > 0) else np.nan
        pao2, fio2 = fs.get("PaO2"), fs.get("FiO2")
        pf = (pao2 / fio2) if (pd.notna(pao2) and pd.notna(fio2) and fio2 > 0) else np.nan

        # 90-day mortality label
        dod = pat.loc[subj, "dod"] if subj in pat.index else pd.NaT
        died_90d = int(pd.notna(dod)
                       and (dod - ep["vent_start"]) <= pd.Timedelta(days=90)
                       and (dod - ep["vent_start"]) >= pd.Timedelta(days=-1))

        rows.append({
            "stay_id": sid,
            "patient_known": int(ep.get("patient_known", 1)),
            "age": ep["age"],
            "weight_kg": ep["weight_kg"],
            "sex_female": int(gender == "F") if gender is not None else 0,
            "sepsis_flag": int(ep["sepsis_flag"]),
            "icu_readmission": int(readmit.get(sid, 0) > 0),
            "sirs_score": _simplified_sirs(fs),
            "sofa_resp": _sofa_resp(fs),
            "shock_index": shock,
            "temp_admit": fs.get("Temp"),
            "pf_ratio_admit": pf,
            "died_90d": died_90d,
        })
    feats = pd.DataFrame(rows)
    # median-impute any remaining NaNs in features
    for col in _FEATURES:
        if feats[col].isna().any():
            feats[col] = feats[col].fillna(feats[col].median())
    return feats


def build_propensity() -> pd.DataFrame:
    config.ensure_output_dirs()
    cohort = pd.read_csv(config.PROCESSED_PATH / "cohort.csv",
                         parse_dates=["vent_start", "vent_end"])
    feats = _build_features(cohort)
    split = splits.load_split()
    train_ids = set(split["train"])

    X = feats[_FEATURES].to_numpy(dtype=float)
    y = feats["died_90d"].to_numpy(dtype=int)
    # Train only on subjects with a REAL patient record (valid mortality label).
    # In degraded mode, missing-patient subjects have unreliable dod → excluded
    # from training but still scored by the fitted model from their features.
    train_mask = (feats["stay_id"].isin(train_ids)
                  & (feats["patient_known"] == 1)).to_numpy()
    known = int((feats["patient_known"] == 1).sum())
    log.info("90-day mortality rate (known patients): %.3f | "
             "n=%d, patient_known=%d, train=%d",
             y[feats["patient_known"] == 1].mean() if known else 0.0,
             len(y), known, train_mask.sum())

    X_train = X[train_mask] if train_mask.any() else X
    y_train = y[train_mask] if train_mask.any() else y

    model = None
    if len(np.unique(y_train)) < 2:
        # Degenerate (common in tiny samples): fall back to constant base rate.
        base = float(y.mean())
        z = np.full(len(feats), base)
        log.warning("Only one mortality class in train split; using base rate %.3f.", base)
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        model.fit(X_train, y_train)
        z = model.predict_proba(X)[:, 1]
        log.info("Fitted logistic propensity model on %d episodes.", train_mask.sum())

    out_df = pd.DataFrame({"stay_id": feats["stay_id"], "propensity_z": np.round(z, 4)})
    out = config.PROCESSED_PATH / "propensity_scores.csv"
    out_df.to_csv(out, index=False)
    log.info("Wrote propensity scores (mean z=%.3f) → %s", float(np.mean(z)), out)

    if model is not None:
        with open(config.MODEL_PATH / "propensity_model.pkl", "wb") as f:
            pickle.dump({"model": model, "features": _FEATURES}, f)
        log.info("Saved propensity model → %s", config.MODEL_PATH / "propensity_model.pkl")
    return out_df


def main() -> None:
    build_propensity()


if __name__ == "__main__":
    main()
