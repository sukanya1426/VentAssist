"""Feature imputer (Section 8.2) — predict missing waveform features from tabular.

One Random-Forest regressor per waveform feature, trained on Track B rows where
all 6 waveform features are present. Used at inference for Track B partial
coverage. Arrhythmia_rate defaults to 0.0 when missing (not imputed).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("feature_imputer")

WAVEFORM = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
            "RRV", "Breathing_Regularity", "Asynchrony_Score"]
TABULAR = config.TABULAR_FEATURES
_PATH = config.MODEL_PATH / "feature_imputer_track_b.pkl"


class FeatureImputer:
    def __init__(self, models: dict | None = None, mae: dict | None = None):
        self.models = models or {}
        self.mae = mae or {}

    # --- training ---
    @classmethod
    def fit_from_files(cls) -> "FeatureImputer":
        states = pd.read_parquet(config.PROCESSED_PATH / "tabular_states_track_b.parquet")
        wave = pd.read_parquet(config.PROCESSED_PATH / "waveform_features_track_b.parquet")
        df = states.merge(wave, on=["stay_id", "hour"], how="inner").dropna(subset=WAVEFORM)
        models, mae = {}, {}
        X = df[TABULAR].to_numpy(float)
        for feat in WAVEFORM:
            y = df[feat].to_numpy(float)
            rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                                       random_state=42, n_jobs=-1)
            rf.fit(X, y)
            models[feat] = rf
            mae[feat] = float(np.mean(np.abs(rf.predict(X) - y)))
        log.info("Trained feature imputer on %d complete rows; MAE=%s",
                 len(df), {k: round(v, 3) for k, v in mae.items()})
        imp = cls(models, mae)
        imp.save()
        return imp

    def save(self):
        with open(_PATH, "wb") as f:
            pickle.dump({"models": self.models, "mae": self.mae}, f)
        (config.LOGS_PATH / "imputer_mae.json").write_text(
            __import__("json").dumps(self.mae, indent=2))

    @classmethod
    def load(cls) -> "FeatureImputer":
        with open(_PATH, "rb") as f:
            d = pickle.load(f)
        return cls(d["models"], d["mae"])

    # --- inference ---
    def impute(self, tabular_state: dict, available: dict) -> dict:
        """Fill missing waveform features from the tabular state."""
        x = np.array([[float(tabular_state[t]) for t in TABULAR]])
        out = {}
        for feat in WAVEFORM:
            v = available.get(feat)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out[feat] = float(v)
            elif feat == "Arrhythmia_rate":
                out[feat] = 0.0                       # default, not imputed
            elif feat in self.models:
                out[feat] = float(self.models[feat].predict(x)[0])
            else:
                out[feat] = 0.0
        return out


def main():
    FeatureImputer.fit_from_files()


if __name__ == "__main__":
    main()
