"""Behaviour-cloning baseline (Section 6.4) — Random Forest over clinician actions."""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("rl_bc")


def train_bc(states: np.ndarray, actions: np.ndarray, track: str = "a"):
    n_est = 200 if len(states) > 500 else 100
    bc = RandomForestClassifier(n_estimators=n_est, max_depth=10,
                                random_state=42, n_jobs=-1)
    bc.fit(states, actions)
    out = config.MODEL_PATH / f"bc_track_{track}.pkl"
    with open(out, "wb") as f:
        pickle.dump(bc, f)
    log.info("Saved BC (train acc=%.3f) → %s", bc.score(states, actions), out)
    return bc
