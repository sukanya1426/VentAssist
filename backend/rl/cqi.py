"""Conservative Q-Improvement decision-tree baseline (Section 6.3).

Fits a shallow, human-readable decision tree predicting the clinician action,
weighting samples by the softmax of the policy's Q(s, a_taken).
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("rl_cqi")


def train_cqi(states: np.ndarray, actions: np.ndarray, q_taken: np.ndarray,
              track: str = "a", max_depth: int = 5):
    q = np.asarray(q_taken, dtype=float)
    w = np.exp(q - q.max())
    w = w / (w.sum() + 1e-9) * len(w)        # normalise to mean weight ~1
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(states, actions, sample_weight=w)
    out = config.MODEL_PATH / f"cqi_track_{track}.pkl"
    with open(out, "wb") as f:
        pickle.dump(tree, f)
    log.info("Saved CQI tree (depth=%d) → %s", tree.get_depth(), out)
    return tree
