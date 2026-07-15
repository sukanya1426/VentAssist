"""Inverse-propensity weighting for confounding-by-indication (SUMMARY §16 item 8).

The deepest offline-RL-for-treatment problem: the logged actions are chosen by
clinicians whose choices correlate with patient severity (confounding by
indication), so a policy trained to imitate/evaluate them inherits spurious
state→action correlations. **Stabilised inverse-propensity weighting (IPW)**
partially corrects this by reweighting each transition by

    w(s, a) = p(a) / π_β(a | s)

where ``π_β(a|s)`` is a behaviour action-model (the propensity of the taken
action given the state) and ``p(a)`` is that action's marginal frequency. Common
action-state pairs — the ones clinicians take *because of* the confounded state —
get w < 1; informative rare pairs get w > 1. Weights are self-normalised to mean
1 and clipped to bound variance, then applied to the HybridIQL losses.

This is a partial, standard correction — not a cure for unmeasured confounding.
It is **opt-in** (config ``ipw.enabled``); the deployed checkpoints train without
it, so behaviour is unchanged unless enabled.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.pipeline.logging_utils import get_logger

log = get_logger("rl_ipw")


def fit_behaviour_propensities(states: np.ndarray, actions: np.ndarray,
                               n_estimators: int = 200, max_depth: int = 12,
                               seed: int = 42) -> np.ndarray:
    """Return π_β(a_i | s_i) for each transition via an out-of-the-box RF classifier.

    Uses out-of-bag-style honesty only loosely (a single RF); good enough for a
    propensity signal. Returns the predicted probability of the *taken* action."""
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=seed, n_jobs=-1)
    clf.fit(states, actions)
    proba = clf.predict_proba(states)                       # (N, n_classes)
    class_to_col = {int(c): j for j, c in enumerate(clf.classes_)}
    cols = np.array([class_to_col[int(a)] for a in actions])
    e = proba[np.arange(len(actions)), cols]                # π_β(a_i | s_i)
    return np.clip(e, 1e-6, 1.0)


def compute_ipw_weights(states: np.ndarray, actions: np.ndarray,
                        clip: tuple[float, float] = (0.1, 10.0),
                        seed: int = 42) -> np.ndarray:
    """Stabilised, self-normalised, clipped IPW weights (mean ≈ 1)."""
    e = fit_behaviour_propensities(states, actions, seed=seed)
    # marginal p(a)
    uniq, counts = np.unique(actions, return_counts=True)
    marg = {int(a): c / len(actions) for a, c in zip(uniq, counts)}
    p_a = np.array([marg[int(a)] for a in actions])
    w = p_a / e                                             # stabilised IPW
    w = w / w.mean()                                        # normalise to mean 1
    w = np.clip(w, clip[0], clip[1])
    w = w / w.mean()                                        # renormalise after clip
    log.info("IPW weights: mean=%.3f std=%.3f min=%.3f max=%.3f (n=%d)",
             w.mean(), w.std(), w.min(), w.max(), len(w))
    return w.astype(np.float32)
