"""Tests for per-group CV bandwidths in the NWE kernel (Methodology §10.3, Task B).

The NWE transition kernel used a single median-heuristic state bandwidth with
fixed ha=1.0, hz=0.2. It now uses per clinical-group state bandwidths
(respiratory / hemodynamic / mental) plus action/propensity bandwidths, all
chosen by grid search minimising held-out 1-step next-state MSE.

Acceptance: the CV bandwidths achieve ≤ the median-heuristic default MSE on a
subsample; bandwidths are positive & finite; the NWE rollout still runs.

Run:  python -m backend.tests.test_nwe_bandwidths
"""

from __future__ import annotations

import math

import numpy as np

from backend.ope import nwe
from backend.pipeline import config


class Skip(Exception):
    """Raised to skip when the MDP / trained model is unavailable."""


# --------------------------------------------------------------------------- #
# Pure structural tests
# --------------------------------------------------------------------------- #
def test_groups_partition_all_twelve_features():
    flat = [f for feats in nwe.FEATURE_GROUPS.values() for f in feats]
    assert sorted(flat) == sorted(config.TABULAR_FEATURES), "groups must partition the 12 dims"
    assert len(flat) == 12 and len(set(flat)) == 12, "groups must be disjoint & complete"


def test_scalar_hs_is_backward_compatible():
    """A scalar hs with no groups reproduces the original single-kernel model."""
    rng = np.random.default_rng(0)
    S = rng.normal(size=(50, 12)); NS = rng.normal(size=(50, 12))
    A = rng.integers(0, 125, 50).astype(float); Z = rng.random(50)
    m = nwe.NWEModel(S, A, NS, Z, hs=2.0, ha=1.0, hz=0.2)   # groups=None
    out = m.next_state_batch(S[:5], A[:5], Z[:5])
    assert out.shape == (5, 12) and np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# CV selection (needs the MDP)
# --------------------------------------------------------------------------- #
def _select_or_skip():
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    return nwe.select_bandwidths("a", ref_cap=4000, val_cap=600, seed=0)


def test_cv_bandwidths_beat_median_default():
    bw = _select_or_skip()
    assert bw["val_mse_selected"] <= bw["val_mse_baseline_median"] + 1e-9, (
        f"CV MSE {bw['val_mse_selected']} > median default "
        f"{bw['val_mse_baseline_median']}")


def test_bandwidths_positive_and_finite():
    bw = _select_or_skip()
    for name, h in bw["hs"].items():
        assert math.isfinite(h) and h > 0, f"state bandwidth {name}={h} not positive/finite"
    assert bw["ha"] > 0 and bw["hz"] > 0, "action/propensity bandwidths must be positive"


def test_rollout_still_runs_with_cv_bandwidths():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    r = nwe.rollout_value("a", T=3, n_starts=30)      # fit(cv=True) inside
    assert math.isfinite(r["V_hat"]), "rollout V_hat must be finite with CV bandwidths"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Skip as e:
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
