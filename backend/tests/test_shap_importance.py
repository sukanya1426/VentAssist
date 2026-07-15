"""Tests for global Q-function feature importance (Methodology §12.4, Task F).

The in-house permutation-sampling Shapley estimator over f(s)=max_a Q(s,a) must:
cover all 12 tabular features, be deterministic under a fixed seed, produce a
ranking sorted by importance, and (if the optional ``shap`` package is present)
agree with ``KernelExplainer`` by Spearman > 0.5.

Run:  python -m backend.tests.test_shap_importance
"""

from __future__ import annotations

import numpy as np

from backend.pipeline import config
from backend.scripts import shap_importance as SI


class Skip(Exception):
    """Raised to skip when the trained policy / MDP is unavailable."""


def _compute_or_skip(backend: str = "inhouse", seed: int = 0):
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    return SI.compute("a", backend=backend, n_states=60, n_perms=16, seed=seed)


# --------------------------------------------------------------------------- #
# Pure-function sanity: Shapley on a known linear function
# --------------------------------------------------------------------------- #
def test_shapley_recovers_linear_weights():
    """For f(x)=Σ w_i x_i with a zero baseline, φ_i(x) = w_i x_i exactly."""
    w = np.array([3.0, -1.0, 0.0, 2.0])
    f = lambda z: (np.asarray(z).reshape(-1, 4) @ w)
    X = np.array([[1.0, 1.0, 5.0, 2.0], [-2.0, 3.0, 1.0, 0.5]], dtype=np.float32)
    baseline = np.zeros(4, dtype=np.float32)
    phi = SI.shapley_permutation(f, X, baseline, n_perms=24, seed=1)
    expected = X * w
    assert np.allclose(phi, expected, atol=1e-5), f"Shapley != w·x\n{phi}\n{expected}"


# --------------------------------------------------------------------------- #
# Model-based tests (need the trained Track A policy)
# --------------------------------------------------------------------------- #
def test_covers_all_twelve_features():
    r = _compute_or_skip()
    assert set(r["importance"].keys()) == set(config.TABULAR_FEATURES)
    assert len(r["ranking"]) == 12
    vals = np.array([v for v in r["importance"].values()])
    assert np.all(np.isfinite(vals)) and np.all(vals >= 0.0), "importances must be finite ≥ 0"


def test_deterministic_under_fixed_seed():
    a = _compute_or_skip(seed=7)["importance"]
    b = _compute_or_skip(seed=7)["importance"]
    assert a == b, "importances must be identical for a fixed seed"


def test_ranking_sorted_descending():
    r = _compute_or_skip()
    imps = [row["importance"] for row in r["ranking"]]
    assert imps == sorted(imps, reverse=True), "ranking must be sorted by importance desc"


def test_shap_crosscheck_if_available():
    try:
        import shap  # noqa: F401
    except ImportError:
        raise Skip("shap not installed — in-house estimator is authoritative.")
    r = _compute_or_skip(backend="shap")
    assert r.get("spearman_inhouse_vs_shap", -1) > 0.5, \
        f"in-house vs shap Spearman too low: {r.get('spearman_inhouse_vs_shap')}"


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
