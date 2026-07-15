"""Tests for the multi-output LMC/ICM GP imputation (Methodology §3, Task A).

The default lab imputation fits an independent Matérn GP per blood-gas signal; the
LMC path (backend/pipeline/gp_lmc.py) fits an intrinsic coregionalization model
that imputes PaO₂/PaCO₂/pH jointly via a learned coregionalization matrix W.

Acceptance:
  * On correlated multi-signal data, LMC's 5-fold held-out RMSE ≤ the independent
    Matérn RMSE (cross-signal information helps);
  * imputed series are finite and within the winsorised physiological ranges;
  * the default (matern) path is unchanged — config default is matern and the
    matern branch never touches the LMC code.

Uses synthetic correlated episodes so the test is fast and self-contained (no
39 GB labevents scan).

Run:  python -m backend.tests.test_gp_imputation_lmc
"""

from __future__ import annotations

import numpy as np

from backend.pipeline import config, gp_lmc
from backend.pipeline import gp_imputation as GPI

_BASE = {"PaO2": 90.0, "PaCO2": 40.0, "pH": 7.4}
_AMP = {"PaO2": 25.0, "PaCO2": 8.0, "pH": 0.06}


def _correlated_episodes(n_ep: int = 60, seed: int = 1) -> list[dict]:
    """Episodes whose 3 signals share a latent temporal driver (→ correlated),
    each observed at a sparse, partially disjoint set of times."""
    rng = np.random.default_rng(seed)
    feats = config.LAB_FEATURES
    eps = []
    for _ in range(n_ep):
        n_h = int(rng.integers(18, 40))
        hrs = np.arange(n_h, dtype=float)
        u = np.sin(2 * np.pi * hrs / rng.uniform(8, 20)) + 0.4 * np.sin(2 * np.pi * hrs / 5)
        u = (u - u.mean()) / (u.std() + 1e-6)
        obs = {}
        for f in feats:
            sig = _BASE[f] + _AMP[f] * u + rng.normal(0, 0.1 * _AMP[f], n_h)
            k = int(rng.integers(4, 8))
            idx = np.sort(rng.choice(n_h, size=k, replace=False))
            obs[f] = (hrs[idx], sig[idx])
        eps.append(obs)
    return eps


def test_lmc_beats_independent_matern_on_correlated_data():
    rep = gp_lmc.crossval_compare(_correlated_episodes())
    assert rep["mean_rmse_lmc"] <= rep["mean_rmse_matern"] + 1e-9, (
        f"LMC mean RMSE {rep['mean_rmse_lmc']:.4f} > Matérn "
        f"{rep['mean_rmse_matern']:.4f}")
    # per-signal: LMC should not be materially worse on any signal here.
    for f in config.LAB_FEATURES:
        assert rep["rmse_lmc"][f] <= rep["rmse_matern"][f] * 1.05 + 1e-9, \
            f"{f}: LMC RMSE {rep['rmse_lmc'][f]} >> Matérn {rep['rmse_matern'][f]}"


def test_W_is_valid_correlation_matrix():
    eps = _correlated_episodes()
    mean, std = gp_lmc._task_stats(eps)
    W = gp_lmc._estimate_W(eps, mean, std)
    D = len(config.LAB_FEATURES)
    assert W.shape == (D, D)
    assert np.allclose(W, W.T), "W must be symmetric"
    assert np.allclose(np.diag(W), 1.0, atol=1e-6), "W must have unit diagonal"
    assert np.all(np.linalg.eigvalsh(W) > -1e-6), "W must be PSD"


def test_imputed_series_finite_and_in_range():
    eps = _correlated_episodes()
    imp, _ = gp_lmc.fit_global(eps)
    hours = np.arange(30, dtype=float)
    pred = imp.impute_episode(eps[0], hours)
    assert set(pred).issubset(set(config.LAB_FEATURES))
    for f, series in pred.items():
        lo, hi = config.LAB_CLIP_RANGES[f]
        assert np.all(np.isfinite(series)), f"{f} imputation has non-finite values"
        assert np.all(series >= lo - 1e-6) and np.all(series <= hi + 1e-6), \
            f"{f} imputation outside winsorised range [{lo},{hi}]"


def test_cross_signal_imputation_when_target_unobserved():
    """A signal with NO observations is still imputed from correlated signals."""
    eps = _correlated_episodes()
    imp, _ = gp_lmc.fit_global(eps)
    obs = dict(eps[0])
    obs.pop("pH")                                   # drop pH entirely
    pred = imp.impute_episode(obs, np.arange(25, dtype=float))
    assert "pH" in pred, "pH should be imputed via cross-signal coupling"
    lo, hi = config.LAB_CLIP_RANGES["pH"]
    assert np.all((pred["pH"] >= lo - 1e-6) & (pred["pH"] <= hi + 1e-6))


def test_default_method_is_matern():
    assert GPI._imputation_method("a") == "matern", \
        "default imputation.method must be matern (deployed path unchanged)"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
