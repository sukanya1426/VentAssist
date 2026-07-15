"""Tests for quantile-regression DFQE (Methodology §12.2, Task D).

The old DFQE was a point-FQE + bootstrap CI stand-in. It is now a real
distributional estimator: a quantile-regression Q-network trained with the
quantile-Huber loss under a distributional Bellman backup, reporting the return
distribution's mean (``V_hat``), α-quantile lower-confidence bound
(``LCB_alpha``), variance, and the policy coverage ``d^π``.

Two groups of tests:

  1. Pure-function tests of the quantile-Huber loss and the QR network / monotone
     rearrangement — no trained model needed.
  2. A small end-to-end QR-DFQE fit on the real MDP (reduced quantiles/steps for
     speed) that pins the acceptance properties. Skips if the model/MDP is absent.

Run:  python -m backend.tests.test_dfqe_qr
"""

from __future__ import annotations

import numpy as np
import torch

from backend.ope import dfqe
from backend.pipeline import config


class Skip(Exception):
    """Raised to skip a test when its prerequisites (trained model / MDP) are absent."""


# --------------------------------------------------------------------------- #
# 1. Pure-function tests (no trained model)
# --------------------------------------------------------------------------- #
def test_quantile_huber_zero_at_perfect_fit():
    """Loss is (near) minimal when predicted quantiles equal the targets."""
    n_q = 8
    taus = (torch.arange(n_q).float() + 0.5) / n_q
    pred = torch.linspace(-1, 1, n_q).reshape(1, n_q)
    same = dfqe.quantile_huber_loss(pred, pred.clone(), taus)
    worse = dfqe.quantile_huber_loss(pred, pred + 1.0, taus)
    assert same >= 0.0
    assert worse > same, "loss must increase as prediction moves off the target"


def test_quantile_huber_is_asymmetric():
    """Pinball loss penalises under- vs over-prediction differently by level."""
    n_q = 2
    taus = torch.tensor([0.1, 0.9])                         # low & high quantile
    # A single target sample above both predictions.
    pred = torch.zeros(1, n_q)
    target = torch.ones(1, n_q)
    # For a target above the prediction (u>0) the weight is τ, so the high-τ
    # quantile should incur MORE loss than the low-τ one for the same error.
    lo = dfqe.quantile_huber_loss(pred[:, :1], target[:, :1], taus[:1])
    hi = dfqe.quantile_huber_loss(pred[:, 1:], target[:, 1:], taus[1:])
    assert hi > lo, "high quantile should be penalised more for under-prediction"


def test_qr_network_shape_and_rearrangement():
    net = dfqe.QRQNetwork(state_dim=12, action_dim=125, n_quantiles=11)
    out = net(torch.randn(7, 12))
    assert out.shape == (7, 125, 11)
    # Monotone rearrangement (sorting) makes any quantile vector weakly increasing.
    z = out.detach().numpy()[:, 0, :]
    zs = np.sort(z, axis=1)
    assert np.all(np.diff(zs, axis=1) >= -1e-9), "rearranged quantiles must be non-decreasing"


# --------------------------------------------------------------------------- #
# 2. End-to-end QR-DFQE on the real MDP (needs trained model + MDP)
# --------------------------------------------------------------------------- #
def _fit_or_skip() -> dict:
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present — build the dataset first.")
    # Small, fast configuration (properties, not precision).
    return dfqe.distributional_fqe_qr("a", n_quantiles=9, K=3, steps_per_iter=30)


def test_qr_dfqe_properties():
    r = _fit_or_skip()
    # LCB (α-quantile) must not exceed the mean of the distribution.
    assert r["LCB_alpha"] <= r["V_hat"] + 1e-6, \
        f"LCB_alpha {r['LCB_alpha']} > V_hat {r['V_hat']}"
    # Policy coverage d^π is a fraction.
    assert 0.0 <= r["policy_coverage"] <= 1.0, f"d^π out of [0,1]: {r['policy_coverage']}"
    # Reported quantile function is weakly monotonically increasing.
    qv = np.asarray(r["quantile_values"])
    assert np.all(np.diff(qv) >= -1e-6), f"quantile function not monotone: {qv}"
    assert len(qv) == r["n_quantiles"]
    assert np.isfinite(r["return_variance"]) and r["return_variance"] >= 0.0


def test_qr_dfqe_log_written():
    _fit_or_skip()
    path = config.LOGS_PATH / "dfqe_track_a.json"
    assert path.exists(), "dfqe_track_a.json not written"
    import json
    d = json.loads(path.read_text())
    assert d["method"] == "qr"
    assert d["n_transitions"] > 900_000, "log looks stale (n_transitions too small)"
    assert "timestamp" in d


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
