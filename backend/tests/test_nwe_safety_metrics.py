"""Tests for the NWE rollout clinical-safety metrics (Methodology §12.3, Task E).

The NWE model-based rollout now reports, per policy (HybridIQL and the logged
clinician), the §12.3 safety block: % episodes ending SpO₂ ≥ 95, mean ΔSpO₂ for
hypoxaemic starts (< 95), % time-steps recommending aggressive settings
(PEEP > 15 or FiO₂ > 0.8), and asynchrony_rate (``null`` for Track A — waveform-
only). These tests pin: fractions ∈ [0,1], both policies' dicts carry the
documented keys, and the rollout still returns a finite V̂.

Run:  python -m backend.tests.test_nwe_safety_metrics
"""

from __future__ import annotations

import math

import numpy as np

from backend.ope import nwe
from backend.mdp import dataset as D
from backend.pipeline import config

_KEYS = {"pct_terminal_spo2_ge95", "mean_delta_spo2_start_lt95",
         "n_episodes_start_lt95", "pct_aggressive_steps", "mean_asynchrony_rate"}


class Skip(Exception):
    """Raised to skip when the trained policy / MDP is unavailable."""


def _fractions_ok(m: dict) -> None:
    assert _KEYS.issubset(m.keys()), f"missing keys: {_KEYS - set(m)}"
    for k in ("pct_terminal_spo2_ge95", "pct_aggressive_steps"):
        assert 0.0 <= m[k] <= 1.0, f"{k}={m[k]} not in [0,1]"
    assert m["mean_asynchrony_rate"] is None, "Track A asynchrony must be null (waveform-only)"
    d = m["mean_delta_spo2_start_lt95"]
    assert d is None or math.isfinite(d), "ΔSpO₂ must be finite or None"


# --------------------------------------------------------------------------- #
# 1. Pure-function test (no model / data)
# --------------------------------------------------------------------------- #
def test_safety_metrics_pure_fractions():
    rng = np.random.default_rng(0)
    start = rng.uniform(80, 99, 50)
    term = rng.uniform(85, 100, 50)
    peep = rng.uniform(5, 20, 300)
    fio2 = rng.uniform(0.3, 1.0, 300)
    m = nwe._safety_metrics(start, term, peep, fio2)
    _fractions_ok(m)
    # A trajectory set where every start is >= 95 → no "start<95" episodes.
    m2 = nwe._safety_metrics(np.full(5, 97.0), np.full(5, 98.0),
                             np.full(3, 10.0), np.full(3, 0.5))
    assert m2["n_episodes_start_lt95"] == 0
    assert m2["mean_delta_spo2_start_lt95"] is None
    assert m2["pct_aggressive_steps"] == 0.0


# --------------------------------------------------------------------------- #
# 2. Clinician arm from logged data (needs the MDP only)
# --------------------------------------------------------------------------- #
def test_clinician_safety_from_logged():
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    d = D.load_mdp("a")
    test = np.where(d["split"] == "test")[0]
    rng = np.random.default_rng(0)
    starts = rng.choice(test, size=min(60, len(test)), replace=False)
    _fractions_ok(nwe._clinician_safety(d, starts, T=6))


# --------------------------------------------------------------------------- #
# 3. End-to-end rollout with both arms (needs the trained model)
# --------------------------------------------------------------------------- #
def test_rollout_reports_both_arms_and_finite_v():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    r = nwe.rollout_value("a", T=4, n_starts=40)   # small & fast
    assert math.isfinite(r["V_hat"]), "rollout V_hat must be finite"
    sm = r["safety_metrics"]
    assert set(sm.keys()) == {"hybrid_iql", "clinician"}
    _fractions_ok(sm["hybrid_iql"])
    _fractions_ok(sm["clinician"])


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
