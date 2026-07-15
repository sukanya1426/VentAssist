"""Tests for the clinical safety-violation metrics (BENCHMARK_PLAN Phase 1, Rule 9).

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_safety_metrics
"""

from __future__ import annotations

import numpy as np

from backend.mdp import action_space
from backend.pipeline import config
from benchmark import safety_metrics as SM


class Skip(Exception):
    pass


def test_ardsnet_min_peep_is_a_monotone_step():
    fio2 = np.array([0.21, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
    mp = SM.ardsnet_min_peep(fio2)
    assert mp[0] == 5.0 and mp[1] == 5.0          # low FiO2 → 5
    assert mp[3] == 8.0                            # 0.50 → 8
    assert mp[4] == 10.0                           # 0.60 → 10
    assert mp[5] == 14.0                           # 0.80 → 14
    assert mp[6] == 18.0                           # 1.00 → 18
    assert all(a <= b for a, b in zip(mp, mp[1:])), "min-PEEP must be non-decreasing in FiO2"


def test_resulting_settings_applies_deltas_and_clips_fio2():
    hold = action_space.encode_action(0, 0, 0.0)
    up = action_space.encode_action(2, 50, 0.10)
    peep = np.array([8.0, 8.0]); tv = np.array([460.0, 460.0]); fio2 = np.array([0.4, 0.95])
    p, t, f = SM._resulting_settings(peep, tv, fio2, np.array([hold, up]))
    assert (p[0], t[0], round(f[0], 2)) == (8.0, 460.0, 0.4)          # hold = no change
    assert (p[1], t[1]) == (10.0, 510.0)                               # +2 PEEP, +50 TV
    assert f[1] <= action_space.FIO2_MAX + 1e-9, "FiO2 must be clipped at 1.0"


def test_violation_rates_are_valid_probabilities():
    n = 100
    rng = np.random.default_rng(0)
    peep = rng.uniform(0, 25, n); tv = rng.uniform(200, 900, n)
    fio2 = rng.uniform(0.21, 1.0, n); spo2 = rng.uniform(80, 100, n)
    w = np.full(n, 70.0)
    v = SM._violations(peep, tv, fio2, spo2, w)
    for k, val in v.items():
        if val is None:
            continue
        assert 0.0 <= val <= 1.0, f"{k} = {val} is not a rate"
    # any_violation must be >= each individual rule
    individual = [val for k, val in v.items()
                  if val is not None and k != "any_violation"]
    assert v["any_violation"] >= max(individual) - 1e-9


def test_driving_pressure_is_reported_as_unavailable_not_faked():
    v = SM._violations(np.array([8.0]), np.array([460.0]), np.array([0.4]),
                       np.array([95.0]), np.array([70.0]))
    assert v["driving_pressure_gt_15"] is None, \
        "driving pressure needs airway-pressure waveform — must be null, never imputed"


def test_clear_violations_are_detected():
    # TV 900 mL on a 70 kg patient = 12.9 mL/kg → volutrauma; PEEP 20 → >15; FiO2 1.0 → >0.8
    v = SM._violations(np.array([20.0]), np.array([900.0]), np.array([1.0]),
                       np.array([99.0]), np.array([70.0]))
    assert v["volutrauma_tv_gt_8ml_per_kg"] == 1.0
    assert v["peep_gt_15"] == 1.0
    assert v["fio2_gt_0.8"] == 1.0
    assert v["needless_hyperoxia_fio2_ge_0.95_and_spo2_ge_96"] == 1.0
    assert v["any_violation"] == 1.0


def test_policy_beats_clinician_on_the_real_split():
    """The headline claim: fewer violations than the clinician on identical states."""
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present.")
    r = SM.evaluate("a")
    d = r["delta_policy_minus_clinician"]["any_violation"]
    print(f"  any_violation: policy={r['policy']['any_violation']} "
          f"clinician={r['clinician']['any_violation']} Δ={d:+.4f}")
    assert d <= 0, "policy should not violate MORE often than the clinician"


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
