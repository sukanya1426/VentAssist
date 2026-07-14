"""Acceptance tests for the action-causal reward fix (SYSTEM_SUMMARY §14).

Two layers:
  1. Pure reward-function tests of ``action_causal_bonus`` — always valid, no model.
  2. Policy-behaviour acceptance tests that target the exact §14.1 symptom
     ("SpO2 84 ... SpO2 99 ... PaCO2 65 ... TV 300 all return Hold"). These require
     the *retrained* Track A policy (the causal reward only takes effect after a
     rebuild+retrain); they skip with a clear message if no model is present yet.

Run:  python -m backend.tests.test_reward_causal
"""

from __future__ import annotations

from backend.mdp import reward as R
from backend.pipeline import config


class Skip(Exception):
    """Raised to skip a test when its prerequisites (a trained model) are absent."""


# Seed patient (SYSTEM_SUMMARY §11), editable per test.
_SEED = {"PEEP": 8, "TV": 480, "FiO2": 0.5, "SpO2": 91, "PaO2": 68, "PaCO2": 44,
         "pH": 7.37, "HR": 92, "SBP": 118, "RR": 22, "RASS": -2, "Temp": 37.2}


def make_seed_state(**overrides) -> dict:
    return {**_SEED, **overrides}


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# --------------------------------------------------------------------------- #
# 1. Pure reward-function tests (no trained model required)
# --------------------------------------------------------------------------- #
def test_bonus_rewards_raising_fio2_when_hypoxaemic():
    s = make_seed_state(SpO2=84, FiO2=0.5)
    assert _approx(R.action_causal_bonus(s, (0, 0, 0.05), 70), R.LAM_CAUSAL)
    assert _approx(R.action_causal_bonus(s, (1, 0, 0.0), 70), R.LAM_CAUSAL)   # PEEP up
    assert R.action_causal_bonus(s, (0, 0, -0.05), 70) == 0.0                 # wrong dir
    # already maxed on FiO2 → no bonus for raising it
    assert R.action_causal_bonus(make_seed_state(SpO2=84, FiO2=0.85),
                                 (0, 0, 0.05), 70) == 0.0


def test_bonus_rewards_weaning_fio2_when_hyperoxic():
    s = make_seed_state(SpO2=99, FiO2=0.6)
    assert _approx(R.action_causal_bonus(s, (0, 0, -0.05), 70), R.LAM_CAUSAL)
    assert R.action_causal_bonus(s, (0, 0, 0.05), 70) == 0.0


def test_bonus_rewards_raising_tv_when_hypercapnic():
    s = make_seed_state(PaCO2=65, TV=300)        # 300/70 ≈ 4.3 mL/kg, well < 8
    assert _approx(R.action_causal_bonus(s, (0, 25, 0.0), 70), R.LAM_CAUSAL)
    assert R.action_causal_bonus(s, (0, -25, 0.0), 70) == 0.0


def test_bonus_zero_when_in_band():
    s = make_seed_state(SpO2=94, PaCO2=40)
    for a in [(0, 0, 0.0), (1, 25, 0.05), (-1, -25, -0.05)]:
        assert R.action_causal_bonus(s, a, 70) == 0.0


# --------------------------------------------------------------------------- #
# 2. Policy-behaviour acceptance tests (require the retrained Track A model)
# --------------------------------------------------------------------------- #
def _router_or_skip():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — retrain first (§14.4 fix).")
    from backend.router.policy_router import PolicyRouter
    return PolicyRouter()


def test_responds_to_hypoxaemia():
    """SpO2=84, mid-range settings → policy should NOT hold."""
    router = _router_or_skip()
    rec = router.run_track_a(make_seed_state(SpO2=84, FiO2=0.5, PEEP=8))
    assert not (rec["delta_PEEP"] == 0 and rec["delta_FiO2"] == 0), \
        "Policy holds despite hypoxaemia — causal reward fix did not take effect"


def test_responds_to_hyperoxia():
    """SpO2=99, FiO2=0.6 → policy should wean FiO2 down."""
    router = _router_or_skip()
    rec = router.run_track_a(make_seed_state(SpO2=99, FiO2=0.6, PEEP=8))
    assert rec["delta_FiO2"] < 0, "Policy does not wean FiO2 on hyperoxia"


def test_responds_to_hypercapnia():
    """PaCO2=65, TV=300 (low for typical weight) → policy should raise TV."""
    router = _router_or_skip()
    rec = router.run_track_a(make_seed_state(PaCO2=65, TV=300))
    assert rec["delta_TV"] > 0, "Policy does not raise TV on hypercapnia with low TV"


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
