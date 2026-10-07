"""Tests for the AI-vs-clinician behavioural analyses (Phase 1 item 6).

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_behaviour_compare
"""

from __future__ import annotations

import numpy as np

from backend.mdp import action_space
from backend.pipeline import config
from benchmark import behaviour_compare as BC


class Skip(Exception):
    pass


HOLD = action_space.encode_action(0, 0, 0.0)
UP = action_space.encode_action(2, 50, 0.10)
DOWN = action_space.encode_action(-2, -50, -0.10)


def test_identical_sequences_agree_perfectly():
    a = np.array([HOLD, UP, DOWN, HOLD])
    r = BC._agreement(a, a)
    assert r["exact_125_way"] == 1.0
    assert r["directional"]["all_three"] == 1.0
    for k in BC.KNOBS:
        assert r["per_knob"][k] == 1.0
    assert r["hold_agreement"] == 1.0


def test_opposite_actions_disagree_on_every_knob():
    r = BC._agreement(np.array([UP, UP]), np.array([DOWN, DOWN]))
    assert r["exact_125_way"] == 0.0
    assert r["directional"]["all_three"] == 0.0


def test_exact_and_directional_agreement_separate_intent_from_step_size():
    """Same direction, different magnitude: directional agrees, exact does not.

    This is the distinction the whole metric exists for — "the policy wants
    something different" vs "the policy wants the same thing, less of it".
    """
    small = action_space.encode_action(1, 25, 0.05)
    big = action_space.encode_action(2, 50, 0.10)
    r = BC._agreement(np.array([small]), np.array([big]))
    assert r["exact_125_way"] == 0.0, "different magnitudes are not an exact match"
    assert r["directional"]["all_three"] == 1.0, "same sign on all three knobs"


def test_deviation_sign_reports_the_direction_of_the_bias():
    """A policy asking for less TV than the clinician must read as negative."""
    lower_tv = action_space.encode_action(0, -50, 0.0)
    r = BC._deviation(np.array([lower_tv] * 4), np.array([HOLD] * 4))
    d = r["delta_TV"]
    assert d["mean_signed"] == -50.0
    assert d["mean_absolute"] == 50.0
    assert d["share_policy_lower"] == 1.0
    assert d["share_policy_higher"] == 0.0


def test_deviation_of_identical_actions_is_exactly_zero():
    a = np.array([HOLD, UP, DOWN])
    for k, v in BC._deviation(a, a).items():
        assert v["mean_signed"] == 0.0 and v["mean_absolute"] == 0.0, k
        assert v["share_equal"] == 1.0, k


def test_churn_ignores_episode_boundaries():
    """Two different stays must never be compared across their boundary.

    Without the stay guard the last hour of one stay is diffed against the first
    hour of the next, and churn becomes a count of episode boundaries.
    """
    actions = np.array([HOLD, HOLD, UP, UP])
    stay = np.array([1, 1, 2, 2])
    hour = np.array([0, 1, 0, 1])
    r = BC._churn(actions, stay, hour)
    assert r["n_consecutive_pairs"] == 2, \
        f"expected 2 within-stay pairs, got {r['n_consecutive_pairs']}"
    assert r["action_changed"] == 0.0, "no change happened inside either stay"


def test_churn_ignores_charting_gaps():
    """Hours 0 and 2 are not consecutive — a gap is not a decision change."""
    r = BC._churn(np.array([HOLD, UP]), np.array([1, 1]), np.array([0, 2]))
    assert r["n_consecutive_pairs"] == 0


def test_churn_counts_a_real_within_stay_change():
    r = BC._churn(np.array([HOLD, UP, UP]), np.array([1, 1, 1]),
                  np.array([0, 1, 2]))
    assert r["n_consecutive_pairs"] == 2
    assert r["action_changed"] == 0.5, "one of two pairs changed"


def test_reversal_is_detected_and_is_stricter_than_a_change():
    """up-then-down on the same knob is a reversal; up-then-hold is only a change."""
    up = action_space.encode_action(2, 0, 0.0)
    down = action_space.encode_action(-2, 0, 0.0)
    rev = BC._churn(np.array([up, down]), np.array([1, 1]), np.array([0, 1]))
    assert rev["per_knob_reversed"]["delta_PEEP"] == 1.0
    plain = BC._churn(np.array([up, HOLD]), np.array([1, 1]), np.array([0, 1]))
    assert plain["per_knob_changed"]["delta_PEEP"] == 1.0
    assert plain["per_knob_reversed"]["delta_PEEP"] == 0.0, \
        "returning to hold is a change, not a reversal"


def test_churn_is_order_independent():
    """Rows arriving unsorted must give the same answer — it lexsorts internally."""
    a = np.array([HOLD, UP, DOWN])
    s = np.array([1, 1, 1])
    h = np.array([0, 1, 2])
    order = np.array([2, 0, 1])
    assert (BC._churn(a, s, h)["action_changed"]
            == BC._churn(a[order], s[order], h[order])["action_changed"])


def test_real_evaluation_runs_and_is_internally_consistent():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present")
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present")
    r = BC.evaluate("a", write=False)
    a = r["agreement"]
    assert 0.0 <= a["exact_125_way"] <= 1.0
    # Exact agreement can never exceed agreement on signs alone.
    assert a["exact_125_way"] <= a["directional"]["all_three"] + 1e-9, \
        "exact agreement exceeded directional agreement — impossible"
    for k in BC.KNOBS:
        assert a["per_knob"][k] >= a["exact_125_way"] - 1e-9, \
            f"{k}: per-knob agreement below the 125-way exact rate"
    cp = r["churn"]["policy"]
    assert cp["n_consecutive_pairs"] > 0
    assert 0.0 <= cp["action_changed"] <= 1.0
    for k in BC.KNOBS:
        assert cp["per_knob_reversed"][k] <= cp["per_knob_changed"][k] + 1e-9, \
            f"{k}: more reversals than changes — a reversal IS a change"


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
