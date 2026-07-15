"""Tests for the reward critique (BENCHMARK_PLAN §0 trap 2).

Pins the structural claim — IntelliLung's RangeReward carries ZERO action signal —
so it cannot be quietly weakened later. Layer 2 (training) is too expensive for the
test suite and is run separately via `python -m benchmark.reward_critique --layer 2`.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_reward_critique
"""

from __future__ import annotations

import numpy as np

from backend.mdp import action_space
from backend.mdp import reward as VA
from benchmark import rewards_intellilung as IL

_S = dict(PEEP=8, TV=480, FiO2=0.5, SpO2=84, PaO2=55, PaCO2=65, pH=7.28,
          HR=104, SBP=112, RR=26, RASS=-2, Temp=37.6)
_NS = dict(PEEP=8, TV=480, FiO2=0.5, SpO2=88, PaO2=62, PaCO2=58, pH=7.31,
           HR=100, SBP=114, RR=24, RASS=-2, Temp=37.5)


def test_intellilung_reward_is_identical_for_every_action():
    """THE claim: hold (s, s') fixed, vary the action → the reward never moves."""
    r = [IL.range_reward_scalar(_NS) for _ in range(action_space.N_ACTIONS)]
    assert max(r) - min(r) == 0.0, \
        "RangeReward must be action-invariant by construction (it is a function of s' alone)"


def test_ventassist_reward_discriminates_between_actions():
    r = [VA.tier1_reward(_S, _NS, action_space.decode_action(a), 74.0, lam_causal=2.0)
         for a in range(action_space.N_ACTIONS)]
    assert max(r) - min(r) > 0.5, \
        "VentAssist reward must carry a real action signal (action cost + causal bonus)"


def test_ventassist_reward_prefers_the_clinically_indicated_action():
    """Hypoxaemic + hypercapnic state: raising PEEP/FiO2 or TV must beat weaning them."""
    def r(dp, dt, df):
        return VA.tier1_reward(_S, _NS, (dp, dt, df), 74.0, lam_causal=2.0)
    assert r(2, 0, 0.0) > r(-2, 0, 0.0), "raising PEEP on hypoxaemia should beat lowering it"
    assert r(0, 25, 0.0) > r(0, -25, 0.0), "raising TV on hypercapnia should beat lowering it"


def test_range_reward_is_bounded_in_minus_one_zero():
    """Their reward is normalised then time-penalised → [-1, 0]."""
    feats = ["PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2", "pH",
             "HR", "SBP", "RR", "RASS", "Temp"]
    rng = np.random.default_rng(0)
    ns = np.stack([
        rng.uniform([0, 200, .21, 70, 40, 20, 7.0, 50, 80, 8, -5, 35],
                    [25, 900, 1.0, 100, 300, 90, 7.6, 150, 180, 40, 2, 40])
        for _ in range(200)])
    r = IL.range_reward(ns, feats)
    assert r.min() >= -1.0 - 1e-9 and r.max() <= 0.0 + 1e-9, f"out of [-1,0]: {r.min()},{r.max()}"


def test_all_in_range_gives_zero_and_none_in_range_gives_minus_one():
    perfect = dict(PaCO2=40, PaO2=70, pH=7.40, SpO2=92, HR=80)      # all inside
    awful = dict(PaCO2=90, PaO2=300, pH=7.05, SpO2=70, HR=150)      # all outside
    assert abs(IL.range_reward_scalar(perfect) - 0.0) < 1e-9
    assert abs(IL.range_reward_scalar(awful) - (-1.0)) < 1e-9


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
