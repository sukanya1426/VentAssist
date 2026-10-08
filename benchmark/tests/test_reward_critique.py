"""Tests for the reward critique (BENCHMARK_PLAN §0 trap 2).

Pins the structural claim — IntelliLung's RangeReward carries ZERO action signal —
so it cannot be quietly weakened later. Layer 2 (training) is too expensive for the
test suite and is run separately via `python -m benchmark.reward_critique --layer 2`.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_reward_critique
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.mdp import action_space
from backend.mdp import reward as VA
from benchmark import rewards_intellilung as IL


class Skip(Exception):
    pass

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


# --------------------------------------------------------------------------- #
# Claims about THEIR source, checked against their source.
#
# These skip when the reference repository is absent (it is untracked, 17 MB), so
# the suite still runs on a clean checkout. When it is present, they turn the
# headline critique from an assertion in a document into something a reviewer can
# re-run. Pinning them here means that if a future reader of ours disputes the
# claim, the check is one command away.
# --------------------------------------------------------------------------- #
_THEIRS = Path(__file__).resolve().parents[2] / \
    "intellilung-advancing-mechanical-ventilation" / "algo_src" / "reward"


def _reward_sources() -> dict[str, str]:
    if not _THEIRS.is_dir():
        raise Skip(f"reference repo not on disk at {_THEIRS}")
    return {f.name: f.read_text() for f in sorted(_THEIRS.glob("*.py"))}


def test_their_reward_interface_has_no_action_parameter():
    """The design-level form of the claim: the action is not in the signature."""
    src = _reward_sources()
    base = src.get("base.py")
    assert base, "algo_src/reward/base.py not found"
    sig = [ln for ln in base.splitlines() if "def __call__" in ln]
    assert sig, "no __call__ in their RewardFunction base"
    for ln in sig:
        assert "action" not in ln, (
            f"their reward base signature now mentions an action: {ln.strip()!r} — "
            "the interface-level claim in BENCHMARK_PLAN must be revisited")


def test_no_reward_implementation_of_theirs_reads_an_action():
    """`grep -n action algo_src/reward/*.py` must stay empty."""
    offenders = {name: [ln.strip() for ln in text.splitlines() if "action" in ln]
                 for name, text in _reward_sources().items()}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        f"an IntelliLung reward now references an action: {offenders}. The critique "
        "claims none of them does; update the claim before the defence.")


def test_their_range_reward_is_computed_from_next_states_only():
    src = _reward_sources()["range.py"]
    assert "shift(-1)" in src, "their RangeReward no longer shifts to next_states"
    assert "next_states[key]" in src, (
        "their RangeReward no longer indexes next_states for the in-range test")


def test_our_port_matches_their_ranges_and_priorities():
    """Every range/priority we implement must equal the value in their config."""
    cfg = _THEIRS.parent / "configs" / "state_vector_ranges_for_reward.json"
    if not cfg.exists():
        raise Skip(f"their ranges file absent at {cfg}")
    theirs = json.loads(cfg.read_text())
    # our feature name -> their config key
    mapping = {"PaCO2": "blood_paco2", "PaO2": "blood_pao2", "pH": "blood_ph",
               "SpO2": "vital_spo2", "HR": "vital_hr"}
    for ours, their_key in mapping.items():
        low, high, prio = IL.RANGES[ours]
        t = theirs[their_key]
        assert [low, high] == [float(x) for x in t["range"]], (
            f"{ours}: our range {[low, high]} != theirs {t['range']}")
        assert prio == t["priority"], (
            f"{ours}: our priority {prio} != theirs {t['priority']}")


def test_the_features_we_omit_are_declared():
    """An omission must be documented, not silent."""
    cfg = _THEIRS.parent / "configs" / "state_vector_ranges_for_reward.json"
    if not cfg.exists():
        raise Skip(f"their ranges file absent at {cfg}")
    theirs = set(json.loads(cfg.read_text()))
    mapped = {"blood_paco2", "blood_pao2", "blood_ph", "vital_spo2", "vital_hr"}
    unaccounted = theirs - mapped - set(IL.OMITTED)
    assert not unaccounted, (
        f"their config has reward features we neither implement nor declare as "
        f"omitted: {unaccounted}")


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
