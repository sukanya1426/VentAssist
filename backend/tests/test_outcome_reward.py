"""Tests for the terminal outcome reward (mortality + ventilator-free days).

Pure function tests of ``reward.outcome_reward`` plus a check that the dataset
builder attaches the term only at the episode terminal and that it is sweepable
via ``recombined_reward``.

Run:  pytest backend/tests/test_outcome_reward.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.mdp import reward as R
from backend.mdp import dataset as D


# --------------------------------------------------------------------------- #
# reward.outcome_reward
# --------------------------------------------------------------------------- #
def test_death_gives_full_mortality_penalty():
    assert R.outcome_reward(died=True, vent_days=0.0, horizon_days=28) == -1.0
    # vent_days is irrelevant once dead
    assert R.outcome_reward(died=True, vent_days=10.0, horizon_days=28) == -1.0


def test_survivor_weaned_immediately_gets_full_vfd_reward():
    assert R.outcome_reward(died=False, vent_days=0.0, horizon_days=28) == 1.0


def test_survivor_ventilated_whole_horizon_gets_zero():
    assert R.outcome_reward(died=False, vent_days=28.0, horizon_days=28) == 0.0
    # beyond the horizon is clamped, not negative
    assert R.outcome_reward(died=False, vent_days=40.0, horizon_days=28) == 0.0


def test_survivor_half_horizon():
    assert abs(R.outcome_reward(died=False, vent_days=14.0, horizon_days=28) - 0.5) < 1e-9


def test_monotonic_decreasing_in_vent_days():
    vals = [R.outcome_reward(False, d, horizon_days=28) for d in range(0, 29)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_weights_scale_terms():
    assert R.outcome_reward(True, 0, w_mortality=3.0) == -3.0
    assert R.outcome_reward(False, 0, w_vfd=2.0) == 2.0


# --------------------------------------------------------------------------- #
# dataset wiring (synthetic, no parquet / no MIMIC needed)
# --------------------------------------------------------------------------- #
def _toy_states():
    # one stay, 3 hourly epochs → 2 transitions; last is terminal.
    base = {f: 0.0 for f in D.TABULAR}
    base.update({"PEEP": 8, "TV": 480, "FiO2": 0.5, "SpO2": 95,
                 "PaO2": 90, "PaCO2": 40, "pH": 7.4, "HR": 80,
                 "SBP": 120, "RR": 16, "RASS": 0, "Temp": 37.0})
    rows = []
    for h in range(3):
        rows.append({"stay_id": 1, "hour": h, **base})
    return pd.DataFrame(rows)


def test_outcome_unit_only_on_terminal(monkeypatch):
    states = _toy_states()
    cohort = pd.DataFrame([{"stay_id": 1, "weight_kg": 70.0}])
    split = {"train": {1}, "val": set(), "test": set()}
    outcomes = {1: (True, 0.0)}  # died → unit term = -1.0

    tx = D._build_transitions(states, D.TABULAR, track="a",
                              cohort=cohort, prop=None, split=split,
                              outcomes=outcomes)
    tx = tx.sort_values("hour").reset_index(drop=True)
    assert len(tx) == 2
    # non-terminal transition carries no outcome term
    assert tx.loc[0, "outcome_unit"] == 0.0
    assert not tx.loc[0, "done"]
    # terminal transition carries the death penalty unit
    assert tx.loc[1, "done"]
    assert abs(tx.loc[1, "outcome_unit"] - (-1.0)) < 1e-9


def test_recombined_reward_respects_w_outcome(monkeypatch):
    df = pd.DataFrame({
        "reward": [0.0, 0.0],
        "reward_base": [0.5, 0.5],
        "causal_unit": [0.0, 0.0],
        "outcome_unit": [0.0, -1.0],
    })
    monkeypatch.setattr(D, "_reward_lam_causal", lambda track: 0.0)
    monkeypatch.setattr(D, "_reward_w_outcome", lambda track: 2.0)
    r = D.recombined_reward(df, "a")
    assert abs(r[0] - 0.5) < 1e-6           # 0.5 + 0 + 2*0
    assert abs(r[1] - (0.5 - 2.0)) < 1e-6   # 0.5 + 0 + 2*(-1)
