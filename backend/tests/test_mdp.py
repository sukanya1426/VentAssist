"""Unit tests for the MDP layer (action space, reward, normaliser).

Run:  python -m pytest backend/tests/test_mdp.py   (or run this file directly)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.mdp import action_space as A
from backend.mdp import normaliser as N
from backend.mdp import reward as R


def test_action_map_complete():
    assert A.N_ACTIONS == 125
    assert len(A.ACTION_MAP) == 125
    for idx in range(125):
        dp, dt, df = A.decode_action(idx)
        assert dp in A.DELTA_PEEP_BINS and dt in A.DELTA_TV_BINS
        assert df in A.DELTA_FIO2_BINS
    # corner + hold actions
    assert A.decode_action(0) == (-2, -50, -0.10)
    assert A.decode_action(124) == (2, 50, 0.10)
    assert A.decode_action(A.encode_action(0, 0, 0.0)) == (0, 0, 0.0)


def test_encode_round_and_clip():
    assert A.decode_action(A.encode_action(2, 50, 0.10)) == (2, 50, 0.10)
    assert A.decode_action(A.encode_action(-2, -50, -0.10)) == (-2, -50, -0.10)
    # rounding to nearest bin
    assert A.encode_action(1.4, 37, 0.04) == A.encode_action(1, 25, 0.05)
    assert A.encode_action(99, 999, 9) == A.encode_action(2, 50, 0.10)      # clip high
    assert A.encode_action(-99, -999, -9) == A.encode_action(-2, -50, -0.10)  # clip low


def test_remap_rare():
    hold = A.encode_action(0, 0, 0.0)
    rare = A.encode_action(2, 50, 0.10)
    s = pd.Series([hold] * 20 + [rare])
    remapped, valid = A.remap_rare(s, min_count=10)
    assert rare not in valid
    assert remapped.iloc[-1] in valid


_STABLE = {"SpO2": 94, "PaCO2": 40, "TV": 450, "FiO2": 0.4, "PEEP": 8}


def test_reward_hold_when_stable():
    # in-band, no change → exactly zero reward (the optimal default)
    assert abs(R.tier1_reward(_STABLE, _STABLE, (0, 0, 0.0), 70)) < 1e-9


def test_reward_action_cost():
    # needless change while stable is penalised by the action cost only
    r = R.tier1_reward(_STABLE, _STABLE, (1, 25, 0.05), 70)
    assert abs(r - (-0.3)) < 1e-9          # 0.1*1 + 0.1*(25/25) + 0.1*(0.05/0.05)


def test_reward_oxygenation_gain():
    s_t = {**_STABLE, "SpO2": 85}
    s_n = {**_STABLE, "SpO2": 92, "FiO2": 0.45}
    r = R.tier1_reward(s_t, s_n, (0, 0, 0.05), 70)
    # Δoxy = f(92)-f(85) = 0-(-7) = 7; cost 0.1; +0.4 action-causal bonus for
    # raising FiO2 while hypoxaemic (SpO2 85 < 92) and not yet maxed (FiO2 < 0.8).
    assert abs(r - (7.0 - 0.1 + 0.4)) < 1e-9


def test_reward_safety_penalties():
    s_n = {**_STABLE, "TV": 650, "FiO2": 0.9, "PEEP": 16}   # volutrauma+O2tox+highPEEP
    r = R.tier1_reward(_STABLE, s_n, (2, 50, 0.10), 70)
    cost = 0.1 * 2 + 0.1 * (50 / 25) + 0.1 * (0.10 / 0.05)  # 0.6
    safety = 0.5 + 0.3 + 0.3                                # 1.1
    assert abs(r - (-cost - safety)) < 1e-9


def test_tier2_extra_penalties():
    s_t = {**_STABLE, "Asynchrony_Score": 1.0, "Arrhythmia_rate": 0.3}
    base = R.tier1_reward(s_t, s_t, (0, 0, 0.0), 70)
    assert abs(R.tier2_reward(s_t, s_t, (0, 0, 0.0), 70) - (base - 0.2 - 0.15)) < 1e-9
    # missing waveform → equals tier1 (0 here)
    assert abs(R.tier2_reward(_STABLE, _STABLE, (0, 0, 0.0), 70)) < 1e-9


def test_normaliser_roundtrip():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(10, 5, 1000), "y": rng.random(1000)})
    st = N.fit(df, ["x", "y"])
    arr = df[["x", "y"]].to_numpy()
    z = N.transform(arr, st, ["x", "y"])
    back = N.inverse_transform(z, st, ["x", "y"])
    mid = (arr[:, 0] > st["x"]["winsor_low"]) & (arr[:, 0] < st["x"]["winsor_high"])
    assert np.allclose(back[mid, 0], arr[mid, 0], atol=1e-6)
    assert abs(z[:, 0].mean()) < 0.05 and abs(z[:, 0].std() - 1) < 0.05


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL TESTS PASSED")
