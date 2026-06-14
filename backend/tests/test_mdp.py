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


def test_reward_hypoxaemic_band():
    s_t = {"SpO2": 90, "TV": 500, "FiO2": 0.5, "PEEP": 8}
    assert abs(R.tier1_reward(s_t, {"SpO2": 93}, 70) - (-2 / 7)) < 1e-6
    assert R.tier1_reward(s_t, {"SpO2": 98}, 70) == 0.0   # hyperoxia → 0 shaping


def test_reward_penalties():
    s_bad = {"SpO2": 90, "TV": 700, "FiO2": 0.9, "PEEP": 16}
    r = R.tier1_reward(s_bad, {"SpO2": 93}, 70)
    assert abs(r - (-2 / 7 - 0.3 - 0.5)) < 1e-6


def test_tier2_extra_penalties():
    s_t = {"SpO2": 90, "TV": 500, "FiO2": 0.5, "PEEP": 8,
           "Asynchrony_Score": 1.0, "Arrhythmia_rate": 0.3}
    base = R.tier1_reward(s_t, {"SpO2": 93}, 70)
    assert abs(R.tier2_reward(s_t, {"SpO2": 93}, 70) - (base - 0.2 - 0.15)) < 1e-6
    # missing waveform → equals tier1
    clean = {"SpO2": 90, "TV": 500, "FiO2": 0.5, "PEEP": 8}
    assert abs(R.tier2_reward(clean, {"SpO2": 93}, 70) - base) < 1e-6


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
