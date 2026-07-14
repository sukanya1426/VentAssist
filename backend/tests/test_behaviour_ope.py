"""Tests for the behaviour (clinician) OPE baseline.

Pure tests of the discounted-return and bootstrap helpers (no model / no parquet).

Run:  pytest backend/tests/test_behaviour_ope.py -v
"""

from __future__ import annotations

import numpy as np

from backend.ope import behaviour as B


def test_episode_returns_discounting():
    # two episodes; rewards given out of hour order to check sorting.
    stay = np.array([1, 1, 2, 1, 2])
    hour = np.array([1, 0, 1, 2, 0])
    rew = np.array([2.0, 1.0, 4.0, 3.0, 3.0])
    gamma = 0.5
    ids, G = B._episode_returns(stay, hour, rew, gamma)

    order = np.argsort(ids)
    ids, G = ids[order], G[order]
    assert list(ids) == [1, 2]
    # episode 1 ordered by hour: [1, 2, 3] → 1 + .5*2 + .25*3 = 2.75
    assert abs(G[0] - 2.75) < 1e-9
    # episode 2 ordered by hour: [3, 4] → 3 + .5*4 = 5.0
    assert abs(G[1] - 5.0) < 1e-9


def test_episode_returns_gamma_one_is_plain_sum():
    stay = np.array([7, 7, 7])
    hour = np.array([0, 1, 2])
    rew = np.array([1.0, 2.0, 3.0])
    ids, G = B._episode_returns(stay, hour, rew, gamma=1.0)
    assert list(ids) == [7]
    assert abs(G[0] - 6.0) < 1e-9


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=500)
    lo, hi = B._bootstrap_ci(x, n_boot=500, seed=1)
    assert lo < x.mean() < hi


def test_bootstrap_ci_empty():
    lo, hi = B._bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)
