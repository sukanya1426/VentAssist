"""Tests for the clinician action-density OOD gate (BENCHMARK_PLAN Phase 1, Rule 5).

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_action_density
"""

from __future__ import annotations

import numpy as np
import torch

from backend.mdp import action_space
from backend.pipeline import config
from benchmark import action_density as AD


class Skip(Exception):
    pass


def test_log_probs_are_normalised():
    """A density model must give properly normalised log-probs (Σ_a p = 1)."""
    m = AD.BehaviourDensity(state_dim=12)
    S = np.random.default_rng(0).normal(size=(32, 12)).astype(np.float32)
    logp = m.log_probs(S)
    assert logp.shape == (32, action_space.N_ACTIONS)
    total = np.exp(logp).sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-4), f"probabilities do not sum to 1: {total[:3]}"


def test_factor_marginals_are_normalised_and_nonpositive():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(64, action_space.N_ACTIONS))
    logp = torch.log_softmax(torch.as_tensor(logits), dim=-1).numpy()
    actions = rng.integers(0, action_space.N_ACTIONS, 64)
    out = AD._factor_loglik(logp, actions)
    assert set(out) == {"delta_PEEP", "delta_TV", "delta_FiO2"}
    for k, v in out.items():
        assert v <= 0.0, f"{k} log-likelihood must be <= 0 (got {v})"


def test_factor_bins_cover_the_action_space():
    """The marginalisation indices must partition all 125 actions correctly."""
    for name, bins in AD._FACTORS.items():
        assert len(bins) == action_space.N_ACTIONS
        assert set(np.unique(bins)) == set(range(5)), f"{name} should have 5 bins"


def test_density_learns_a_skewed_behaviour_policy():
    """Fit on data where one action dominates → that action gets the highest log-prob."""
    rng = np.random.default_rng(0)
    S = rng.normal(size=(2000, 12)).astype(np.float32)
    hold = action_space.encode_action(0, 0, 0.0)
    A = np.full(2000, hold, dtype=np.int64)
    A[:100] = action_space.encode_action(2, 0, 0.0)          # 5% take a different action
    m = AD.fit(S, A, epochs=5)
    logp = m.log_probs(S[:50])
    assert int(logp[0].argmax()) == hold, "density did not learn the dominant behaviour action"


def test_policy_is_on_support_vs_clinician():
    """The headline Rule-5 gate: Δ = loglik(policy) − loglik(clinician) must be > 0."""
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present.")
    r = AD.evaluate("a", epochs=8)
    d = r["delta_policy_minus_clinician"]
    print(f"  loglik policy={r['loglik_policy']} clinician={r['loglik_clinician']} Δ={d:+.4f}")
    assert d > 0, ("policy acts OFF the clinician's support (Δ<=0) — an FQE gain here "
                   "would be extrapolation, not improvement")


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
