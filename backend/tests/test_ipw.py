"""Tests for IPW confounding correction (SYSTEM_SUMMARY §16 item 8).

1. `compute_ipw_weights` properties on synthetic confounded data: mean ≈ 1,
   clipped to bounds, and rare (informative) action-state pairs up-weighted
   relative to the common confounded ones.
2. `HybridIQL.update` accepts per-transition weights and stays finite; the
   expectile loss actually honours the sample weights.
3. The config default is OFF, so normal training is unchanged.

Run:  python -m backend.tests.test_ipw
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

from backend.pipeline import config
from backend.rl import ipw as IPW
from backend.rl.hybrid_iql import Batch, HybridIQL, expectile_loss


def _confounded_dataset(n: int = 3000, seed: int = 0):
    """Action is largely determined by the state (confounding): when feature 0 is
    negative clinicians almost always take action 0; positive → action 1; a small
    minority take the 'off-policy' action for their region."""
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, 4)).astype(np.float32)
    A = np.where(S[:, 0] < 0, 0, 1)
    flip = rng.random(n) < 0.08                      # rare, informative deviations
    A = np.where(flip, 1 - A, A).astype(np.int64)
    return S, A


def test_ipw_weights_are_normalised_and_bounded():
    S, A = _confounded_dataset()
    w = IPW.compute_ipw_weights(S, A, clip=(0.1, 10.0))
    assert abs(w.mean() - 1.0) < 0.05, f"weights not mean-1 (got {w.mean():.3f})"
    assert w.min() >= 0.1 - 1e-4 and w.max() <= 10.0 + 1e-4, "weights not clipped"
    assert len(w) == len(A)


def test_ipw_upweights_rare_action_state_pairs():
    S, A = _confounded_dataset()
    w = IPW.compute_ipw_weights(S, A)
    # rows where the action disagrees with the confounded default are rare given
    # the state → higher propensity-corrected weight on average.
    default = np.where(S[:, 0] < 0, 0, 1)
    rare = A != default
    assert w[rare].mean() > w[~rare].mean(), "IPW did not up-weight rare pairs"


def test_expectile_loss_honours_sample_weights():
    delta = torch.tensor([1.0, 1.0, 1.0])
    base = expectile_loss(delta, 0.8)
    up = expectile_loss(delta, 0.8, torch.tensor([2.0, 2.0, 2.0]))
    assert torch.isclose(up, 2.0 * base), "sample weights not applied in expectile loss"


def test_hybrid_iql_update_accepts_weights():
    torch.manual_seed(0)
    m = HybridIQL(state_dim=4, action_dim=5, hidden_dim=32)
    B = 64
    def mk(weights):
        return Batch(states=torch.randn(B, 4), actions=torch.randint(0, 5, (B,)),
                     rewards=torch.randn(B), next_states=torch.randn(B, 4),
                     dones=torch.zeros(B), weights=weights)
    out0 = m.update(mk(None))
    out1 = m.update(mk(torch.rand(B) + 0.5))
    for k in ("v_loss", "q_loss", "pi_loss"):
        assert np.isfinite(out0[k]) and np.isfinite(out1[k]), f"{k} non-finite"


def test_config_ipw_default_off():
    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs" / "track_a_config.yaml").read_text())
    assert cfg.get("ipw", {}).get("enabled", False) is False, \
        "IPW must default OFF so deployed training is unchanged"


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
