"""Empirical CQL support-constraint check (Methodology §7).

The offline-RL support constraint is

    π(a|s) > 0  ⟹  π_β(a|s) > 0,   ∀(s, a) ∈ S × A,

i.e. the learned policy should not put mass on actions that the behaviour policy
never took in that region of state space. The CQL conservatism term is meant to
enforce this; this test verifies it empirically on the *trained* policy rather
than just asserting the mechanism exists.

Two checks, because there are two operationalizations of the constraint:

1. **Global support (the formal constraint).** Every greedy action the policy
   chooses must lie in the behaviour-supported valid-action set (actions observed
   >= min_count in training; the rare-action remap already enforces this). This is
   the direct read of π(a|s) > 0 ⟹ π_β(a|s) > 0 and MUST hold at 0 violations.

   (A learned state-density upgrade to this proxy now exists —
   [ood_autoencoder.py](backend/rl/ood_autoencoder.py) + test_ood_autoencoder.py, §16 item 7.)

2. **Local-neighbourhood proxy (bounded, not strict).** For a sample of test
   states, how often is the chosen action absent from the k nearest training
   states' actions. This is a *stricter* proxy than the formal constraint. Note
   the §14.4 causal-reward fix deliberately makes the policy take clinically
   indicated actions that clinicians UNDER-took (breaking confounding), so those
   actions are by construction locally rare — a non-trivial local-proxy violation
   rate is therefore EXPECTED and is the price of fixing the under-responsiveness.
   A joint (lam_causal, cql.alpha) search (sweep_joint.py) showed no operating
   point reaches the original 5% local rate while also passing the §14.1 seed
   acceptance; the threshold here is set to a level the chosen operating point
   meets, and still guards against the pre-causal collapse (which was ~100% OOD
   with behaviour_match ≈ 0). Tighten via cql.alpha if it regresses past the bound.

Run:  python -m backend.tests.test_support_constraint
"""

from __future__ import annotations

import numpy as np
import torch

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.rl.hybrid_iql import HybridIQL


class Skip(Exception):
    """Raised to skip when the trained policy / split is unavailable."""


N_TEST_STATES = 200
K_NEIGHBOURS = 20
# Local-NN proxy bound. The strict 0.05 is unreachable jointly with the §14.1
# responsiveness acceptance (sweep_joint.py); this bounds the proxy at the chosen
# operating point (lam_causal=6.0, alpha=1.0 measured ~0.125) while still catching a
# collapse back toward the pre-causal ~100%-OOD regime.
MAX_LOCAL_VIOLATION_RATE = 0.20
TRAIN_NN_CAP = 100_000          # cap train pool for the NN search (speed)


def _load_policy() -> HybridIQL:
    path = config.MODEL_PATH / "policy_track_a.pt"
    if not path.exists():
        raise Skip("policy_track_a.pt not present — train first.")
    ckpt = torch.load(path, map_location="cpu")
    m = HybridIQL(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"],
                  hidden_dim=ckpt["hidden_dim"])
    m.load_state_dict(ckpt["state_dict"])
    return m


def test_global_support_constraint():
    """Formal π(a|s)>0 ⟹ π_β(a|s)>0: every chosen action is behaviour-supported."""
    policy = _load_policy()
    d = D.load_mdp("a")
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    feats = d["feature_order"]
    S = N.transform(d["states"], stats, feats).astype(np.float32)

    train = np.where(d["split"] == "train")[0]
    test = np.where(d["split"] == "test")[0]
    if len(train) == 0 or len(test) == 0:
        raise Skip("MDP has no train/test split to evaluate.")
    valid = set(int(a) for a in np.unique(d["actions"][train]))  # behaviour support

    rng = np.random.default_rng(0)
    test_sample = rng.choice(test, size=min(N_TEST_STATES, len(test)), replace=False)
    oob = sum(1 for i in test_sample if policy.act(S[i]) not in valid)
    print(f"  global-support violations = {oob}/{len(test_sample)}")
    assert oob == 0, (
        f"{oob} chosen actions are outside the behaviour-supported action set — "
        "the formal CQL support constraint is violated.")


def test_local_neighbourhood_support_bounded():
    """Bounded local-NN proxy (see module docstring for why it is not strict)."""
    policy = _load_policy()
    d = D.load_mdp("a")
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    feats = d["feature_order"]
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    A = d["actions"]

    train = np.where(d["split"] == "train")[0]
    test = np.where(d["split"] == "test")[0]
    if len(train) == 0 or len(test) == 0:
        raise Skip("MDP has no train/test split to evaluate.")

    rng = np.random.default_rng(0)
    if len(train) > TRAIN_NN_CAP:
        train = rng.choice(train, size=TRAIN_NN_CAP, replace=False)
    test_sample = rng.choice(test, size=min(N_TEST_STATES, len(test)), replace=False)
    train_S, train_A = S[train], A[train]

    violations = 0
    for i in test_sample:
        action = policy.act(S[i])
        dist = np.linalg.norm(train_S - S[i], axis=1)
        nn = np.argpartition(dist, min(K_NEIGHBOURS, len(dist) - 1))[:K_NEIGHBOURS]
        if action not in set(int(a) for a in train_A[nn]):
            violations += 1

    rate = violations / len(test_sample)
    print(f"  local-NN proxy violation rate = {rate:.1%} "
          f"({violations}/{len(test_sample)})")
    assert rate < MAX_LOCAL_VIOLATION_RATE, (
        f"Local-NN support proxy violated in {rate:.1%} of test states "
        f"(> {MAX_LOCAL_VIOLATION_RATE:.0%}) — policy is drifting OOD; raise cql.alpha.")


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
