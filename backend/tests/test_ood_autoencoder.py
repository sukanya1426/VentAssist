"""Tests for the learned OOD/support detector (SYSTEM_SUMMARY §16 item 7).

The autoencoder's reconstruction error is a state-conditional density proxy that
upgrades the local-NN support check in test_support_constraint.py. These tests
assert the detector actually discriminates:

  1. In-distribution (train/test) states are mostly in support; a state pushed far
     outside the physiological manifold is flagged OOD (low support_ratio).
  2. Through the router, an implausible state yields lower confidence than a
     normal one, and the recommendation carries an `in_support` signal.

Requires ``models/ood_ae_track_a.pt`` + the policy; skips cleanly if absent.

Run:  python -m backend.tests.test_ood_autoencoder
"""

from __future__ import annotations

import numpy as np

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.router import policy_router as PR


class Skip(Exception):
    pass


def _detector_or_skip():
    if not (config.MODEL_PATH / "ood_ae_track_a.pt").exists():
        raise Skip("ood_ae_track_a.pt not present — run `python -m backend.rl.ood_autoencoder`.")
    from backend.rl.ood_autoencoder import OODDetector
    return OODDetector.load("a")


def test_in_distribution_states_mostly_in_support():
    det = _detector_or_skip()
    d = D.load_mdp("a")
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    S = N.transform(d["states"], stats, d["feature_order"]).astype(np.float32)
    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(S))
    rng = np.random.default_rng(0)
    sample = rng.choice(test, size=min(500, len(test)), replace=False)
    in_support = np.mean([det.in_support(S[i]) for i in sample])
    print(f"  test-split in-support rate = {in_support:.1%}")
    # threshold is the train p99, so held-out in-support should be high (~>=0.9)
    assert in_support >= 0.85, f"held-out in-support rate {in_support:.1%} too low"


def test_far_out_state_is_flagged_ood():
    det = _detector_or_skip()
    normal = np.zeros(12, dtype=np.float32)                 # z-space origin = typical patient
    extreme = np.full(12, 12.0, dtype=np.float32)           # 12 SD out on every axis
    e_normal = det.evaluate(normal)
    e_extreme = det.evaluate(extreme)
    print(f"  score normal={e_normal['score']:.4f} (support {e_normal['support_ratio']:.2f}) | "
          f"extreme={e_extreme['score']:.4f} (support {e_extreme['support_ratio']:.2f})")
    assert e_normal["in_support"] and not e_extreme["in_support"]
    assert e_extreme["support_ratio"] < e_normal["support_ratio"]


def test_router_lowers_confidence_for_ood_state():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present.")
    _detector_or_skip()                                     # ensure detector loads
    router = PR.PolicyRouter()
    if router.ood is None:
        raise Skip("router did not load an OOD detector.")
    normal = dict(PEEP=8, TV=460, FiO2=0.4, SpO2=95, PaO2=88, PaCO2=40, pH=7.40,
                  HR=84, SBP=120, RR=16, RASS=-1, Temp=37.0)
    # physiologically incoherent extremes (still within API bounds)
    weird = dict(PEEP=28, TV=200, FiO2=1.0, SpO2=55, PaO2=40, PaCO2=115, pH=6.85,
                 HR=210, SBP=60, RR=55, RASS=4, Temp=41.5)
    rn = router.run_track_a(normal)
    rw = router.run_track_a(weird)
    print(f"  normal: in_support={rn['in_support']} support={rn['support_ratio']} conf={rn['confidence']}")
    print(f"  weird : in_support={rw['in_support']} support={rw['support_ratio']} conf={rw['confidence']}")
    assert rn["in_support"] is True
    assert rw["support_ratio"] <= rn["support_ratio"]
    assert rw["confidence"] <= rn["confidence"]


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
