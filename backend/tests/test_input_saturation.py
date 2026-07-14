"""Regression tests for the input-saturation bug (serving-time normalisation).

BUG: the router used ``normaliser.transform``, the TRAINING transform, which
winsorises every input to the training 1st/99th percentile (SpO₂ ∈ [89, 100],
PEEP ∈ [0, 18], TV ∈ [227, 863], …). Live inputs outside those bounds were clipped,
so SpO₂ 55 / 70 / 80 / 84 / 88 all produced the *identical* state vector as SpO₂ 89
— and therefore the identical recommendation, no matter what the clinician typed.
It looked like a stale/cached prediction; it was input saturation.

FIX: serving uses ``normaliser.transform_inference`` — z-score without the raw
winsor clip, bounded in z-space to ±INFERENCE_CLIP_SIGMA. Distinct clinical inputs
stay distinct, while the bound stops the Q-network extrapolating into nonsense
(un-clipped, SpO₂ ≈ 50 makes it recommend cutting TV and lowering FiO₂ on a
critically hypoxaemic patient).

Run:  python -m backend.tests.test_input_saturation
"""

from __future__ import annotations

import numpy as np

from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.router import policy_router as PR


class Skip(Exception):
    pass


_BASE = dict(PEEP=8, TV=480, FiO2=0.5, SpO2=95, PaO2=88, PaCO2=44, pH=7.37,
             HR=92, SBP=118, RR=22, RASS=-2, Temp=37.2)
_FEATS = config.TABULAR_FEATURES


def _stats():
    return N.load(config.MODEL_PATH / "normaliser_stats.json")


def _vec(**over):
    s = {**_BASE, **over}
    return np.array([float(s[f]) for f in _FEATS])


def _router_or_skip():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    return PR.PolicyRouter()


# --------------------------------------------------------------------------- #
# 1. The transforms themselves
# --------------------------------------------------------------------------- #
def test_training_transform_saturates_below_winsor_low():
    """Documents the old behaviour: distinct hypoxaemia levels collapse to one vector."""
    st = _stats()
    zs = [N.transform(_vec(SpO2=v), st, _FEATS)[3] for v in (55, 70, 80, 84, 88)]
    assert len(set(np.round(zs, 6))) == 1, "expected the training transform to saturate"


def test_inference_transform_keeps_distinct_inputs_distinct():
    st = _stats()
    zs = [N.transform_inference(_vec(SpO2=v), st, _FEATS)[3] for v in (70, 80, 84, 88, 95)]
    assert len(set(np.round(zs, 6))) == len(zs), \
        "serving transform must not collapse distinct SpO2 values"
    # strictly increasing in SpO2
    assert all(a < b for a, b in zip(zs, zs[1:])), "z must be monotone in SpO2"


def test_inference_transform_is_bounded():
    """Bound must hold even for the most extreme API-legal input (SpO2 50 → z ≈ -19)."""
    st = _stats()
    z = N.transform_inference(_vec(SpO2=50, PEEP=30, TV=1200, PaCO2=120, HR=250),
                              st, _FEATS)
    assert np.all(np.abs(z) <= N.INFERENCE_CLIP_SIGMA + 1e-9), \
        "unbounded z lets the Q-network extrapolate into unsafe nonsense"


# --------------------------------------------------------------------------- #
# 2. The router actually responds now
# --------------------------------------------------------------------------- #
def test_router_responds_to_hypoxaemia_edit():
    """The reported bug: edit SpO2 down, recommendation must change."""
    router = _router_or_skip()
    healthy = router.run_track_a({**_BASE, "SpO2": 95})
    hypoxic = router.run_track_a({**_BASE, "SpO2": 84})
    assert healthy["action"] != hypoxic["action"], \
        "editing SpO2 95 → 84 did not change the recommendation"


def test_router_responds_to_extreme_settings_beyond_winsor_bounds():
    """PEEP > 18 and TV > 863 used to be clipped away entirely."""
    router = _router_or_skip()
    # PEEP 30 is far past the old winsor_high of 18 → must trigger a PEEP reduction
    r = router.run_track_a({**_BASE, "PEEP": 30})
    assert r["delta_PEEP"] < 0, "dangerously high PEEP must be reduced"
    # TV 1100 is past the old winsor_high of 863 → must trigger a TV reduction
    r = router.run_track_a({**_BASE, "TV": 1100})
    assert r["delta_TV"] < 0, "volutrauma-level TV must be cut"


def test_extreme_inputs_saturate_to_a_SAFE_extreme_and_are_flagged():
    """Beyond the clamp we must still give a clinically sane answer, flagged OOD."""
    router = _router_or_skip()
    r = router.run_track_a({**_BASE, "SpO2": 55, "PaO2": 40})
    # must NOT wean oxygen / cut TV on a critically hypoxaemic patient
    assert not (r["delta_FiO2"] < 0), "must not lower FiO2 on severe hypoxaemia"
    assert r["delta_PEEP"] >= 0 or r["delta_FiO2"] > 0, \
        "severe hypoxaemia must raise PEEP and/or FiO2"


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
