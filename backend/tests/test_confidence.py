"""Tests for the Q-value-based recommendation confidence (SYSTEM_SUMMARY §9).

The confidence used to be a hard-coded per-track constant (Track A always 0.60),
so the dashboard bar never moved regardless of the patient. It is now derived
from the policy's own Q-values by
``backend.router.policy_router._decision_confidence``:

    confidence = clip(info_weight * sigmoid(margin / temp), floor, ceil)

where ``margin`` is the best-minus-second-best Q gap. These tests pin the
properties that make it honest:

  1. Pure-function tests of ``_decision_confidence`` — always valid, no model.
  2. Policy tests over the 6 dashboard presets — require the trained Track A
     model; they skip with a clear message if it is absent.

Run:  python -m backend.tests.test_confidence
"""

from __future__ import annotations

import numpy as np

from backend.pipeline import config
from backend.router import policy_router as PR


class Skip(Exception):
    """Raised to skip a test when its prerequisites (a trained model) are absent."""


# --------------------------------------------------------------------------- #
# 1. Pure-function tests (no trained model required)
# --------------------------------------------------------------------------- #
def test_flat_q_gives_floor_confidence():
    """A dead tie between the top two actions → sharpness 0.5 (coin flip)."""
    q = np.zeros(125)
    out = PR._decision_confidence(q, PR.CONF_WEIGHT_CLINICAL)
    assert abs(out["decision_margin"]) < 1e-9
    # sharpness == 0.5, so confidence == 0.5 * info_weight
    assert abs(out["confidence"] - 0.5 * PR.CONF_WEIGHT_CLINICAL) < 1e-3


def test_larger_margin_gives_higher_confidence():
    q_small = np.zeros(125); q_small[0] = 0.2
    q_large = np.zeros(125); q_large[0] = 2.0
    c_small = PR._decision_confidence(q_small, PR.CONF_WEIGHT_CLINICAL)["confidence"]
    c_large = PR._decision_confidence(q_large, PR.CONF_WEIGHT_CLINICAL)["confidence"]
    assert c_large > c_small, "confidence must rise with the decision margin"


def test_confidence_is_bounded():
    for peak in (-5.0, 0.0, 0.3, 50.0):
        q = np.zeros(125); q[0] = peak
        c = PR._decision_confidence(q, PR.CONF_WEIGHT_WAVEFORM_FULL)["confidence"]
        assert PR.CONF_FLOOR <= c <= PR.CONF_CEIL, f"confidence {c} out of bounds"


def test_waveform_weight_raises_ceiling():
    """Same Q-values, more information (waveform) → at least as confident."""
    q = np.zeros(125); q[0] = 1.0
    c_clin = PR._decision_confidence(q, PR.CONF_WEIGHT_CLINICAL)["confidence"]
    c_wave = PR._decision_confidence(q, PR.CONF_WEIGHT_WAVEFORM_FULL)["confidence"]
    assert c_wave >= c_clin
    assert c_wave > c_clin  # strictly, since neither is clipped here


# --------------------------------------------------------------------------- #
# 2. Policy tests over the dashboard presets (need the trained model)
# --------------------------------------------------------------------------- #
_PRESETS = {
    "stable":      dict(PEEP=8, TV=460, FiO2=0.4, SpO2=95, PaO2=88, PaCO2=40, pH=7.40,
                        HR=84, SBP=120, RR=16, RASS=-1, Temp=37.0),
    "hypoxaemia":  dict(PEEP=8, TV=480, FiO2=0.5, SpO2=84, PaO2=55, PaCO2=44, pH=7.34,
                        HR=104, SBP=112, RR=26, RASS=-2, Temp=37.6),
    "hyperoxia":   dict(PEEP=8, TV=480, FiO2=0.7, SpO2=100, PaO2=90, PaCO2=44, pH=7.37,
                        HR=92, SBP=118, RR=22, RASS=-2, Temp=37.2),
    "hypercapnia": dict(PEEP=8, TV=320, FiO2=0.5, SpO2=93, PaO2=72, PaCO2=65, pH=7.28,
                        HR=98, SBP=118, RR=28, RASS=-1, Temp=37.3),
    "high_peep":   dict(PEEP=18, TV=470, FiO2=0.5, SpO2=94, PaO2=80, PaCO2=43, pH=7.38,
                        HR=90, SBP=105, RR=20, RASS=-3, Temp=37.2),
    "volutrauma":  dict(PEEP=8, TV=760, FiO2=0.5, SpO2=95, PaO2=90, PaCO2=38, pH=7.44,
                        HR=86, SBP=122, RR=14, RASS=-2, Temp=37.0),
}

_FULL_WAVEFORM = {"HRV_SDNN": 31.2, "Arrhythmia_rate": 0.03, "Perfusion_Index": 2.1,
                  "RRV": 0.19, "Breathing_Regularity": 0.81, "Asynchrony_Score": 0.1}


def _router_or_skip():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    return PR.PolicyRouter()


def test_confidence_varies_across_presets():
    """The bug was a constant 0.60 everywhere; confidence must now differ."""
    router = _router_or_skip()
    confs = [round(router.run_track_a(s)["confidence"], 3) for s in _PRESETS.values()]
    assert len(set(confs)) >= 3, f"confidence barely varies across presets: {confs}"
    assert not all(abs(c - 0.60) < 1e-6 for c in confs), "still the constant 0.60 bug"


def test_confidence_in_bounds_for_all_presets():
    router = _router_or_skip()
    for name, s in _PRESETS.items():
        c = router.run_track_a(s)["confidence"]
        assert PR.CONF_FLOOR <= c <= PR.CONF_CEIL, f"{name}: confidence {c} out of bounds"


def test_waveform_track_is_more_confident_than_clinical():
    """Same tabular state: full-waveform Track B ≥ Track A (more information)."""
    router = _router_or_skip()
    for name, s in _PRESETS.items():
        ca = router.run_track_a(s)["confidence"]
        cb = router.run_track_b(s, dict(_FULL_WAVEFORM))["confidence"]
        assert cb >= ca - 1e-9, f"{name}: waveform confidence {cb} < clinical {ca}"


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
