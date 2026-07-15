"""Tests for mode-aware action masking (SYSTEM_SUMMARY §16 item 6).

Pressure-control masks every ΔTV ≠ 0 action; volume-control / unknown mask
nothing. Layer 1 is pure (no model); layer 2 checks the router actually refuses
to recommend a TV change in pressure-control on a state that otherwise wants one.

Run:  python -m backend.tests.test_action_masking
"""

from __future__ import annotations

from backend.mdp import action_masking as AM
from backend.mdp import action_space
from backend.pipeline import config
from backend.router import policy_router as PR


class Skip(Exception):
    pass


# --------------------------------------------------------------------------- #
# 1. Pure-function tests
# --------------------------------------------------------------------------- #
def test_classify_categories_and_raw_modes():
    assert AM.classify_ventilator_mode("pressure_control") == "pressure_control"
    assert AM.classify_ventilator_mode("VC") == "volume_control"
    assert AM.classify_ventilator_mode("PCV+") == "pressure_control"      # raw mode
    assert AM.classify_ventilator_mode("VOL/AC") == "volume_control"      # raw mode
    assert AM.classify_ventilator_mode(None) is None
    assert AM.classify_ventilator_mode("ASV") is None                     # ambiguous → no mask


def test_pressure_control_masks_all_tv_changes():
    mask = AM.mode_action_mask("pressure_control")
    for idx, (_dp, dtv, _df, _d) in action_space.ACTION_MAP.items():
        assert mask[idx] == (dtv == 0), f"action {idx} (ΔTV={dtv}) mask wrong"
    # exactly the ΔTV==0 slice survives: 5 PEEP × 1 × 5 FiO2 = 25 allowed
    assert int(mask.sum()) == 25


def test_volume_control_and_unknown_mask_nothing():
    assert AM.mode_action_mask("volume_control").all()
    assert AM.mode_action_mask(None).all()


# --------------------------------------------------------------------------- #
# 2. Router integration
# --------------------------------------------------------------------------- #
# Hypercapnia with low TV: the policy wants to RAISE TV — pressure-control must
# block that and fall back to a ΔTV==0 action.
_HYPERCAPNIA = dict(PEEP=8, TV=320, FiO2=0.5, SpO2=93, PaO2=72, PaCO2=65, pH=7.28,
                    HR=98, SBP=118, RR=28, RASS=-1, Temp=37.3)


def _router_or_skip():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    return PR.PolicyRouter()


def test_router_default_recommends_tv_change_on_hypercapnia():
    router = _router_or_skip()
    rec = router.run_track_a(_HYPERCAPNIA)                 # no mode → no mask
    assert rec["delta_TV"] != 0, "baseline should want a TV change here"
    assert rec["tv_masked"] is False


def test_router_pressure_control_masks_the_tv_change():
    router = _router_or_skip()
    rec = router.run_track_a(_HYPERCAPNIA, ventilation_mode="pressure_control")
    assert rec["delta_TV"] == 0, "pressure-control must not recommend a ΔTV"
    assert rec["tv_masked"] is True
    assert rec["ventilation_mode"] == "pressure_control"


def test_router_volume_control_leaves_recommendation_unchanged():
    router = _router_or_skip()
    base = router.run_track_a(_HYPERCAPNIA)
    vc = router.run_track_a(_HYPERCAPNIA, ventilation_mode="volume_control")
    assert vc["action"] == base["action"] and vc["tv_masked"] is False


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
