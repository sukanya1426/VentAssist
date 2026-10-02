"""Tests for the runtime responsiveness knob (SYSTEM_SUMMARY §16 item 9).

``responsiveness`` ∈ [0, 1] biases the policy toward acting by adding a bonus to
every non-hold action before the argmax
([policy_router.py](backend/router/policy_router.py) `_apply_responsiveness`).
The invariants that make it safe/honest:

  1. ``responsiveness == 0`` is the identity (deployed/gated behaviour).
  2. Raising it can only ever turn "hold" into "act" — it never changes *which*
     action is chosen when the policy already acts (same bonus on all non-hold
     actions ⇒ argmax among them is preserved).
  3. A responsiveness-induced action is reported with low confidence (the raw Q
     preferred to hold, so the decision margin is negative).

Layers 1–2 are pure-function tests (no model). Layer 3 needs the trained Track A
model and skips cleanly if absent.

Run:  python -m backend.tests.test_responsiveness
"""

from __future__ import annotations

import numpy as np

from backend.mdp import action_space
from backend.pipeline import config
from backend.router import policy_router as PR


class Skip(Exception):
    """Raised to skip a test when its prerequisites (a trained model) are absent."""


# --------------------------------------------------------------------------- #
# 1. Pure-function tests of _apply_responsiveness (no trained model required)
# --------------------------------------------------------------------------- #
def test_zero_responsiveness_is_identity():
    q = np.random.default_rng(0).normal(size=125)
    out = PR._apply_responsiveness(q, 0.0)
    assert np.array_equal(out, q)


def test_responsiveness_can_unseat_a_narrowly_held_hold():
    """Hold wins by a thin margin → enough responsiveness flips it to act."""
    q = np.zeros(125)
    q[PR.HOLD_ACTION] = 0.3          # hold is the argmax, but only just
    other = (PR.HOLD_ACTION + 1) % 125
    q[other] = 0.1
    assert int(np.argmax(q)) == PR.HOLD_ACTION          # baseline holds
    # bonus needed to flip: 0.3 - 0.1 = 0.2 → responsiveness > 0.1 at scale 2.0
    flipped = PR._apply_responsiveness(q, 0.5)
    assert int(np.argmax(flipped)) == other, "responsiveness failed to unseat hold"


def test_responsiveness_never_changes_choice_among_actions():
    """When a non-hold action already wins, the choice is invariant to the knob."""
    q = np.zeros(125)
    winner = (PR.HOLD_ACTION + 7) % 125
    q[winner] = 5.0                  # a clear non-hold winner
    q[PR.HOLD_ACTION] = 1.0
    base = int(np.argmax(q))
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert int(np.argmax(PR._apply_responsiveness(q, r))) == base


# --------------------------------------------------------------------------- #
# 2. Policy tests over presets (need the trained model)
# --------------------------------------------------------------------------- #
_STABLE = dict(PEEP=8, TV=460, FiO2=0.4, SpO2=95, PaO2=88, PaCO2=40, pH=7.40,
               HR=84, SBP=120, RR=16, RASS=-1, Temp=37.0)
# Already-acting cases: the recommendation must be invariant to responsiveness.
_ACTING = {
    "hypoxaemia":  dict(PEEP=8, TV=480, FiO2=0.5, SpO2=84, PaO2=55, PaCO2=44, pH=7.34,
                        HR=104, SBP=112, RR=26, RASS=-2, Temp=37.6),
    "hypercapnia": dict(PEEP=8, TV=320, FiO2=0.5, SpO2=93, PaO2=72, PaCO2=65, pH=7.28,
                        HR=98, SBP=118, RR=28, RASS=-1, Temp=37.3),
}


def _router_or_skip():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present — train first.")
    return PR.PolicyRouter()


def test_default_matches_deploy_gate_hold_on_stable():
    router = _router_or_skip()
    rec = router.run_track_a(_STABLE)                     # responsiveness defaults to 0
    assert (rec["delta_PEEP"], rec["delta_TV"], rec["delta_FiO2"]) == (0, 0, 0.0), \
        "stable must still hold at responsiveness=0 (deploy-gate behaviour)"


def test_high_responsiveness_makes_stable_act():
    router = _router_or_skip()
    rec = router.run_track_a(_STABLE, responsiveness=1.0)
    acted = any([rec["delta_PEEP"], rec["delta_TV"], rec["delta_FiO2"]])
    assert acted, "max responsiveness should move the stable patient off hold"
    # ...and it should be honestly low-confidence (raw Q preferred hold)
    assert rec["decision_margin"] <= 0, "an induced action should have a non-positive margin"


def test_overridden_hold_is_surfaced_as_an_alternative():
    """The action responsiveness overrode must stay visible to the clinician.

    Regression: the ranking was computed on the responsiveness-BIASED Q, which
    lifted every non-hold action above "hold" and pushed "hold" out of the top-3
    — so at responsiveness ≥ 0.5 the panel showed three variants of "act" and no
    hint that the policy had preferred to do nothing.
    """
    router = _router_or_skip()
    for r in (0.5, 1.0):
        rec = router.run_track_a(_STABLE, responsiveness=r)
        if rec["action"] == PR.HOLD_ACTION:
            continue                      # never overridden; nothing to surface
        holds = [a for a in rec["alternatives"]
                 if (a["delta_PEEP"], a["delta_TV"], a["delta_FiO2"]) == (0, 0, 0.0)]
        assert holds, (f"responsiveness={r} overrode hold but hold is absent from "
                       f"alternatives — the override is invisible to the clinician")
        assert holds[0]["margin_from_best"] < 0, \
            "an overridden hold must carry a negative margin (policy rated it higher)"


def test_reported_margins_are_free_of_the_responsiveness_bonus():
    """Margins describe the policy's real preference, not the biased ranking.

    Regression: margins were inflated by the bonus, so an alternative the policy
    rated ABOVE the served action was reported with a positive margin — the sign
    was flipped, reading as "slightly worse" when it was actually better.
    """
    router = _router_or_skip()
    base = router.run_track_a(_STABLE, responsiveness=0.0)
    q_by_action = {e["action_idx"]: e["q_value"] for e in base["ranked"]}
    for r in (0.25, 0.5, 1.0):
        rec = router.run_track_a(_STABLE, responsiveness=r)
        for e in rec["ranked"]:
            if e["action_idx"] not in q_by_action:
                continue                  # not in the unbiased top-3; nothing to compare
            assert abs(e["q_value"] - q_by_action[e["action_idx"]]) < 1e-6, \
                (f"responsiveness={r} leaked the bonus into the reported q_value for "
                 f"action {e['action_idx']}")
        chosen_q = rec["ranked"][0]["q_value"]
        for e in rec["ranked"]:
            assert abs(e["margin_from_best"] - (chosen_q - e["q_value"])) < 1e-6, \
                "margin must equal (served Q − alternative Q) on the raw Q-values"


def test_ranking_leads_with_the_served_action():
    """Rank 0 is what the clinician was told to do, at every responsiveness."""
    router = _router_or_skip()
    for r in (0.0, 0.25, 0.5, 1.0):
        rec = router.run_track_a(_STABLE, responsiveness=r)
        assert rec["ranked"][0]["action_idx"] == rec["action"], \
            f"responsiveness={r}: ranked[0] is not the served action"
        assert rec["ranked"][0]["margin_from_best"] == 0.0
        idxs = [e["action_idx"] for e in rec["ranked"]]
        assert len(idxs) == len(set(idxs)), "duplicate action in the ranked list"


def test_acting_cases_are_invariant_to_responsiveness():
    router = _router_or_skip()
    for name, s in _ACTING.items():
        base = router.run_track_a(s, responsiveness=0.0)
        for r in (0.25, 0.5, 1.0):
            rec = router.run_track_a(s, responsiveness=r)
            assert rec["action"] == base["action"], \
                f"{name}: responsiveness={r} changed the chosen action (must not)"


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
