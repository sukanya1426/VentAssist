"""Regression test: the deployed Track A model must pass the holistic deploy gate.

Wraps backend.scripts.verify_before_deploy so the §15.3 degeneracy check runs with the
rest of the suite (`python -m backend.tests.test_deploy_gate`), not just inside the
training pipeline. Fails if the on-disk policy_track_a.pt moves the wrong way on any of
the 8 clinical battery cases (incl. the two safety reflexes).
"""

from __future__ import annotations

from backend.pipeline import config
from backend.scripts.verify_before_deploy import BATTERY, verify


def test_deployed_model_passes_battery():
    model_path = config.MODEL_PATH / "policy_track_a.pt"
    assert model_path.exists(), f"missing checkpoint: {model_path}"
    assert verify(model_path), "deployed model failed the holistic deploy battery"
    print("PASS test_deployed_model_passes_battery")


def test_gate_rejects_degenerate_policy():
    # The §15.3 'always raise FiO2 +0.10' collapse must be flagged by the battery.
    degenerate = {"PEEP": 0, "TV": 0, "FiO2": 0.10}
    failed = [name for name, (_, _, ok) in BATTERY.items() if not ok(degenerate)]
    for required in ("stable", "high_peep", "volutrauma", "hypercapnic", "hypocapnic"):
        assert required in failed, f"gate too weak: '{required}' should fail the FiO2-spammer"
    print("PASS test_gate_rejects_degenerate_policy")


if __name__ == "__main__":
    test_deployed_model_passes_battery()
    test_gate_rejects_degenerate_policy()
    print("ALL TESTS PASSED")
