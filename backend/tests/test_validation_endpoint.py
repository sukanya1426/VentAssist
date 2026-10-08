"""GET /api/validation must always have something to show — for BOTH tracks.

WHY THIS FILE EXISTS. The dashboard's entry point to the model-evidence page is
`ValidationLink`, and it hid itself on this condition:

    if (failed || !data || (!data.estimators.length && !data.safety.length)) return null;

For Track A that is harmless. For Track B both arrays are legitimately empty —
Track B has no FQE/DFQE/NWE record of its own, because its deployed checkpoint IS
Track A's weights with an untouched `[I | 0]` adapter, so there is no distinct
policy to evaluate. The card therefore vanished with no error and no message, and
the "How this model was evaluated" page became unreachable from a Track B patient.

A silently absent page is worse than a broken one: there is nothing to notice. So
the contract pinned here is that the endpoint must always return at least one of
`model`, `estimators`, `safety` or `waveform_ablation`, and that Track B must
carry the waveform ablation specifically — a measured null result is evidence,
silence is not.

Run:  .venv/bin/python -m pytest backend/tests/test_validation_endpoint.py -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import models as M
from backend.api.auth import current_user
from backend.api.main import app


@pytest.fixture()
def client():
    """Signed-in client. /api/validation is behind the auth gate."""
    app.dependency_overrides[current_user] = lambda: M.AuthUser(
        id="test-validation", username="test-validation")
    yield TestClient(app)
    app.dependency_overrides.clear()


def _payload(client, track: str) -> dict:
    r = client.get(f"/api/validation?track={track}")
    assert r.status_code == 200, f"track {track}: HTTP {r.status_code}"
    return r.json()


def _has_content(d: dict) -> bool:
    """The exact condition ValidationLink uses to decide whether to render."""
    return bool(d.get("estimators") or d.get("safety")
                or d.get("waveform_ablation") or d.get("model"))


@pytest.mark.parametrize("track", ["a", "b"])
def test_validation_always_has_something_to_render(client, track):
    """THE regression: the entry card must never hide itself on either track."""
    d = _payload(client, track)
    assert _has_content(d), (
        f"track {track}: the endpoint returned nothing renderable, so "
        "ValidationLink hides itself and the model-evidence page becomes "
        "unreachable from the dashboard — with no error shown. That is the "
        "defect this file exists to prevent.")


@pytest.mark.parametrize("track", ["a", "b"])
def test_deployed_checkpoint_provenance_is_always_reported(client, track):
    """Provenance comes from the checkpoint itself, so it is never track-specific."""
    d = _payload(client, track)
    assert d.get("model"), f"track {track}: no deployed-checkpoint provenance"


def test_track_a_carries_the_three_estimators_and_the_baseline(client):
    d = _payload(client, "a")
    keys = {e["key"] for e in d["estimators"]}
    assert {"fqe", "dfqe", "nwe"} <= keys, (
        f"track a is missing estimators: expected fqe/dfqe/nwe, got {sorted(keys)}. "
        "These read from backend/logs/*.json, which must be TRACKED in git — the "
        "container is built from a clone, so an untracked log is an absent log.")
    assert d.get("baseline"), "track a: no clinician baseline to read the estimates against"


def test_track_b_carries_the_waveform_ablation(client):
    """Track B's only evidence is the ablation, and its answer is zero."""
    d = _payload(client, "b")
    wa = d.get("waveform_ablation")
    assert wa is not None, (
        "track b: no waveform ablation. Without it Track B has nothing to show "
        "and the entry card hides itself.")
    assert wa["delta_v"] == 0.0, (
        f"waveform delta_v is {wa['delta_v']}, expected exactly 0.0. The deployed "
        "Track B checkpoint is Track A's weights with an untouched [I | 0] adapter, "
        "so the two arms are the same policy. A non-zero value here means the "
        "checkpoint changed and the report's 'honest negative' claim needs revisiting.")
    assert wa["is_underpowered"] is True, (
        "the ablation rests on 8 held-out episodes; it must keep declaring itself "
        "underpowered so no significance is read into the interval")


def test_track_a_has_no_waveform_ablation(client):
    """The ablation is a Track B artifact; showing it on Track A would mislead."""
    assert _payload(client, "a").get("waveform_ablation") is None


def test_every_value_carries_the_reward_relative_caveat(client):
    """No V̂ may be presented as a patient outcome."""
    for track in ("a", "b"):
        caveat = _payload(client, track).get("caveat") or ""
        assert "reward function" in caveat, (
            f"track {track}: the caveat no longer states that values are returns "
            "under VentAssist's own reward. Every number on this page is "
            "reward-model-relative and the UI must carry that wording.")


def test_unknown_track_falls_back_to_a_rather_than_erroring(client):
    """A bad query string must not break the page."""
    d = _payload(client, "nonsense")
    assert d["track"] == "a"
    assert _has_content(d)
