"""The test suite must not overwrite the evaluation artifacts it reads.

Found while running the T1–T18 plan: three tests called the OPE entry points with
deliberately tiny settings to stay fast, and those entry points wrote
``backend/logs/*.json`` unconditionally. So running the suite silently replaced
the real evaluation — a 992,100-transition, 400-start rollout — with a 30-start,
3-step toy run, and the validation dashboard then showed those numbers to a
clinician as the evidence behind the deployed model.

That is worse than a wrong number: it is a wrong number wearing the costume of a
real one, and nothing in the system would have flagged it. This file is the
guard. It records each canonical artifact's content hash, runs the OPE-touching
tests in-process, and asserts nothing changed.

Run:  pytest backend/tests/test_artifacts_not_clobbered.py -v
"""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.pipeline import config

# The artifacts the validation view reads, plus the fitted models an evaluation
# depends on. Each is written by a deliberate evaluation run, never by a test.
CANONICAL = [
    config.LOGS_PATH / "nwe_track_a.json",
    config.LOGS_PATH / "dfqe_track_a.json",
    config.LOGS_PATH / "fqe_track_a.json",
    config.LOGS_PATH / "behaviour_track_a.json",
    config.MODEL_PATH / "policy_track_a.pt",
    config.MODEL_PATH / "normaliser_stats.json",
]


def _digest(paths):
    out = {}
    for p in paths:
        out[p.name] = (hashlib.sha256(p.read_bytes()).hexdigest()
                       if p.exists() else None)
    return out


def test_ope_tests_do_not_rewrite_the_canonical_artifacts():
    """Running the OPE-touching tests must leave every artifact byte-identical."""
    before = _digest(CANONICAL)
    if all(v is None for v in before.values()):
        pytest.skip("no artifacts present to protect")

    # Import and run the tests that call into the OPE entry points.
    from backend.tests import test_nwe_safety_metrics as A
    from backend.tests import test_nwe_bandwidths as B
    for mod, fn in ((A, "test_rollout_reports_both_arms_and_finite_v"),
                    (B, "test_rollout_still_runs_with_cv_bandwidths")):
        f = getattr(mod, fn, None)
        if f is None:
            continue
        try:
            f()
        except Exception as e:
            if "Skip" in type(e).__name__:
                continue
            raise

    after = _digest(CANONICAL)
    changed = [k for k in before if before[k] != after[k]]
    assert not changed, (
        f"the test suite rewrote {changed} — an evaluation artifact must only be "
        "written by a deliberate evaluation run, never by a fast smoke test "
        "(pass write=False)")


def test_evaluation_artifacts_describe_a_real_run():
    """A canonical artifact must not be a toy run masquerading as evidence.

    The floors are set well below a genuine evaluation and well above the smoke
    settings the tests use, so this catches a clobbered artifact even if the
    hash check above is somehow bypassed.
    """
    nwe = config.LOGS_PATH / "nwe_track_a.json"
    if not nwe.exists():
        pytest.skip("nwe_track_a.json not present")
    d = json.load(open(nwe))
    assert d.get("n_transitions", 0) > 100_000, \
        f"nwe_track_a.json was built from {d.get('n_transitions')} transitions"
    starts = d.get("n_starts")
    if starts is not None:
        assert starts >= 100, (
            f"nwe_track_a.json is a {starts}-start run — the tests use 30–40, so "
            "this artifact is almost certainly a clobbered smoke run")
    sm = d.get("safety_metrics", {}).get("hybrid_iql", {})
    n_hypox = sm.get("n_episodes_start_lt95")
    if n_hypox is not None:
        assert n_hypox >= 10, (
            f"the hypoxaemic-start safety metric rests on {n_hypox} episode(s); "
            "that is not a number to put in front of a clinician")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
