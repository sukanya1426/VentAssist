"""Tests for the comparison table (BENCHMARK_PLAN Phase 4).

The table's job is not arithmetic — it reads numbers other modules computed. Its
job is to make two specific mistakes impossible, and those are what is pinned
here:

1. A row that does not say which policy it describes. `fqe_track_a.json` is the
   DEPLOYED checkpoint and `runner_track_*.json` are fresh seeds under a
   different checkpoint-selection config; averaging or comparing them is the
   error the plan calls out by name.

2. An agreement figure reported without a value estimate or an action-diversity
   figure beside it. Clinician agreement rises as the policy collapses toward
   holding, so a bare agreement number inverts the conclusion.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_comparison_table
"""

from __future__ import annotations

import inspect

from benchmark import comparison_table as CT


class Skip(Exception):
    pass


def test_rule5_rejects_a_bare_agreement_figure():
    """The guardrail that matters most: agreement alone must raise."""
    try:
        CT._rule5(agreement=0.681, value=None, diversity=None,
                  policy="whatever", source="a bare agreement number")
    except AssertionError:
        return
    raise AssertionError(
        "_rule5 accepted an agreement figure with no value and no diversity. "
        "That is exactly the reporting error the benchmark exists to prevent.")


def test_rule5_accepts_agreement_when_value_or_diversity_is_present():
    for value, diversity in ((2.4823, None), (None, 34.8), (2.4823, 34.8)):
        row = CT._rule5(agreement=0.5322, value=value, diversity=diversity,
                        policy="p", source="s")
        assert row["agreement"] == 0.5322


def test_rule5_allows_a_row_with_no_agreement_at_all():
    """A value-only row is fine — it is agreement that needs chaperoning."""
    row = CT._rule5(agreement=None, value=1.8787, diversity=None,
                    policy="p", source="s")
    assert row["value"] == 1.8787


def test_every_row_must_name_its_policy():
    try:
        CT._assert_single_policy([{"agreement": 0.5}], "synthetic")
    except AssertionError:
        return
    raise AssertionError(
        "_assert_single_policy accepted a row with no `policy` key, so the "
        "deployed checkpoint and fresh seeds could be tabulated as one system.")


def test_deployed_and_fresh_seed_policies_have_distinct_labels():
    labels = {CT.DEPLOYED, CT.SEEDS_COARSE, CT.SEEDS_FINE, CT.CLINICIAN}
    assert len(labels) == 4, "policy provenance labels must be distinguishable"


def test_build_takes_a_write_flag_defaulting_to_true():
    """Ground rule 4: every entry point that persists a result takes `write`."""
    p = inspect.signature(CT.build).parameters
    assert "write" in p, "build() persists artifacts with no way to opt out"
    assert p["write"].default is True, "write must default to True for the CLI"


def test_a_missing_required_artifact_raises_rather_than_blanking_a_cell():
    assert issubclass(CT.MissingArtifact, Exception)
    try:
        CT._load("this_artifact_does_not_exist.json")
    except CT.MissingArtifact:
        return
    raise AssertionError("_load silently tolerated a missing required artifact")


def test_the_retracted_capability_claim_is_marked_not_deleted():
    """A differentiator that failed its check must stay visible, marked false.

    Deleting it would leave the plan's claim #4 unchallenged in any document
    that was written before the check.
    """
    retracted = [c for c in CT.CAPABILITIES if not c["verified"]]
    assert retracted, (
        "no capability claim is marked unverified — the 1h-vs-4h retraction "
        "should be recorded here")
    for c in retracted:
        assert "RETRACTED" in c["source"], f"{c['claim']}: no retraction rationale"


def test_capability_claims_each_cite_a_source():
    for c in CT.CAPABILITIES:
        assert c.get("source"), f"{c['claim']}: no source path given"


def test_every_layer2_metric_is_classified_by_scale():
    """SCALE_FREE and SCALE_DEPENDENT must together cover every aggregated metric.

    A metric added to `reward_critique._AGG` and forgotten here would be rendered
    as cross-arm comparable by omission — and Q-denominated metrics are not,
    because their reward scales differ by ~85x.
    """
    from benchmark import reward_critique as RC
    classified = set(CT.SCALE_FREE) | set(CT.SCALE_DEPENDENT)
    missing = set(RC._AGG) - classified
    assert not missing, (
        f"these Layer 2 metrics are unclassified: {sorted(missing)} — add each to "
        "SCALE_FREE or SCALE_DEPENDENT in comparison_table.py")
    assert not (set(CT.SCALE_FREE) & set(CT.SCALE_DEPENDENT)), \
        "a metric cannot be both scale-free and scale-dependent"


def test_q_denominated_metrics_are_not_marked_cross_arm_comparable():
    """The specific error: quoting a Q margin across two different rewards."""
    for m in ("q_top2_margin", "q_spread_max_minus_min", "best_val_q"):
        assert m in CT.SCALE_DEPENDENT, (
            f"{m} is denominated in Q units and scales with reward magnitude; it "
            "must not be presented as a cross-arm comparison")


def test_build_renders_without_the_optional_artifacts():
    """The table must be assemblable before the multi-seed critique exists."""
    try:
        r = CT.build(write=False)
    except CT.MissingArtifact as e:
        raise Skip(f"a required artifact is absent: {e}")
    md = CT.render_markdown(r)
    assert "Table A" in md and "Table B" in md and "Table C" in md and "Table D" in md
    assert "NOT the arena" in md, "the scope deviation must survive rendering"


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
