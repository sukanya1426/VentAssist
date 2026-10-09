"""The selection-criterion experiment's verdict logic, on synthetic traces.

The experiment itself trains five seeds for 30,000 steps, which is far too slow
for the test suite. What CAN be tested cheaply — and is where a reporting bug
would actually hide — is the arithmetic that turns a trace into a verdict:

  * does it attribute the right checkpoint to each criterion,
  * does it respect the guards when CRS would otherwise pick a bad checkpoint,
  * does the rank correlation mean what the artifact says it means.

The synthetic trace below is shaped like the real finding (val Q-loss bottoms out
early at a mediocre battery score; CRS keeps climbing to a better one), so a change
that inverted the comparison would fail here rather than in a 12-minute run.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_selection_criterion
"""

from __future__ import annotations

import inspect

from benchmark import selection_criterion as SC


def _trace(rows: list[tuple[int, float, float, int, bool]]) -> list[dict]:
    """(step, val_q_loss, crs, battery, selectable) → the trace shape evaluate() builds."""
    return [{"step": st, "val_q_loss": vq, "crs_training_encoding": crs,
             "crs_serving_encoding": crs, "battery_passed_n": bat,
             "selectable": sel}
            for st, vq, crs, bat, sel in rows]


# The real shape: TD-error is minimised at 5,750 with battery 6/8, while the
# reflexes keep being acquired out to 27,000 where the battery reaches 8/8.
REAL_SHAPE = _trace([
    (1000,  1.60, 0.41, 5, True),
    (5750,  1.48, 0.45, 6, True),
    (10000, 1.53, 0.49, 7, True),
    (20000, 1.57, 0.50, 8, True),
    (27000, 1.61, 0.51, 8, True),
])


def test_each_criterion_is_attributed_its_own_checkpoint():
    v = SC._verdict_for_trace(REAL_SHAPE, "crs_training_encoding")
    assert v["val_q_loss_criterion"]["step"] == 5750, v["val_q_loss_criterion"]
    assert v["val_q_loss_criterion"]["battery_at_that_step"] == 6
    assert v["crs_criterion"]["step"] == 27000, v["crs_criterion"]
    assert v["crs_criterion"]["battery_at_that_step"] == 8
    assert v["crs_beats_val_q_loss_on_the_battery"] is True
    assert v["battery_gain"] == 2
    print("PASS test_each_criterion_is_attributed_its_own_checkpoint")


def test_the_oracle_is_the_earliest_step_reaching_the_best_battery():
    """The oracle is a ceiling, not a criterion — it must not flatter itself with a
    later step that scores the same."""
    v = SC._verdict_for_trace(REAL_SHAPE, "crs_training_encoding")
    assert v["battery_oracle"]["battery"] == 8
    assert v["battery_oracle"]["step"] == 20000, v["battery_oracle"]
    print("PASS test_the_oracle_is_the_earliest_step_reaching_the_best_battery")


def test_a_disqualified_checkpoint_is_not_selected_even_at_the_highest_crs():
    """The guards are part of the criterion, so the verdict must honour them."""
    t = _trace([
        (5750,  1.48, 0.45, 6, True),
        (20000, 1.57, 0.50, 8, True),
        (27000, 1.61, 0.99, 8, False),       # best CRS, but collapsed/unsafe
    ])
    v = SC._verdict_for_trace(t, "crs_training_encoding")
    assert v["crs_criterion"]["step"] == 20000, v["crs_criterion"]
    assert v["crs_criterion"]["n_eligible_checkpoints"] == 2
    print("PASS test_a_disqualified_checkpoint_is_not_selected_even_at_the_highest_crs")


def test_crs_ties_are_broken_towards_the_lower_validation_loss():
    t = _trace([
        (5750,  1.48, 0.50, 6, True),
        (20000, 1.57, 0.50, 8, True),
    ])
    v = SC._verdict_for_trace(t, "crs_training_encoding")
    assert v["crs_criterion"]["step"] == 5750, (
        "a CRS tie must fall back to TD-error, not to the later step")
    print("PASS test_crs_ties_are_broken_towards_the_lower_validation_loss")


def test_the_reported_rank_correlation_distinguishes_the_two_criteria():
    """CRS must order checkpoints like the battery; TD-error must not.

    Not 1.0 even on a monotone trace: the battery saturates at 8/8 while CRS keeps
    climbing, so the tie costs a little rank agreement. That ceiling is a property
    of an 8-point integer target, and it is why the artifact reports the two
    correlations side by side rather than CRS's alone.
    """
    v = SC._verdict_for_trace(REAL_SHAPE, "crs_training_encoding")
    rc = v["rank_correlation_with_battery"]
    assert rc["crs"] > 0.95, rc
    assert rc["negative_val_q_loss"] is not None and rc["negative_val_q_loss"] < 0, rc
    assert rc["crs"] > rc["negative_val_q_loss"], rc
    print("PASS test_the_reported_rank_correlation_distinguishes_the_two_criteria")


def test_spearman_handles_ties_constants_and_short_series():
    assert SC._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert SC._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert SC._spearman([1, 2], [1, 2]) is None, "too short to be meaningful"
    assert SC._spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None, "no variance"
    # ties averaged rather than ordered arbitrarily
    assert SC._spearman([1, 1, 2, 2], [1, 1, 2, 2]) == 1.0
    print("PASS test_spearman_handles_ties_constants_and_short_series")


def test_the_experiment_cannot_feed_the_battery_into_the_criterion():
    """The battery is the ground truth. If it reached selection.score, the whole
    experiment would be circular."""
    from backend.rl import selection as SEL

    params = set(inspect.signature(SEL.score).parameters)
    assert not params & {"battery", "battery_passed_n", "gate"}, params
    src = inspect.getsource(SC._trace_one_seed)
    scoring = [l for l in src.splitlines() if "SEL.score(" in l or "SEL.clinical_reflex" in l]
    assert scoring, "could not find the criterion calls to check"
    for line in scoring:
        assert "battery" not in line, f"battery passed to the criterion: {line.strip()}"
    print("PASS test_the_experiment_cannot_feed_the_battery_into_the_criterion")


def test_evaluate_can_be_told_not_to_write():
    p = inspect.signature(SC.evaluate).parameters["write"]
    assert p.default is True
    print("PASS test_evaluate_can_be_told_not_to_write")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
