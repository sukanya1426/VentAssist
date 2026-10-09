"""Running the benchmark tests must not overwrite the canonical results.

The same defect was found and fixed once already in `backend/ope/` (see
`backend/tests/test_artifacts_not_clobbered.py`): tests called the evaluation
entry points with small settings, those entry points wrote unconditionally, and
the artifact the validation dashboard cited turned out to be a toy run.

`benchmark/` had the identical problem and no guard. `action_density.evaluate`
and `safety_metrics.evaluate` both wrote unconditionally, and
`test_action_density` called the first with `epochs=8` against a default of 15 —
so `results/action_density_track_a.json`, the artifact the Rule 5 likelihood pair
is quoted from, was last written by a reduced-budget test run.

Both entry points now take `write`, their tests pass `write=False`, and
`action_density` records `epochs` so a short run is detectable rather than
silent. This module keeps it that way.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_artifacts_not_clobbered
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from benchmark import action_density as AD
from benchmark import behaviour_compare as BC
from benchmark import checkpoint_battery as CB
from benchmark import comparison_table as CT
from benchmark import fragility_probe as FP
from benchmark import confidence_calibration as CC
from benchmark import reward_critique as RC
from benchmark import runner as RN
from benchmark import safety_metrics as SM
from benchmark import split_compat as SC

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Every entry point that persists a canonical artifact.
WRITERS = {
    "action_density.evaluate": AD.evaluate,
    "safety_metrics.evaluate": SM.evaluate,
    "behaviour_compare.evaluate": BC.evaluate,
    "confidence_calibration.evaluate": CC.evaluate,
    "split_compat.build": SC.build,
    "runner.run": RN.run,
    "checkpoint_battery.evaluate": CB.evaluate,
    "comparison_table.build": CT.build,
    "fragility_probe.evaluate": FP.evaluate,
    "reward_critique.layer1": RC.layer1,
    "reward_critique.layer2": RC.layer2,
    "reward_critique.layer2_multiseed": RC.layer2_multiseed,
}


class Skip(Exception):
    pass


def test_every_writer_can_be_told_not_to_write():
    missing = [n for n, fn in WRITERS.items()
               if "write" not in inspect.signature(fn).parameters]
    assert not missing, (
        f"these persist an artifact with no way to opt out: {missing}. A test that "
        "calls one of them overwrites the canonical result.")


def test_write_defaults_to_true_so_the_cli_still_produces_artifacts():
    """The guard must not have been implemented by making writing opt-in."""
    for name, fn in WRITERS.items():
        default = inspect.signature(fn).parameters["write"].default
        assert default is True, f"{name}: write defaults to {default!r}, expected True"


def test_no_benchmark_test_calls_a_writer_without_disabling_the_write():
    """Source-level check across the whole test package.

    Cheaper and more reliable than running every module and diffing mtimes, and
    it fails on the line that would cause the damage rather than after the fact.
    """
    tests_dir = Path(__file__).resolve().parent
    calls = [("AD.evaluate(", "action_density"), ("SM.evaluate(", "safety_metrics"),
             ("BC.evaluate(", "behaviour_compare"),
             ("CC.evaluate(", "confidence_calibration"),
             ("SC.build(", "split_compat"), ("RN.run(", "runner"),
             ("RC.layer1(", "reward_critique"), ("RC.layer2(", "reward_critique"),
             ("RC.layer2_multiseed(", "reward_critique"),
             ("CT.build(", "comparison_table"),
             ("CB.evaluate(", "checkpoint_battery"),
             ("FP.evaluate(", "fragility_probe")]
    offenders = []
    for f in sorted(tests_dir.glob("test_*.py")):
        if f.name == Path(__file__).name:
            continue
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.split("#")[0]
            for call, mod in calls:
                if call in stripped and "write=False" not in stripped:
                    offenders.append(f"{f.name}:{lineno} calls {mod} without write=False")
    assert not offenders, "tests would overwrite canonical artifacts:\n  " + \
        "\n  ".join(offenders)


def test_action_density_artifact_records_its_epoch_budget():
    """A reduced-budget run must be detectable from the artifact alone."""
    f = RESULTS / "action_density_track_a.json"
    if not f.exists():
        raise Skip("action_density_track_a.json not present — run the module once")
    d = json.loads(f.read_text())
    assert "epochs" in d, (
        "the artifact does not say how many epochs produced it, so a short run "
        "cannot be told from the canonical one")
    default = inspect.signature(AD.evaluate).parameters["epochs"].default
    assert d["epochs"] >= default, (
        f"canonical artifact was produced with epochs={d['epochs']}, below the "
        f"default {default} — regenerate it with "
        "`python -m benchmark.action_density --track a`")


def test_canonical_artifacts_are_full_size_not_toy_runs():
    """n_test on each artifact must be the real held-out split, not a sample."""
    expected = {}
    for name in ("action_density_track_a.json", "safety_metrics_track_a.json",
                 "behaviour_compare_track_a.json",
                 "confidence_calibration_track_a.json"):
        f = RESULTS / name
        if f.exists():
            expected[name] = json.loads(f.read_text()).get("n_test")
    if len(expected) < 2:
        raise Skip("not enough benchmark artifacts present to cross-check")
    sizes = {k: v for k, v in expected.items() if v}
    biggest = max(sizes.values())
    small = {k: v for k, v in sizes.items() if v < biggest}
    assert not small, (
        f"these artifacts were scored on fewer rows than their siblings "
        f"({biggest}): {small} — regenerate them at full size")


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
