"""Tests for the stratified/frozen benchmark split (BENCHMARK_PLAN Phase 1 item 3).

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_split_compat
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark import split_compat as SC


class Skip(Exception):
    pass


def _toy(n: int = 400) -> pd.DataFrame:
    """A synthetic per-stay frame: lengths spread wide, 20% mortality."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "stay_id": np.arange(30000000, 30000000 + n),
        "n_transitions": rng.integers(4, 200, size=n),
        "shipped_split": rng.choice(["train", "val", "test"], size=n,
                                    p=[0.7, 0.1, 0.2]),
        "died_horizon": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })


def test_strata_are_length_quartile_by_outcome():
    g = SC._assign_strata(_toy())
    assert g["length_quartile"].nunique() == 4, "expected four length quartiles"
    assert set(g["outcome_class"]) <= {"died", "survived", "outcome_unknown"}
    # 4 quartiles x 2 observed outcome classes
    assert g["stratum"].nunique() == 8, f"expected 8 strata, got {g['stratum'].nunique()}"


def test_quartiles_are_balanced_even_with_heavy_ties():
    """qcut on raw values collapses bins when many stays share a length.

    Short ventilation courses are extremely common, so the lowest quartile is
    exactly where ties pile up. Ranking first is what keeps four non-empty bins.
    """
    g = _toy(200)
    g["n_transitions"] = 6                      # every stay identical
    out = SC._assign_strata(g)
    counts = out["length_quartile"].value_counts()
    assert len(counts) == 4, f"ties collapsed the quartiles: {dict(counts)}"
    assert counts.max() - counts.min() <= 1, f"quartiles not balanced: {dict(counts)}"


def test_outcome_column_survives_both_int_and_bool_dtypes():
    """outcomes.csv gives int64; an in-memory frame gives bool. Both must map.

    A dict literal {True: 1, False: 0} matches neither an int64 nor a float
    column, which silently mapped every row to NaN and halved the stratum count
    while reporting 'no outcome labels available'.
    """
    for val in (1, True, np.int64(1), 1.0):
        g = _toy(80)
        g["died_horizon"] = val
        out = SC._assign_strata(g)
        assert out["died"].notna().all(), f"dtype {type(val)} lost the outcome"
        assert (out["outcome_class"] == "died").all(), f"dtype {type(val)} mis-mapped"


def test_unknown_outcome_is_its_own_class_not_a_survivor():
    g = _toy(120)
    g.loc[:39, "died_horizon"] = np.nan
    out = SC._assign_strata(g)
    assert (out["outcome_class"] == "outcome_unknown").sum() == 40, \
        "stays with no outcome label must not be folded in with survivors"


def test_allocation_preserves_the_shipped_proportions():
    g = SC._assign_strata(_toy(600))
    prop = SC._target_proportions(g)
    g["bench_split"] = SC._stratified_split(g, prop)
    assert set(g["bench_split"]) == {"train", "val", "test"}
    for s in SC.SPLITS:
        got = (g["bench_split"] == s).mean()
        assert abs(got - prop[s]) < 0.02, \
            f"{s}: got {got:.4f}, shipped proportion {prop[s]:.4f}"


def test_every_stay_is_assigned_exactly_once():
    g = SC._assign_strata(_toy(300))
    g["bench_split"] = SC._stratified_split(g, SC._target_proportions(g))
    assert g["bench_split"].notna().all(), "some stay was left unassigned"
    assert len(g) == len(g["stay_id"].unique())


def test_allocation_is_deterministic_from_stay_ids_alone():
    """A reviewer rebuilding the cohort must get the same split, row order aside."""
    g = SC._assign_strata(_toy(300))
    prop = SC._target_proportions(g)
    a = SC._stratified_split(g, prop)
    shuffled = g.sample(frac=1.0, random_state=7)
    b = SC._stratified_split(shuffled, prop)
    joined = pd.DataFrame({"a": a, "b": b.reindex(a.index)})
    assert (joined["a"] == joined["b"]).all(), \
        "split changed when the input row order changed"


def test_stratified_split_is_better_balanced_than_a_random_one():
    g = SC._assign_strata(_toy(800))
    prop = SC._target_proportions(g)
    g["bench_split"] = SC._stratified_split(g, prop)
    strat = max(r["abs_deviation"]
                for r in SC._balance_table(g, "bench_split").values())
    shipped = max(r["abs_deviation"]
                  for r in SC._balance_table(g, "shipped_split").values())
    assert strat <= shipped + 1e-9, \
        f"stratifying made balance worse: {strat:.4f} vs random {shipped:.4f}"


def test_load_refuses_to_guess_when_the_split_was_never_frozen():
    """A benchmark run must never silently fall back to the shipped split."""
    missing = SC.RESULTS / "split_compat_track_zzz.csv"
    if missing.exists():
        raise Skip("unexpected artifact present")
    try:
        SC.load("zzz")
    except FileNotFoundError as e:
        assert "frozen" in str(e).lower() or "run" in str(e).lower()
    else:
        raise AssertionError("load() must raise when the frozen split is absent")


def test_real_split_audit_runs_and_reports_a_verdict():
    from backend.pipeline import config
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present")
    r = SC.build("a", write=False)
    assert r["n_strata"] >= 4
    assert r["n_stays"] > 0 and r["n_transitions"] > 0
    assert set(r["target_proportions"]) == set(SC.SPLITS)
    for name in ("shipped", "stratified"):
        s = r[name]["summary"]
        assert s["n_stays"] > 0
        assert 0.0 <= s["max_abs_stratum_deviation"] <= 1.0
    assert r["stratified"]["summary"]["max_abs_stratum_deviation"] <= \
        r["shipped"]["summary"]["max_abs_stratum_deviation"] + 1e-6, \
        "stratification must not worsen balance on the real cohort"
    assert isinstance(r["verdict"], str) and r["verdict"]


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
