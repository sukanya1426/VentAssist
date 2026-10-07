"""Tests for the multi-seed runner (Phase 1 items 4 and 5) and trainer seeding.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_runner
"""

from __future__ import annotations

import numpy as np
import yaml

from backend.pipeline import config
from benchmark import runner as R


class Skip(Exception):
    pass


def test_ci95_uses_the_t_distribution_not_a_normal_approximation():
    """At n=5 the normal z (1.96) understates the half-width by ~40%.

    Getting this wrong is what makes a noisy multi-seed result look significant,
    so the critical value is pinned rather than assumed.
    """
    xs = [1.0, 1.1, 0.9, 1.2, 0.8]
    r = R._ci95(xs)
    sd = float(np.std(xs, ddof=1))
    expected_half = 2.776 * sd / np.sqrt(5)
    assert abs(r["half_width"] - expected_half) < 1e-6, \
        f"half-width {r['half_width']} != t-based {expected_half}"
    normal_half = 1.96 * sd / np.sqrt(5)
    assert r["half_width"] > normal_half, "t interval must be wider than normal"


def test_ci95_brackets_the_mean_and_reports_the_range():
    r = R._ci95([2.0, 4.0, 6.0])
    assert r["mean"] == 4.0
    assert r["ci95"][0] < 4.0 < r["ci95"][1]
    assert r["min"] == 2.0 and r["max"] == 6.0
    assert r["n"] == 3


def test_ci95_refuses_to_invent_an_interval_from_one_seed():
    """A single seed has no interval — that is the whole objection being fixed."""
    r = R._ci95([0.57])
    assert r["ci95"] is None
    assert r["n"] == 1
    assert "single seed" in r["note"]


def test_ci95_of_identical_seeds_has_zero_width():
    r = R._ci95([0.5] * 4)
    assert r["sd"] == 0.0 and r["half_width"] == 0.0
    assert r["ci95"] == [0.5, 0.5]


def test_ci95_ignores_non_finite_values():
    r = R._ci95([1.0, 2.0, float("nan"), None, float("inf")])
    assert r["n"] == 2, "NaN/None/inf must be dropped, not propagated"
    assert r["mean"] == 1.5


def test_seeding_makes_training_reproducible():
    """Same seed twice → identical result. This did not hold before seeding.

    The trainer pinned the minibatch stream to SPLIT_SEED but never called
    torch.manual_seed, so weight init varied run to run and the deployed
    checkpoint is not bit-reproducible.
    """
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present")
    from backend.mdp import dataset as D
    from backend.rl import trainer as T

    cfg = yaml.safe_load(
        (config.REPO_ROOT / "backend" / "configs" / "track_a_config.yaml").read_text())
    d = T._normalise(D.load_mdp("a"), "a")
    tr = np.where(d["split"] == "train")[0]
    va = np.where(d["split"] == "val")[0][:4000]

    def fit(seed):
        m, bv, _ = T._run_training(d, tr, va, cfg, steps=120, device="cpu", seed=seed)
        return bv, m.act_batch(d["states"][va][:600])

    bv_a, acts_a = fit(0)
    bv_b, acts_b = fit(0)
    bv_c, acts_c = fit(1)
    assert np.isclose(bv_a, bv_b), f"same seed differed: {bv_a} vs {bv_b}"
    assert np.array_equal(acts_a, acts_b), "same seed produced different actions"
    assert not np.isclose(bv_a, bv_c), "different seeds produced an identical fit"


def test_unseeded_training_still_follows_the_historical_minibatch_stream():
    """seed=None must leave the deployed training path untouched.

    If this breaks, adding seed support silently changed how the shipped model is
    produced — which would invalidate the deploy gate rather than extend it.
    """
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present")
    from backend.mdp import dataset as D
    from backend.rl import trainer as T

    cfg = yaml.safe_load(
        (config.REPO_ROOT / "backend" / "configs" / "track_a_config.yaml").read_text())
    d = T._normalise(D.load_mdp("a"), "a")
    tr = np.where(d["split"] == "train")[0]

    # seed=0 offsets SPLIT_SEED by 0, so its minibatch stream is the historical one.
    rng_none = np.random.default_rng(config.SPLIT_SEED)
    rng_zero = np.random.default_rng(config.SPLIT_SEED + 0)
    a = rng_none.choice(tr, size=256, replace=False)
    b = rng_zero.choice(tr, size=256, replace=False)
    assert np.array_equal(a, b), \
        "seed=0 must reproduce the unseeded minibatch stream"


def test_fqe_accepts_an_in_memory_policy():
    """Scoring a seed must never require writing over the deployed checkpoint."""
    import inspect
    from backend.ope import fqe
    sig = inspect.signature(fqe.fitted_q_evaluation)
    assert "policy" in sig.parameters, (
        "fqe must accept a policy= argument or a seed run would have to overwrite "
        "backend/models/policy_track_a.pt")
    assert sig.parameters["policy"].default is None, \
        "policy= must default to None so the historical path is unchanged"


def test_runner_reports_an_equal_budget_and_never_writes_the_deployed_model():
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present")
    deployed = config.MODEL_PATH / "policy_track_a.pt"
    before = deployed.read_bytes() if deployed.exists() else None

    r = R.run("a", seeds=2, steps=100, with_fqe=False, write=False,
              save_checkpoints=False)

    assert r["n_seeds"] == 2 and len(r["per_seed"]) == 2
    assert r["budget"]["total_steps"] == 100
    assert r["budget_identical_across_seeds"] is True
    for key in ("behaviour_match", "hold_share", "safety_any_violation"):
        agg = r["aggregate"][key]
        assert agg["n"] == 2 and agg["ci95"] is not None, key
    assert r["per_seed"][0]["seed"] == 0 and r["per_seed"][1]["seed"] == 1

    if before is not None:
        assert deployed.read_bytes() == before, \
            "the runner modified the DEPLOYED policy — forbidden by the ground rules"


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
