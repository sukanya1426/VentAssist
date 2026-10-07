"""Tests for the confidence calibration study.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.tests.test_confidence_calibration
"""

from __future__ import annotations

import numpy as np

from backend.pipeline import config
from backend.router import policy_router as PR
from benchmark import confidence_calibration as CC


class Skip(Exception):
    pass


def test_perfect_calibration_has_zero_ece():
    """Confidence equal to the realised rate in every bin → ECE 0."""
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 0.85, 20000)
    hit = (rng.uniform(size=20000) < conf).astype(float)
    r = CC._reliability(conf, hit)
    assert r["ece"] < 0.02, f"well-calibrated input gave ECE {r['ece']}"


def test_systematic_overconfidence_is_detected():
    """Claim 0.9 everywhere, be right half the time → ECE near 0.4."""
    conf = np.full(5000, 0.9)
    conf[0] = 0.89                       # a little spread so bins exist
    hit = np.zeros(5000); hit[:2500] = 1.0
    r = CC._reliability(conf, hit)
    assert r["ece"] > 0.3, f"overconfidence not detected (ECE {r['ece']})"
    assert r["bins"][-1]["gap_confidence_minus_observed"] > 0, \
        "a positive gap must mean 'claimed more than delivered'"


def test_constant_confidence_is_reported_not_silently_scored():
    """Zero spread cannot be calibrated — say so rather than return ECE 0."""
    r = CC._reliability(np.full(100, 0.6), np.ones(100))
    assert r["ece"] is None
    assert "constant" in r["note"].lower()


def test_reliability_bins_partition_every_row():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.3, 0.85, 3000)
    hit = rng.integers(0, 2, 3000).astype(float)
    r = CC._reliability(conf, hit)
    assert sum(b["n"] for b in r["bins"]) == len(conf), "bins lost or duplicated rows"
    # `share` is rounded to 4dp for the artifact, so 10 bins can accumulate up to
    # 10 * 5e-5 of rounding error. The exact check is the count above.
    assert abs(sum(b["share"] for b in r["bins"]) - 1.0) < 1e-3


def test_spearman_matches_known_cases():
    x = np.arange(100, dtype=float)
    assert CC._spearman(x, x) > 0.999
    assert CC._spearman(x, -x) < -0.999
    assert abs(CC._spearman(np.ones(50), np.arange(50.0))) < 1e-9 or \
        np.isnan(CC._spearman(np.ones(50), np.arange(50.0)))


def test_confidence_uses_the_served_formula_not_a_copy():
    """_confidences must agree with the router's own function, row by row.

    A calibration study of a reimplemented formula would audit something the UI
    never shows, so this pins the delegation rather than the arithmetic.
    """
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(40, 125)).astype(np.float32)
    conf, margin, chosen = CC._confidences(Q, None)
    for i in range(len(Q)):
        ref = PR._decision_confidence(Q[i], PR.CONF_WEIGHT_CLINICAL,
                                      chosen=int(Q[i].argmax()))
        assert abs(conf[i] - ref["confidence"]) < 1e-9
        assert abs(margin[i] - ref["decision_margin"]) < 1e-9
        assert chosen[i] == int(Q[i].argmax())


def test_confidence_respects_the_structural_bounds():
    rng = np.random.default_rng(4)
    # Huge margins would push an unclipped sigmoid product to 1.0.
    Q = rng.normal(size=(200, 125)).astype(np.float32) * 50
    conf, _, _ = CC._confidences(Q, None)
    assert conf.min() >= PR.CONF_FLOOR - 1e-9
    assert conf.max() <= PR.CONF_CEIL + 1e-9
    # Track A can never exceed its information-content cap.
    assert conf.max() <= PR.CONF_WEIGHT_CLINICAL + 1e-9, \
        "clinical-only confidence exceeded CONF_WEIGHT_CLINICAL"


def test_ood_tempering_lowers_confidence():
    """An out-of-support state must not be as confident as an in-support one."""
    rng = np.random.default_rng(5)
    Q = rng.normal(size=(50, 125)).astype(np.float32)
    full = CC._confidences(Q, np.ones(50))[0]
    ood = CC._confidences(Q, np.zeros(50))[0]
    assert np.all(ood <= full + 1e-9), "OOD states were not tempered"
    assert np.any(ood < full), "tempering had no effect at all"


def test_real_calibration_runs_and_reports_both_targets():
    if not (config.MODEL_PATH / "policy_track_a.pt").exists():
        raise Skip("policy_track_a.pt not present")
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present")
    r = CC.evaluate("a", write=False, cap=4000)
    p = r["primary_agreement_with_clinician"]
    assert 0.0 <= p["ece"] <= 1.0 and 0.0 <= p["base_rate"] <= 1.0
    assert p["target"].startswith("P(policy action")
    cd = r["confidence_distribution"]
    assert PR.CONF_FLOOR - 1e-9 <= cd["p05"] <= cd["median"] <= cd["p95"]
    # The mortality association must be present AND labelled as not-calibration.
    if "secondary_outcome_association" in r:
        s = r["secondary_outcome_association"]
        assert "NOT a calibration target" in s["interpretation"], \
            "the confounded outcome association must be labelled as such"
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
