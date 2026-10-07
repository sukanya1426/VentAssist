"""Track B end-to-end on REAL extracted waveform features.

These were written after an end-to-end run on the real `mimic4wdb` extractions
showed Track B "working" while doing none of what it claimed. Three defects, all
pinned below:

  1. The waveform values reached nothing. The router evaluated the 12-dim Track A
     Q-function, so no waveform number could influence the output — a perfectly
     trained 18-dim policy would have been ignored too.
  2. Confidence rose with the COUNT of supplied waveform numbers, not their
     content: perturbing any feature across its whole observed range left the
     confidence bit-identical, yet supplying all six lifted it 0.696 -> 0.819 over
     Track A for the same patient. Unearned confidence in a CDSS.
  3. ``imputation_used`` was reported True whenever coverage < 1 while the loaded
     ``FeatureImputer`` was never called.

The central property is subtle: it is CORRECT for waveform data to change nothing
today, because the trained model assigns the waveform dims zero weight. What must
not happen is the system claiming otherwise. So the tests assert the *coupling*
between influence and effect, in both directions — no influence means no effect
and no extra confidence; real influence means real effect.

Run:  pytest backend/tests/test_track_b_waveform.py -v
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import torch

from backend.pipeline import config
from backend.router import policy_router as PR

TABULAR = config.TABULAR_FEATURES
WAVEFORM = PR.WAVEFORM


class Skip(Exception):
    pass


def _router():
    try:
        return PR.PolicyRouter()
    except Exception as e:
        pytest.skip(f"router unavailable: {e}")


def _real_rows(n: int = 40) -> pd.DataFrame:
    """Real (tabular state, extracted waveform) pairs from the Track B pipeline."""
    wv_p = config.PROCESSED_PATH / "waveform_features_track_b.parquet"
    st_p = config.PROCESSED_PATH / "tabular_states_track_b.parquet"
    if not wv_p.exists() or not st_p.exists():
        pytest.skip("Track B artifacts not built — run run_track_b_pipeline first.")
    wv = pd.read_parquet(wv_p)
    st = pd.read_parquet(st_p)
    paired = st.merge(wv, on=["stay_id", "hour"], how="inner")
    full = paired.dropna(subset=WAVEFORM + list(TABULAR))
    if full.empty:
        pytest.skip("no complete real (state, waveform) rows available")
    return full.head(n).reset_index(drop=True)


def _split(row) -> tuple[dict, dict]:
    return ({f: float(row[f]) for f in TABULAR},
            {f: float(row[f]) for f in WAVEFORM})


# --------------------------------------------------------------------------- #
# It serves real waveform data at all
# --------------------------------------------------------------------------- #
def test_serves_every_real_waveform_row():
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    rows = _real_rows()
    for i, row in rows.iterrows():
        S, WV = _split(row)
        out = r.run_track_b(S, WV)
        assert 0 <= out["action"] < 125, f"row {i}: action out of range"
        assert np.isfinite(out["confidence"]) and 0.0 <= out["confidence"] <= 1.0, \
            f"row {i}: confidence {out['confidence']} not a probability"
        assert out["waveform_coverage"] == pytest.approx(1.0)
        assert out["waveform_used"] is True


def test_uses_the_full_18_dim_state_when_the_policy_is_trusted():
    """The recommendation must be computed FROM the waveform state.

    Regression for defect 1: the router used to evaluate the 12-dim Track A
    Q-function, so the waveform dims were structurally unable to matter.
    """
    r = _router()
    if r.track_b is None or not r.track_b_trusted:
        pytest.skip("Track B policy not trusted (no init_from) — delegation is correct")
    S, WV = _split(_real_rows(1).iloc[0])
    out = r.run_track_b(S, WV)
    assert out["delegated_to_track_a"] is False
    assert len(out["feature_order"]) == len(TABULAR) + len(WAVEFORM)
    assert len(out["state_norm"]) == len(TABULAR) + len(WAVEFORM), \
        "the served state vector does not include the waveform dims"


# --------------------------------------------------------------------------- #
# Confidence must be earned (defect 2)
# --------------------------------------------------------------------------- #
def test_no_confidence_bonus_when_waveform_has_no_influence():
    """Track B may not read more confident than Track A for unused data.

    With a [I | 0] warm start that fine-tuning never improved on, Track B computes
    exactly Track A's recommendation — so it must report exactly Track A's
    confidence. It used to add up to +0.12 purely for the presence of numbers.
    """
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    if r.waveform_influence > 0:
        pytest.skip("this model does use the waveform dims — see the paired test")
    for _, row in _real_rows(25).iterrows():
        S, WV = _split(row)
        b, a = r.run_track_b(S, WV), r.run_track_a(S)
        assert b["confidence"] == pytest.approx(a["confidence"]), (
            "Track B claims more confidence than Track A while the waveform dims "
            "have zero influence on the recommendation")
        assert b["action"] == a["action"]
        assert b["waveform_informative"] is False


def test_waveform_values_cannot_change_output_without_influence():
    """No influence ⇒ no effect, across each feature's whole observed range."""
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    if r.waveform_influence > 0:
        pytest.skip("this model does use the waveform dims")
    rows = _real_rows(200)
    S, WV = _split(rows.iloc[0])
    base = r.run_track_b(S, WV)
    for f in WAVEFORM:
        for v in (rows[f].min(), rows[f].max()):
            out = r.run_track_b(S, {**WV, f: float(v)})
            assert out["action"] == base["action"]
            assert out["confidence"] == pytest.approx(base["confidence"])


def test_influence_detector_reads_the_adapter():
    """[I | 0] ⇒ 0 influence; a weighted waveform block ⇒ positive influence."""
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    assert PR._waveform_influence(r.track_b, len(TABULAR)) == pytest.approx(
        r.waveform_influence)
    # A 12-dim identity-adapter policy has no extra dims at all.
    assert PR._waveform_influence(r.track_a, len(TABULAR)) == 0.0


def test_waveform_influences_output_once_the_model_gives_it_weight():
    """The paired direction: real influence ⇒ real effect.

    Without this, "nothing changed" is indistinguishable from a dead pathway. The
    adapter's waveform block is given weight on a COPY of the loaded policy, and
    the recommendation must then respond to the waveform values.
    """
    r = _router()
    if r.track_b is None or not r.track_b_trusted:
        pytest.skip("Track B policy not trusted")
    rows = _real_rows(200)
    S, WV = _split(rows.iloc[0])
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        r.track_b.adapter.proj.weight[:, len(TABULAR):] = torch.randn(
            len(TABULAR), len(WAVEFORM), generator=g) * 0.8
    r.waveform_influence = PR._waveform_influence(r.track_b, len(TABULAR))
    assert r.waveform_influence > 0

    base = r.run_track_b(S, WV)
    assert base["waveform_informative"] is True
    moved = 0
    for f in WAVEFORM:
        for v in (rows[f].min(), rows[f].max()):
            out = r.run_track_b(S, {**WV, f: float(v)})
            if out["action"] != base["action"] or \
                    abs(out["confidence"] - base["confidence"]) > 1e-9:
                moved += 1
    assert moved > 0, ("waveform values changed nothing even with a weighted "
                       "adapter — the 18-dim pathway is dead")


# --------------------------------------------------------------------------- #
# Imputation must be real (defect 3)
# --------------------------------------------------------------------------- #
def test_missing_features_are_actually_imputed():
    """``imputation_used`` must describe work that happened.

    The flag used to be ``coverage < 1.0`` while the FeatureImputer sat unused.
    """
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    if r.imputer is None:
        pytest.skip("feature imputer not available")
    S, WV = _split(_real_rows(1).iloc[0])
    partial = {**WV, "RRV": None, "Breathing_Regularity": None}
    out = r.run_track_b(S, partial)
    assert out["imputation_used"] is True
    assert out["waveform_coverage"] == pytest.approx(4 / 6)
    # The imputer must return usable numbers for the dropped features.
    filled = r.imputer.impute(S, partial)
    for f in ("RRV", "Breathing_Regularity"):
        assert np.isfinite(filled[f]), f"{f} was not imputed to a finite value"


def test_full_coverage_does_not_claim_imputation():
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    S, WV = _split(_real_rows(1).iloc[0])
    out = r.run_track_b(S, WV)
    assert out["imputation_used"] is False
    assert out["waveform_coverage"] == pytest.approx(1.0)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_markers_are_both_treated_as_absent(missing):
    """``None`` from JSON and ``NaN`` from a parquet row must behave identically."""
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    S, _ = _split(_real_rows(1).iloc[0])
    out = r.run_track_b(S, {f: missing for f in WAVEFORM})
    assert out["waveform_coverage"] == pytest.approx(0.0)
    assert 0 <= out["action"] < 125
    assert np.isfinite(out["confidence"])


def test_zero_coverage_still_produces_a_recommendation():
    """No waveform at all must degrade to a clinical answer, not an error."""
    r = _router()
    if r.track_b is None:
        pytest.skip("Track B model not available")
    S, _ = _split(_real_rows(1).iloc[0])
    out = r.run_track_b(S, {})
    assert out["waveform_coverage"] == pytest.approx(0.0)
    assert 0 <= out["action"] < 125


def test_client_does_not_fabricate_waveform_values():
    """The dashboard must not substitute invented numbers for a missing recording.

    The router honesty asserted above is only as good as what reaches it. The
    store used to hold six plausible constants (``DEFAULT_WAVEFORM`` = HRV 31.2,
    arrhythmia 0.03, perfusion 2.1, RRV 0.19, regularity 0.81, asynchrony 0.1)
    and fall back to them for any patient without a waveform — which is all six
    presets, and any clinical-only upload once Track B was toggled on. They were
    POSTed as ``ecg_features`` / ``pleth_features`` / ``resp_features``, so:

      * ``waveform_coverage`` always read 1.0 and ``imputation_used`` always read
        False, for patients with no recording at all;
      * ``backend/router/feature_imputer.py`` — trained for exactly this case —
        could never run, because the client never admitted anything was missing;
      * the fabricated values were written to the ``recommendation`` row's
        waveform columns, so the stored clinical record claimed measurements that
        were never taken.

    This is the client-side half of defect 2 in this module's docstring, and it
    survived the server-side fix. Asserted at source level because the dashboard
    has no test runner; the check is narrow on purpose — the defaults object must
    be all-null, whatever it is called.
    """
    store = config.REPO_ROOT / "frontend" / "src" / "store" / "useStore.ts"
    if not store.exists():
        pytest.skip("frontend store not present")
    src = store.read_text()
    assert "DEFAULT_WAVEFORM" not in src, \
        "the fabricating waveform defaults are back in useStore.ts"
    m = re.search(r"const\s+NO_WAVEFORM\s*=\s*\{(.*?)\}", src, re.S)
    assert m, "useStore.ts has no NO_WAVEFORM defaults object"
    body = m.group(1)
    for field in ["ecgHRV", "ecgArr", "pleth", "rrv", "breathReg", "asynchrony"]:
        assert re.search(rf"\b{field}\s*:\s*null\b", body), \
            f"{field} must default to null (not measured), got: {body.strip()}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn() if name != "test_missing_markers_are_both_treated_as_absent" \
                    else [fn(m) for m in (None, float("nan"))]
                print(f"PASS {name}")
            except (Skip, pytest.skip.Exception) as e:
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
