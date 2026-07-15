"""Config-driven data-quality gate for the processed MDP datasets.

Mirrors the IntelliLung reference repo's ``data_validation/`` pytest suite: a
single YAML contract (``backend/configs/data_validation.yaml``) drives checks for
required columns, NaNs, physiological ranges, valid splits/actions, and episode
structure. This is the data-side complement to the model-side ``verify_before_deploy``
gate — run it after any ``python -m backend.mdp.dataset`` rebuild.

If a track's parquet has not been built yet, its data tests SKIP with a clear
message (so the suite stays green on a fresh checkout); the contract/no-drift
test always runs because it needs no data.

Run:  pytest backend/tests/test_data_validation.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from backend.pipeline import config

_CONTRACT = config.REPO_ROOT / "backend" / "configs" / "data_validation.yaml"


def _load_contract() -> dict:
    return yaml.safe_load(_CONTRACT.read_text())


CONTRACT = _load_contract()


def _parquet_path(track: str):
    name = CONTRACT[f"dataset_track_{track}"]
    return config.PROCESSED_PATH / name


def _load_track(track: str) -> pd.DataFrame:
    path = _parquet_path(track)
    if not path.exists():
        pytest.skip(f"{path.name} not built yet — run `python -m backend.mdp.dataset`.")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Contract integrity (no data required) — guards against silent drift.
# --------------------------------------------------------------------------- #
def test_feature_ranges_match_pipeline_clip_ranges():
    """The YAML ranges must equal the pipeline's actual clip ranges."""
    expected = {k: [float(lo), float(hi)]
                for k, (lo, hi) in {**config.CHART_CLIP_RANGES,
                                    **config.LAB_CLIP_RANGES}.items()}
    declared = {k: [float(v[0]), float(v[1])]
                for k, v in CONTRACT["feature_ranges"].items()}
    assert declared == expected, (
        "data_validation.yaml feature_ranges drifted from config.*_CLIP_RANGES; "
        f"declared={declared} expected={expected}")


def test_contract_covers_all_state_features():
    for feat in config.TABULAR_FEATURES:
        assert feat in CONTRACT["feature_ranges"], f"no range declared for {feat}"


# --------------------------------------------------------------------------- #
# Per-track data checks.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("track", ["a", "b"])
def test_required_columns_present(track):
    df = _load_track(track)
    missing = [c for c in CONTRACT["required_columns"] if c not in df.columns]
    assert not missing, f"track {track}: missing required columns {missing}"


@pytest.mark.parametrize("track", ["a", "b"])
def test_no_nulls_in_required_columns(track):
    df = _load_track(track)
    for col in CONTRACT["non_null_columns"]:
        if col in df.columns:
            n = int(df[col].isna().sum())
            assert n == 0, f"track {track}: {n} nulls in required column {col!r}"


@pytest.mark.parametrize("track", ["a", "b"])
def test_feature_values_within_physiological_range(track):
    df = _load_track(track)
    tol = float(CONTRACT.get("range_violation_tolerance", 0.0))
    problems = []
    for feat, (lo, hi) in CONTRACT["feature_ranges"].items():
        for prefix in ("s_", "ns_"):
            col = f"{prefix}{feat}"
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=float)
            oob = np.mean((vals < lo) | (vals > hi)) if len(vals) else 0.0
            if oob > tol:
                problems.append(f"{col}: {oob:.4%} out of [{lo}, {hi}]")
    assert not problems, f"track {track}: range violations: {problems}"


@pytest.mark.parametrize("track", ["a", "b"])
def test_splits_are_valid(track):
    df = _load_track(track)
    bad = set(df["split"].unique()) - set(CONTRACT["valid_splits"])
    assert not bad, f"track {track}: unexpected split labels {bad}"


@pytest.mark.parametrize("track", ["a", "b"])
def test_no_stay_leaks_across_splits(track):
    """Each stay_id must live in exactly one split (no patient-level leakage)."""
    df = _load_track(track)
    per_stay = df.groupby("stay_id")["split"].nunique()
    leaked = per_stay[per_stay > 1]
    assert leaked.empty, f"track {track}: {len(leaked)} stays span multiple splits"


@pytest.mark.parametrize("track", ["a", "b"])
def test_actions_valid(track):
    df = _load_track(track)
    if CONTRACT.get("require_nonnegative_actions", True):
        assert int((df["action"] < 0).sum()) == 0, f"track {track}: negative actions"


@pytest.mark.parametrize("track", ["a", "b"])
def test_rewards_finite(track):
    df = _load_track(track)
    if CONTRACT.get("require_finite_rewards", True):
        assert np.isfinite(df["reward"].to_numpy(dtype=float)).all(), \
            f"track {track}: non-finite rewards present"


@pytest.mark.parametrize("track", ["a", "b"])
def test_weight_kg_plausible(track):
    df = _load_track(track)
    lo, hi = CONTRACT["weight_kg_range"]
    w = df["weight_kg"].to_numpy(dtype=float)
    assert (w >= lo).all() and (w <= hi).all(), \
        f"track {track}: weight_kg outside [{lo}, {hi}]"


@pytest.mark.parametrize("track", ["a", "b"])
def test_episode_structure(track):
    df = _load_track(track)
    min_tx = int(CONTRACT["min_transitions_per_episode"])
    max_h = int(CONTRACT["max_episode_hours"])
    sizes = df.groupby("stay_id").size()
    assert int(sizes.min()) >= min_tx, \
        f"track {track}: an episode has < {min_tx} transitions"
    # episode span (hours) must respect the truncation limit
    span = df.groupby("stay_id")["hour"].agg(lambda h: h.max() - h.min())
    assert int(span.max()) <= max_h, \
        f"track {track}: episode span {int(span.max())}h exceeds {max_h}h limit"


@pytest.mark.parametrize("track", ["a", "b"])
def test_exactly_one_terminal_per_episode(track):
    df = _load_track(track)
    dones = df.groupby("stay_id")["done"].sum()
    bad = dones[dones != 1]
    assert bad.empty, f"track {track}: {len(bad)} episodes without exactly one done"
