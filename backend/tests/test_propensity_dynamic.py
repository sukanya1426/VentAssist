"""Tests for the time-varying propensity z_t (Methodology §10.4, Task C).

z_t = P(SpO₂ < 88 in the next 6 h | recent tabular state incl. RASS/FiO₂), a
logistic model used as an opt-in OPE-ablation confounder. Acceptance:

  * z_t ∈ [0, 1];
  * the model trains and predicts on a held-out slice with AUC > 0.5;
  * the static path is byte-for-byte unchanged when the flag is off (the MDP
    parquet gains no column and column-materialisation is refused).

Run:  python -m backend.tests.test_propensity_dynamic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from backend.pipeline import config, propensity_dynamic as PD


class Skip(Exception):
    """Raised to skip when the MDP is unavailable."""


# --------------------------------------------------------------------------- #
# Pure-function test of the adverse-event labelling (no data / model)
# --------------------------------------------------------------------------- #
def test_adverse_labels_lookahead():
    """SpO₂ dropping below 88 within the horizon flags the earlier transitions."""
    i = config.TABULAR_FEATURES.index("SpO2")
    def state(spo2):
        v = np.zeros(12); v[i] = spo2; return v
    # one stay, hours 0,1,2 with SpO₂ 98, 96, 85 (state) and terminal ns 90.
    states = np.stack([state(98), state(96), state(85)])
    next_states = np.stack([state(96), state(85), state(90)])
    d = {"stay_id": np.array([7, 7, 7]), "hour": np.array([0, 1, 2]),
         "states": states, "next_states": next_states,
         "dones": np.array([False, False, True])}
    y = PD.adverse_labels(d, horizon=6, thresh=88.0)
    assert y.tolist() == [1, 1, 0], f"unexpected labels {y.tolist()}"
    assert set(np.unique(y)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# Model tests (need the MDP)
# --------------------------------------------------------------------------- #
def _fit_or_skip():
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    return PD.fit_dynamic("a")


def test_zt_in_unit_interval():
    r = _fit_or_skip()
    z = r["z_t"]
    assert z.min() >= 0.0 - 1e-9 and z.max() <= 1.0 + 1e-9, \
        f"z_t out of [0,1]: [{z.min()}, {z.max()}]"
    assert np.all(np.isfinite(z))


def test_held_out_auc_above_half():
    r = _fit_or_skip()
    assert r["auc"] > 0.5, f"held-out AUC {r['auc']} not > 0.5"


# --------------------------------------------------------------------------- #
# Static path unchanged when the flag is off
# --------------------------------------------------------------------------- #
def test_config_flag_default_off():
    cfg = yaml.safe_load(
        (config.REPO_ROOT / "backend" / "configs" / "track_a_config.yaml").read_text())
    assert (cfg.get("propensity", {}) or {}).get("dynamic_zt", False) is False, \
        "propensity.dynamic_zt must default OFF so the static path is unchanged"


def test_static_mdp_has_no_dynamic_column():
    if not (config.PROCESSED_PATH / "mdp_track_a.parquet").exists():
        raise Skip("mdp_track_a.parquet not present.")
    cols = pd.read_parquet(config.PROCESSED_PATH / "mdp_track_a.parquet",
                           columns=None).columns
    assert "propensity_zt" not in cols, \
        "dynamic z_t column leaked into the MDP parquet — static path changed"


def test_materialise_refused_when_flag_off():
    if PD._dynamic_zt_enabled("a"):
        raise Skip("propensity.dynamic_zt is ON in config — guard not exercised.")
    try:
        PD.materialise_column("a")           # must refuse (flag off, force=False)
    except RuntimeError:
        return
    raise AssertionError("materialise_column must refuse when the flag is off")


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
