"""Time-varying propensity z_t (Methodology §10.4) — opt-in OPE ablation.

The static propensity ``propensity_z`` (backend/pipeline/propensity_score.py) is a
single per-patient 90-day-mortality risk, constant across a stay. §10.4 also asks
for a **time-varying** propensity

    z_t = P(adverse event in the next 6 h | recent features, RASS, FiO₂),

a logistic model refit through time, so the NW OPE kernel can condition on the
patient's *current* risk rather than a stay-level constant.

Adverse-event definition (chosen & justified)
---------------------------------------------
**SpO₂ < 88 % at any point within the next 6 hours.** Rationale:

  * It is an acute, clinically meaningful *deterioration in oxygenation* — exactly
    the risk a ventilator PEEP/FiO₂ decision is trying to avert — so conditioning
    the transition kernel on it targets the confounding §10.4 cares about.
  * It is observable for **every** transition directly from the tabular state
    (SpO₂ is one of the 12 dims), so z_t is defined densely along each trajectory,
    unlike a mortality label which is one event per stay.
  * The 6-hour horizon matches the methodology's refit cadence.

(The alternative "death within 6 h" is far too rare per-transition to fit a useful
per-step logistic here — it is the *static* z's job over a 90-day horizon.)

Features: the current normalised 12-dim tabular state (which already includes RASS
and FiO₂ and the recent respiratory signals). We fit ONE logistic model on the
train split and predict z_t for every transition — a documented simplification of
"refit every 6 h" that is sufficient for the OPE ablation (the model is stationary
in *state*, not wall-clock, so a single fit over all hours is well-specified).

This module is **evaluation-only** and opt-in: it never touches the deployed
policy, the reward, or the default (static) OPE path.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("propensity_dynamic")

SPO2_ADVERSE = 88.0          # hypoxaemia threshold
HORIZON_HOURS = 6            # look-ahead window for the adverse event
_SPO2_I = config.TABULAR_FEATURES.index("SpO2")


def adverse_labels(d: dict, horizon: int = HORIZON_HOURS,
                   thresh: float = SPO2_ADVERSE) -> np.ndarray:
    """1 if SpO₂ < ``thresh`` at any hour in (h, h+horizon] within the same stay.

    Uses each transition's state SpO₂ plus each stay's terminal next-state SpO₂ so
    the final hour is covered."""
    sid, hour = d["stay_id"], d["hour"]
    spo2 = d["states"][:, _SPO2_I]
    nspo2 = d["next_states"][:, _SPO2_I]
    lut: dict[tuple[int, int], float] = {}
    for i in range(len(sid)):
        lut[(int(sid[i]), int(hour[i]))] = float(spo2[i])
    # terminal next-state SpO₂ at hour+1 (last transition per stay)
    for i in np.where(d["dones"])[0]:
        lut[(int(sid[i]), int(hour[i]) + 1)] = float(nspo2[i])

    y = np.zeros(len(sid), dtype=np.int64)
    for i in range(len(sid)):
        s, h = int(sid[i]), int(hour[i])
        for k in range(1, horizon + 1):
            v = lut.get((s, h + k))
            if v is not None and v < thresh:
                y[i] = 1
                break
    return y


def _features(d: dict) -> np.ndarray:
    """Normalised 12-dim tabular state (recent features incl. RASS + FiO₂)."""
    stats = N.load(config.MODEL_PATH /
                   ("normaliser_stats.json" if d["feature_order"][0] == "PEEP"
                    else "normaliser_stats_track_b.json"))
    return N.transform(d["states"], stats, d["feature_order"]).astype(np.float32)[:, :12]


def fit_dynamic(track: str = "a", seed: int = 0) -> dict:
    """Fit z_t on the train split, predict for all transitions, report held-out AUC.

    Returns ``{"z_t": (N,), "auc": float, "event_rate": float, ...}``; z_t is
    aligned with ``load_mdp(track)`` row order."""
    d = D.load_mdp(track)
    X = _features(d)
    y = adverse_labels(d)

    train = np.where(d["split"] == "train")[0]
    held = np.where(np.isin(d["split"], ["val", "test"]))[0]
    if len(train) < 50 or len(np.unique(y[train])) < 2:
        # degenerate (e.g. track B) — fall back to a constant risk = event rate
        z = np.full(len(y), float(y.mean()), dtype=np.float64)
        return {"z_t": z, "auc": float("nan"), "event_rate": float(y.mean()),
                "n_train": int(len(train)), "n_held": int(len(held)),
                "model": None}

    model = Pipeline([("scale", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                random_state=seed))])
    model.fit(X[train], y[train])
    z = model.predict_proba(X)[:, 1]

    auc = float("nan")
    if len(held) and len(np.unique(y[held])) == 2:
        auc = float(roc_auc_score(y[held], z[held]))
    log.info("Dynamic z_t Track %s: event-rate=%.3f held-out AUC=%.3f (n_train=%d)",
             track.upper(), float(y.mean()), auc, len(train))
    return {"z_t": z.astype(np.float64), "auc": auc, "event_rate": float(y.mean()),
            "n_train": int(len(train)), "n_held": int(len(held)), "model": model}


def compute_zt(track: str = "a") -> np.ndarray:
    """Convenience: per-transition z_t aligned with ``load_mdp(track)`` order."""
    return fit_dynamic(track)["z_t"]


def _dynamic_zt_enabled(track: str) -> bool:
    import yaml
    cfg_name = "track_a_config.yaml" if track == "a" else "track_b_config.yaml"
    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs" / cfg_name).read_text())
    return bool((cfg.get("propensity", {}) or {}).get("dynamic_zt", False))


def materialise_column(track: str = "a", force: bool = False) -> None:
    """OPT-IN: write z_t back to the MDP parquet as ``propensity_zt`` (a NEW column;
    ``propensity_z`` is left untouched). Gated behind config ``propensity.dynamic_zt``;
    not run by default so the deployed parquet / static path stay byte-for-byte
    identical. Does NOT rebuild states/rewards or retrain."""
    if not force and not _dynamic_zt_enabled(track):
        raise RuntimeError(
            "propensity.dynamic_zt is false — refusing to modify the MDP parquet. "
            "Set the config flag (or pass force=True) to materialise the column.")
    import pandas as pd
    path = config.PROCESSED_PATH / f"mdp_track_{track}.parquet"
    df = pd.read_parquet(path)
    df["propensity_zt"] = compute_zt(track)
    df.to_parquet(path, index=False)
    log.info("Wrote propensity_zt column (%d rows) → %s", len(df), path.name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fit/report time-varying propensity z_t.")
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--materialise", action="store_true",
                    help="write z_t as a new MDP column (opt-in; propensity.dynamic_zt)")
    args = ap.parse_args()
    r = fit_dynamic(args.track)
    print(f"event_rate={r['event_rate']:.3f} held-out AUC={r['auc']:.3f} "
          f"z_t∈[{r['z_t'].min():.3f}, {r['z_t'].max():.3f}]")
    if args.materialise:
        materialise_column(args.track)
