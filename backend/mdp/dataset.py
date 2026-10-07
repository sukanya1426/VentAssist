"""Assemble the final MDP transition datasets (Section 5.3).

For each episode, consecutive hourly states form transitions:
    (state_t, action_t, reward_t, state_{t+1}, done)
where action_t is inferred from the ΔPEEP/ΔTV/ΔFiO₂ change (action_space.encode_action)
and reward_t is computed on RAW physiological values (reward.tier1/tier2_reward).

TIER 1 (12-dim) is always built. TIER 2 (18-dim) is built only when
``waveform_features.parquet`` is present; per Section 4.9, TIER 2 transitions keep
only rows where both the tabular AND all 6 waveform features are available (no
imputation at training time).

Storage: flat parquet with columns
    stay_id, hour, split, weight_kg, propensity_z, action, reward, done,
    s_<feat>… , ns_<feat>…           (feat order = canonical state order)

Outputs:
    data/processed/mdp_tier1.parquet
    data/processed/mdp_tier2.parquet         (only if waveform features exist)
    models/action_map.json
    (train_val_test_split.json already written by state_builder)

Run:
    python -m backend.mdp.dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from backend.mdp import action_space, normaliser, reward
from backend.pipeline import config, splits
from backend.pipeline.logging_utils import get_logger

log = get_logger("mdp_dataset")

TABULAR = config.TABULAR_FEATURES                       # 12
WAVEFORM = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
            "RRV", "Breathing_Regularity", "Asynchrony_Score"]  # 6
WAVEFORM_PARQUET = config.PROCESSED_PATH / "waveform_features.parquet"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fill_states(states: pd.DataFrame, feats: list[str],
                 train_ids: set[int]) -> pd.DataFrame:
    """Within-episode ffill/bfill, then train-median fill for any remainder."""
    states = states.sort_values(["stay_id", "hour"]).copy()
    states[feats] = (states.groupby("stay_id")[feats]
                     .transform(lambda g: g.ffill().bfill()))
    train_med = (states[states["stay_id"].isin(train_ids)][feats].median()
                 if train_ids else states[feats].median())
    states[feats] = states[feats].fillna(train_med)
    # any feature entirely empty even in train → 0
    states[feats] = states[feats].fillna(0.0)
    return states


def _split_of(stay_id: int, split: dict[str, set]) -> str:
    if stay_id in split["train"]:
        return "train"
    if stay_id in split["val"]:
        return "val"
    return "test"


def _track_cfg(track: str) -> dict:
    cfg_name = "track_a_config.yaml" if track == "a" else "track_b_config.yaml"
    return yaml.safe_load((config.REPO_ROOT / "backend" / "configs" / cfg_name).read_text())


def _reward_cfg(track: str) -> dict:
    return _track_cfg(track).get("reward", {}) or {}


def _inherit_tabular_norm(stats: dict, track: str = "b") -> dict:
    """Reuse the source policy's tabular normalisation when warm-starting.

    A warm-started policy inherits the source's Q-function, which was fitted on
    states z-scored with the SOURCE normaliser. Normalising the same raw features
    with this track's own mean/std would feed that Q a different scale: on the
    37-stay waveform cohort the shared features drift up to ~0.5 sigma in the mean
    (FiO2) and ~25% in the std, so ``warm_start_from``'s exact [I | 0] identity
    would hold in latent space while silently breaking on raw inputs — the
    inherited Q would be evaluated off its training distribution.

    So when ``init_from`` is set, the shared leading features keep the source's
    statistics and only the track-specific extra features get new ones. This is
    also what makes the 12-vs-18 ablation honest: both arms then see identical
    z-values for the 12 clinical dims, and the only difference left is the
    waveform dims themselves.
    """
    init_from = _track_cfg(track).get("init_from")
    if not init_from:
        return stats
    src = config.MODEL_PATH / "normaliser_stats.json"
    if not src.exists():
        log.warning("init_from=%s but %s is missing — keeping this track's own "
                    "tabular normalisation (the warm start will be inexact).",
                    init_from, src.name)
        return stats
    src_stats = normaliser.load(src)
    shared = [f for f in TABULAR if f in src_stats]
    stats = {**stats, **{f: src_stats[f] for f in shared}}
    log.info("Inherited normalisation for %d shared tabular features from %s "
             "(warm start → the inherited Q sees its own z-scale).",
             len(shared), src.name)
    return stats


def _reward_lam_causal(track: str) -> float:
    """Read the action-causal bonus weight from the track config (Section 14.4)."""
    return float(_reward_cfg(track).get("lam_causal", reward.LAM_CAUSAL))


def _reward_w_outcome(track: str) -> float:
    """Terminal outcome-reward weight (mortality + VFD). 0.0 = disabled (default)."""
    return float(_reward_cfg(track).get("w_outcome", 0.0))


def _outcome_horizon(track: str) -> float:
    return float(_reward_cfg(track).get("outcome_horizon_days", 28.0))


def _load_outcomes() -> dict[int, tuple[bool, float]]:
    """stay_id → (died_within_horizon, vent_days); empty if outcomes.csv absent."""
    path = config.PROCESSED_PATH / "outcomes.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    # Only stays with a reliable outcome label contribute a terminal reward;
    # unknown-label stays (degraded mode) are omitted → neutral 0.0 downstream.
    if "outcome_label_known" in df.columns:
        df = df[df["outcome_label_known"] == 1]
    return {int(r.stay_id): (bool(r.died_horizon), float(r.vent_days))
            for r in df.itertuples()}


def recombined_reward(df: pd.DataFrame, track: str) -> np.ndarray:
    """Reward = reward_base + lam_causal·causal_unit + w_outcome·outcome_unit.

    Keeps the per-step causal bonus (§14.4) and the terminal outcome reward
    sweepable via config without rebuilding the dataset. Falls back to the
    materialised ``reward`` column for legacy datasets without the split columns.
    """
    if "reward_base" in df.columns and "causal_unit" in df.columns:
        r = df["reward_base"] + _reward_lam_causal(track) * df["causal_unit"]
        if "outcome_unit" in df.columns:
            r = r + _reward_w_outcome(track) * df["outcome_unit"]
        return r.to_numpy(np.float32)
    return df["reward"].to_numpy(np.float32)


def _build_transitions(states: pd.DataFrame, feats: list[str], track: str,
                       cohort: pd.DataFrame, prop: pd.DataFrame,
                       split: dict[str, set],
                       outcomes: dict[int, tuple[bool, float]] | None = None) -> pd.DataFrame:
    """Build per-transition rows for a given track ('a' tabular / 'b' waveform)."""
    weight = cohort.set_index("stay_id")["weight_kg"].to_dict()
    zscore = (prop.set_index("stay_id")["propensity_z"].to_dict()
              if prop is not None and not prop.empty else {})
    reward_fn = reward.tier1_reward if track == "a" else reward.tier2_reward
    lam_causal = _reward_lam_causal(track)
    w_outcome = _reward_w_outcome(track)
    outcomes = outcomes or {}
    horizon = _outcome_horizon(track)

    rows: list[dict] = []
    for sid, g in states.groupby("stay_id"):
        g = g.sort_values("hour").reset_index(drop=True)
        hours = g["hour"].to_numpy()
        w = float(weight.get(sid, 80.0))
        z = float(zscore.get(sid, 0.5))
        # UNIT terminal outcome term (scaled by config w_outcome in load_mdp).
        # No outcome label for this stay → neutral 0.0 (NOT a survivor's +reward).
        if int(sid) in outcomes:
            died, vent_days = outcomes[int(sid)]
            outcome_unit_terminal = reward.outcome_reward(died, vent_days, horizon_days=horizon)
        else:
            outcome_unit_terminal = 0.0

        # Collect this episode's transitions first; the terminal flag and outcome
        # reward are assigned to the LAST emitted transition afterwards. (Keying
        # ``done`` off ``hour == last_hour`` mis-fires on duplicate hours or when
        # the final transition is gap-dropped — see test_data_validation.)
        stay_rows: list[dict] = []
        for i in range(len(g) - 1):
            if hours[i + 1] - hours[i] > config.VENT_MERGE_GAP_HOURS:
                continue  # gap too large — drop transition
            s = g.loc[i, feats]
            ns = g.loc[i + 1, feats]
            s_dict = {f: float(s[f]) for f in feats}
            ns_dict = {f: float(ns[f]) for f in feats}
            a = action_space.encode_action(ns_dict["PEEP"] - s_dict["PEEP"],
                                            ns_dict["TV"] - s_dict["TV"],
                                            ns_dict["FiO2"] - s_dict["FiO2"])
            atuple = action_space.decode_action(a)
            # Store the base reward (no causal bonus) and a UNIT causal bonus
            # separately so lam_causal can be swept at train time (load_mdp
            # recombines them) without rebuilding the dataset. The materialised
            # ``reward`` uses the config lam_causal for direct/legacy consumers.
            r_base = reward_fn(s_dict, ns_dict, atuple, w, lam_causal=0.0)
            causal_unit = reward.action_causal_bonus(s_dict, atuple, w, lam_causal=1.0)
            row = {
                "stay_id": int(sid), "hour": int(hours[i]),
                "split": _split_of(int(sid), split),
                "weight_kg": w, "propensity_z": z,
                "action": int(a),
                "reward": float(r_base + lam_causal * causal_unit),
                "reward_base": float(r_base), "causal_unit": float(causal_unit),
                "outcome_unit": 0.0,
                "done": False,
            }
            row.update({f"s_{f}": s_dict[f] for f in feats})
            row.update({f"ns_{f}": ns_dict[f] for f in feats})
            stay_rows.append(row)

        if stay_rows:
            # Mark exactly one terminal per episode and attach the outcome reward.
            term = stay_rows[-1]
            term["done"] = True
            term["outcome_unit"] = float(outcome_unit_terminal)
            term["reward"] = float(term["reward"] + w_outcome * outcome_unit_terminal)
            rows.extend(stay_rows)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main builders
# --------------------------------------------------------------------------- #
def build_track_a() -> pd.DataFrame:
    """Track A — clinical 12-dim MDP over the full cohort."""
    config.ensure_output_dirs()
    states = pd.read_parquet(config.PROCESSED_PATH / "tabular_states.parquet")
    cohort = pd.read_csv(config.PROCESSED_PATH / "cohort.csv")
    prop = pd.read_csv(config.PROCESSED_PATH / "propensity_scores.csv")
    split = {k: set(v) for k, v in splits.load_split().items()}

    states = _fill_states(states, TABULAR, split["train"])
    tx = _build_transitions(states, TABULAR, track="a",
                            cohort=cohort, prop=prop, split=split,
                            outcomes=_load_outcomes())
    if tx.empty:
        raise ValueError("No Track A transitions produced.")

    train_mask = (tx["split"] == "train").to_numpy()
    tx["action"], valid = action_space.remap_rare(tx["action"], train_mask, min_count=10)
    action_space.save_action_map(valid)

    out = config.PROCESSED_PATH / "mdp_track_a.parquet"
    tx.to_parquet(out, index=False)
    _log_summary("TRACK_A", tx, valid, out)
    return tx


def build_track_b() -> pd.DataFrame | None:
    """Track B — waveform-anchored 18-dim MDP (proof-of-concept scale)."""
    wave_path = config.PROCESSED_PATH / "waveform_features_track_b.parquet"
    states_path = config.PROCESSED_PATH / "tabular_states_track_b.parquet"
    if not wave_path.exists() or not states_path.exists():
        log.warning("Track B inputs missing (%s / %s) — skipping.",
                    states_path.name, wave_path.name)
        return None

    states = pd.read_parquet(states_path)
    wave = pd.read_parquet(wave_path)
    cohort = pd.read_csv(config.PROCESSED_PATH / "cohort_track_b.csv")
    split = {k: set(v) for k, v in
             splits.load_split("train_val_test_split_track_b.json").items()}

    states = _fill_states(states, TABULAR, split["train"])
    merged = states.merge(wave, on=["stay_id", "hour"], how="inner")
    merged = merged.dropna(subset=WAVEFORM)   # keep only complete-waveform rows
    feats18 = TABULAR + WAVEFORM
    tx = _build_transitions(merged, feats18, track="b",
                            cohort=cohort, prop=None, split=split,
                            outcomes=_load_outcomes())
    if tx.empty:
        log.warning("No complete Track B transitions after waveform filtering.")
        return None

    train_mask = (tx["split"] == "train").to_numpy()
    tx["action"], valid = action_space.remap_rare(tx["action"], train_mask, min_count=2)

    # extend Track B normaliser with waveform-feature stats (train only)
    norm_path = config.MODEL_PATH / "normaliser_stats_track_b.json"
    stats = normaliser.load(norm_path) if norm_path.exists() else {}
    wtrain = tx[tx["split"] == "train"][[f"s_{f}" for f in WAVEFORM]]
    if wtrain.empty:
        wtrain = tx[[f"s_{f}" for f in WAVEFORM]]
    wtrain.columns = WAVEFORM
    stats.update(normaliser.fit(wtrain, WAVEFORM))
    stats = _inherit_tabular_norm(stats, track="b")
    normaliser.save(stats, norm_path)

    out = config.PROCESSED_PATH / "mdp_track_b.parquet"
    tx.to_parquet(out, index=False)
    _log_summary("TRACK_B", tx, valid, out)
    return tx


def _log_summary(tag: str, tx: pd.DataFrame, valid: set[int], out) -> None:
    counts = tx["split"].value_counts().to_dict()
    log.info("%s: %d transitions (%s), %d stays, %d valid actions → %s",
             tag, len(tx), counts, tx["stay_id"].nunique(), len(valid), out)
    log.info("  reward: mean=%.4f min=%.4f max=%.4f | done rate=%.3f",
             tx["reward"].mean(), tx["reward"].min(), tx["reward"].max(),
             tx["done"].mean())
    top = tx["action"].value_counts().head(5).to_dict()
    log.info("  top actions %s", {int(k): int(v) for k, v in top.items()})


def load_mdp(track: str = "a") -> dict:
    """Load an MDP parquet into arrays for the RL trainer ('a' or 'b').

    The reward is recombined as ``reward_base + lam_causal * causal_unit`` using
    the current config lam_causal, so the action-causal weight can be swept by
    editing the config and retraining — no dataset rebuild needed (§14.4).
    """
    path = config.PROCESSED_PATH / f"mdp_track_{track}.parquet"
    df = pd.read_parquet(path)
    feats = TABULAR if track == "a" else TABULAR + WAVEFORM
    return {
        "states": df[[f"s_{f}" for f in feats]].to_numpy(np.float32),
        "next_states": df[[f"ns_{f}" for f in feats]].to_numpy(np.float32),
        "actions": df["action"].to_numpy(np.int64),
        "rewards": recombined_reward(df, track),
        "dones": df["done"].to_numpy(bool),
        "split": df["split"].to_numpy(),
        "stay_id": df["stay_id"].to_numpy(np.int64),
        "hour": df["hour"].to_numpy(np.int64) if "hour" in df.columns else np.zeros(len(df), np.int64),
        "weight_kg": df["weight_kg"].to_numpy(np.float32),
        "propensity_z": df["propensity_z"].to_numpy(np.float32),
        "feature_order": feats,
    }


def main() -> None:
    build_track_a()
    build_track_b()


if __name__ == "__main__":
    main()
