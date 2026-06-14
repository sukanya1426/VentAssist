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


def _build_transitions(states: pd.DataFrame, feats: list[str], track: str,
                       cohort: pd.DataFrame, prop: pd.DataFrame,
                       split: dict[str, set]) -> pd.DataFrame:
    """Build per-transition rows for a given track ('a' tabular / 'b' waveform)."""
    weight = cohort.set_index("stay_id")["weight_kg"].to_dict()
    zscore = (prop.set_index("stay_id")["propensity_z"].to_dict()
              if prop is not None and not prop.empty else {})
    reward_fn = reward.tier1_reward if track == "a" else reward.tier2_reward

    rows: list[dict] = []
    for sid, g in states.groupby("stay_id"):
        g = g.sort_values("hour").reset_index(drop=True)
        hours = g["hour"].to_numpy()
        w = float(weight.get(sid, 80.0))
        z = float(zscore.get(sid, 0.5))
        last_hour = hours.max()
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
            r = reward_fn(s_dict, ns_dict, w)
            row = {
                "stay_id": int(sid), "hour": int(hours[i]),
                "split": _split_of(int(sid), split),
                "weight_kg": w, "propensity_z": z,
                "action": int(a), "reward": float(r),
                "done": bool(hours[i + 1] == last_hour),
            }
            row.update({f"s_{f}": s_dict[f] for f in feats})
            row.update({f"ns_{f}": ns_dict[f] for f in feats})
            rows.append(row)
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
                            cohort=cohort, prop=prop, split=split)
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
                            cohort=cohort, prop=None, split=split)
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
    """Load an MDP parquet into arrays for the RL trainer ('a' or 'b')."""
    path = config.PROCESSED_PATH / f"mdp_track_{track}.parquet"
    df = pd.read_parquet(path)
    feats = TABULAR if track == "a" else TABULAR + WAVEFORM
    return {
        "states": df[[f"s_{f}" for f in feats]].to_numpy(np.float32),
        "next_states": df[[f"ns_{f}" for f in feats]].to_numpy(np.float32),
        "actions": df["action"].to_numpy(np.int64),
        "rewards": df["reward"].to_numpy(np.float32),
        "dones": df["done"].to_numpy(bool),
        "split": df["split"].to_numpy(),
        "stay_id": df["stay_id"].to_numpy(np.int64),
        "weight_kg": df["weight_kg"].to_numpy(np.float32),
        "propensity_z": df["propensity_z"].to_numpy(np.float32),
        "feature_order": feats,
    }


def main() -> None:
    build_track_a()
    build_track_b()


if __name__ == "__main__":
    main()
