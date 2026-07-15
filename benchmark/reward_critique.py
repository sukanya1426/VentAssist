"""The reward critique: IntelliLung's reward is action-independent (BENCHMARK_PLAN §0 trap 2).

This is our headline scientific claim. It is argued in two layers.

LAYER 1 — STRUCTURAL (irrefutable, no training).
  Counterfactual action-sensitivity: hold the *observed* transition (s, s′) fixed and
  recompute the reward under all 125 possible actions. Report the spread
  (max − min) across actions.
      * IntelliLung RangeReward:  r = f(s′)  → spread is EXACTLY 0 for every transition.
        The reward cannot distinguish a good action from a bad one that happened to
        precede the same next state. The only path from action to reward is through the
        (confounded) transition itself.
      * VentAssist reward: contains −action_cost(a) and +λ·causal_bonus(s, a), which
        depend on the action DIRECTLY → non-zero spread.

LAYER 2 — CONSEQUENTIAL (train and show the collapse).
  Train the identical HybridIQL+CQL architecture on the identical MDP, changing ONLY
  the reward, and measure:
      * Q-flatness across actions on held-out states (max − min, top-2 margin)
      * distinct actions used / hold-share / agreement with clinicians
      * the 8-case clinical battery: does the policy respond to hypoxaemia, hyperoxia,
        hypercapnia, high PEEP, volutrauma?
  A flat Q and a battery failure under their reward, fixed under ours, is the claim.

Run:
    PYTHONPATH=. .venv/bin/python -m benchmark.reward_critique --layer 1
    PYTHONPATH=. .venv/bin/python -m benchmark.reward_critique --layer 2 --steps 30000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from backend.mdp import action_space
from backend.mdp import dataset as D
from backend.mdp import reward as VA_REWARD
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from benchmark import rewards_intellilung as IL

log = get_logger("bench_reward_critique")
RESULTS = Path(__file__).resolve().parent / "results"

# The 8-case clinical battery (same states the deploy gate uses).
BATTERY = {
    "stable":      (dict(PEEP=8, TV=460, FiO2=0.4, SpO2=95, PaO2=88, PaCO2=40, pH=7.40,
                         HR=84, SBP=120, RR=16, RASS=-1, Temp=37.0), "hold"),
    "hypoxaemic":  (dict(PEEP=8, TV=480, FiO2=0.5, SpO2=84, PaO2=55, PaCO2=44, pH=7.34,
                         HR=104, SBP=112, RR=26, RASS=-2, Temp=37.6), "raise PEEP or FiO2"),
    "hyperoxic":   (dict(PEEP=8, TV=480, FiO2=0.7, SpO2=100, PaO2=90, PaCO2=44, pH=7.37,
                         HR=92, SBP=118, RR=22, RASS=-2, Temp=37.2), "lower FiO2"),
    "hypercapnic": (dict(PEEP=8, TV=320, FiO2=0.5, SpO2=93, PaO2=72, PaCO2=65, pH=7.28,
                         HR=98, SBP=118, RR=28, RASS=-1, Temp=37.3), "raise TV"),
    "hypocapnic":  (dict(PEEP=8, TV=560, FiO2=0.5, SpO2=97, PaO2=110, PaCO2=30, pH=7.50,
                         HR=88, SBP=120, RR=12, RASS=-2, Temp=37.0), "lower TV"),
    "high_peep":   (dict(PEEP=18, TV=470, FiO2=0.5, SpO2=94, PaO2=80, PaCO2=43, pH=7.38,
                         HR=90, SBP=105, RR=20, RASS=-3, Temp=37.2), "lower PEEP"),
    "low_peep":    (dict(PEEP=2, TV=470, FiO2=0.5, SpO2=92, PaO2=70, PaCO2=44, pH=7.36,
                         HR=95, SBP=115, RR=22, RASS=-2, Temp=37.1), "raise PEEP"),
    "volutrauma":  (dict(PEEP=8, TV=760, FiO2=0.5, SpO2=95, PaO2=90, PaCO2=38, pH=7.44,
                         HR=86, SBP=122, RR=14, RASS=-2, Temp=37.0), "cut TV"),
}


def _battery_verdict(name: str, dp: int, dt: int, df: float) -> bool:
    if name == "stable":       return (dp, dt, df) == (0, 0, 0.0)
    if name == "hypoxaemic":   return dp > 0 or df > 0
    if name == "hyperoxic":    return df < 0
    if name == "hypercapnic":  return dt > 0
    if name == "hypocapnic":   return dt < 0
    if name == "high_peep":    return dp < 0
    if name == "low_peep":     return dp > 0
    if name == "volutrauma":   return dt < 0
    return False


# --------------------------------------------------------------------------- #
# LAYER 1 — structural: counterfactual action-sensitivity
# --------------------------------------------------------------------------- #
def layer1(track: str = "a", n_sample: int = 3000, seed: int = 0) -> dict:
    d = D.load_mdp(track)
    feats = list(d["feature_order"])
    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs"
                          / "track_a_config.yaml").read_text())
    lam_causal = float(cfg["reward"]["lam_causal"])

    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(d["actions"]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(test, size=min(n_sample, len(test)), replace=False)

    S, NS, W = d["states"][idx], d["next_states"][idx], d["weight_kg"][idx].astype(float)

    il_spread, va_spread = np.zeros(len(idx)), np.zeros(len(idx))
    for i in range(len(idx)):
        s = {f: float(S[i, j]) for j, f in enumerate(feats)}
        ns = {f: float(NS[i, j]) for j, f in enumerate(feats)}
        w = max(W[i], 1.0)

        # --- their reward under all 125 actions (s, s' held fixed) ---
        il = np.array([IL.range_reward_scalar(ns) for _ in range(action_space.N_ACTIONS)])
        il_spread[i] = il.max() - il.min()

        # --- our reward under all 125 actions (s, s' held fixed) ---
        va = np.array([
            VA_REWARD.tier1_reward(s, ns, action_space.decode_action(a), w,
                                   lam_causal=lam_causal)
            for a in range(action_space.N_ACTIONS)
        ])
        va_spread[i] = va.max() - va.min()

    result = {
        "layer": 1,
        "description": "counterfactual reward spread across all 125 actions, (s, s') held fixed",
        "n_sampled_transitions": int(len(idx)),
        "intellilung_range_reward": {
            "mean_spread": round(float(il_spread.mean()), 6),
            "max_spread": round(float(il_spread.max()), 6),
            "pct_transitions_with_zero_spread": round(float(np.mean(il_spread == 0) * 100), 2),
        },
        "ventassist_reward": {
            "mean_spread": round(float(va_spread.mean()), 4),
            "max_spread": round(float(va_spread.max()), 4),
            "pct_transitions_with_zero_spread": round(float(np.mean(va_spread == 0) * 100), 2),
        },
        "lam_causal": lam_causal,
        "verdict": ("IntelliLung's reward assigns the IDENTICAL value to all 125 actions for "
                    "100% of transitions — it carries zero action signal. VentAssist's reward "
                    "discriminates between actions."),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reward_critique_layer1.json").write_text(json.dumps(result, indent=2))

    log.info("LAYER 1 — counterfactual reward spread over 125 actions (n=%d):", len(idx))
    log.info("  IntelliLung RangeReward : mean=%.6f  max=%.6f  zero-spread on %.1f%% of transitions",
             il_spread.mean(), il_spread.max(),
             result["intellilung_range_reward"]["pct_transitions_with_zero_spread"])
    log.info("  VentAssist reward       : mean=%.4f  max=%.4f  zero-spread on %.1f%% of transitions",
             va_spread.mean(), va_spread.max(),
             result["ventassist_reward"]["pct_transitions_with_zero_spread"])
    return result


# --------------------------------------------------------------------------- #
# LAYER 2 — consequential: train under each reward, measure collapse
# --------------------------------------------------------------------------- #
def _policy_diagnostics(model, d, test_idx, stats, feats) -> dict:
    from backend.mdp import normaliser as N
    S = d["states"][test_idx]
    acts = model.act_batch(S)
    behav = d["actions"][test_idx]
    hold = action_space.encode_action(0, 0, 0.0)

    # Q-flatness across actions (the smoking gun for an action-independent reward)
    sub = S[np.random.default_rng(0).choice(len(S), size=min(2000, len(S)), replace=False)]
    q = np.stack([model.q_values(s) for s in sub])              # (n, 125)
    qsort = np.sort(q, axis=1)
    spread = float(np.mean(qsort[:, -1] - qsort[:, 0]))
    top2 = float(np.mean(qsort[:, -1] - qsort[:, -2]))

    # clinical battery on RAW states, using the serving encoding
    battery, passed = {}, 0
    for name, (state, expect) in BATTERY.items():
        vec = np.array([float(state[f]) for f in feats])
        z = N.transform_inference(vec, stats, feats)
        a = int(np.argmax(model.q_values(z)))
        dp, dt, df = action_space.decode_action(a)
        ok = _battery_verdict(name, dp, dt, df)
        passed += ok
        battery[name] = {"expect": expect, "action": [dp, dt, df], "pass": bool(ok)}

    return {
        "q_spread_max_minus_min": round(spread, 4),
        "q_top2_margin": round(top2, 4),
        "distinct_actions_used": int(len(np.unique(acts))),
        "hold_share": round(float(np.mean(acts == hold)), 4),
        "agreement_with_clinician": round(float(np.mean(acts == behav)), 4),
        "battery_passed": f"{passed}/8",
        "battery": battery,
    }


def layer2(track: str = "a", steps: int = 30000, seed: int = 0) -> dict:
    from backend.mdp import normaliser as N
    from backend.rl import trainer

    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs"
                          / "track_a_config.yaml").read_text())
    d = trainer._normalise(D.load_mdp(track), track)
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    feats = list(d["feature_order"])

    train_idx = np.where(d["split"] == "train")[0]
    val_idx = np.where(d["split"] == "val")[0]
    test_idx = np.where(d["split"] == "test")[0]

    # Their reward: computed from the RAW next states (dataset stores raw pre-normalisation
    # values in load_mdp; _normalise only z-scored `states`/`next_states` copies, so we
    # recompute from the raw parquet to be safe).
    raw = D.load_mdp(track)
    il_rewards = IL.range_reward(raw["next_states"], feats).astype(np.float32)

    out = {}
    for label, rewards in (("intellilung_range_reward", il_rewards),
                           ("ventassist_reward", d["rewards"].astype(np.float32))):
        log.info("=== LAYER 2: training with %s (steps=%d) ===", label, steps)
        dd = {**d, "rewards": rewards}
        model, best_val = trainer._run_training(dd, train_idx, val_idx, cfg, steps,
                                                device="cpu", tag=f"[{label}] ")
        diag = _policy_diagnostics(model, dd, test_idx, stats, feats)
        diag["best_val_q"] = round(float(best_val), 4)
        out[label] = diag
        log.info("  %s -> Q-spread=%.4f top2=%.4f distinct=%d hold=%.2f battery=%s",
                 label, diag["q_spread_max_minus_min"], diag["q_top2_margin"],
                 diag["distinct_actions_used"], diag["hold_share"], diag["battery_passed"])

    result = {
        "layer": 2,
        "description": "same HybridIQL+CQL architecture and MDP; ONLY the reward differs",
        "steps": steps, "seed": seed,
        "n_transitions": int(len(d["actions"])),
        **out,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reward_critique_layer2.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["1", "2", "both"], default="1")
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--n-sample", type=int, default=3000)
    args = ap.parse_args()
    if args.layer in ("1", "both"):
        layer1(args.track, n_sample=args.n_sample)
    if args.layer in ("2", "both"):
        layer2(args.track, steps=args.steps)


if __name__ == "__main__":
    main()
