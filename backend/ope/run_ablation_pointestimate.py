"""Track B waveform ablation as an honest, explicitly-underpowered point estimate.

Methodology §9 defines the defining waveform comparison as

    ΔV̂ = V̂(π_18-dim) − V̂(π_12-dim),  with CI_(1-α) excluding zero.

SYSTEM_SUMMARY §4/§14 document that this is **not achievable** with the ~102-
transition / ~15-stay Track B proof-of-concept cohort: there is nowhere near the
power to exclude zero. This script does NOT pretend otherwise. It reports ΔV̂ as a
point estimate with both arms' bootstrap CIs, sets ``is_underpowered: true`` and
records ``n_test_episodes`` so any downstream consumer (dashboard / report) can
render the caveat automatically instead of relying on a human to remember it.

Both arms are evaluated on the *same* Track B test episodes:
  * 18-dim arm: the Track B policy over the full 12 clinical + 6 waveform dims.
  * 12-dim arm: the Track A policy over the 12 clinical dims of the same states.
Each state's value is the policy's greedy Q-value V(s) = max_a Q(s, a) under its own
trained Q-network (a model-consistent point estimate; a full FQE refit on a handful
of test transitions would itself be meaningless).

Run:
    python -m backend.ope.run_ablation_pointestimate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl.hybrid_iql import HybridIQL

log = get_logger("ope_ablation")

TABULAR = config.TABULAR_FEATURES


def _load_policy(track: str) -> tuple[HybridIQL, dict]:
    """Return the policy and the provenance needed to interpret ΔV̂.

    ``n_transitions`` alone is misleading for a warm-started policy: it counts
    only the fine-tuning transitions, while the Q-function it inherited was fitted
    on the source policy's far larger cohort. ``effective_n`` is what the
    comparison actually turns on.
    """
    ckpt = torch.load(config.MODEL_PATH / f"policy_track_{track}.pt",
                      map_location="cpu", weights_only=False)
    m = HybridIQL(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"],
                  hidden_dim=ckpt["hidden_dim"])
    m.load_state_dict(ckpt["state_dict"])
    n = ckpt.get("n_transitions")
    inherited = ckpt.get("init_from_n_transitions")
    return m, {
        "n_transitions": n,
        "init_from": ckpt.get("init_from"),
        "init_from_n_transitions": inherited,
        # A warm-started arm inherits the source's fit, so the data its Q-function
        # has seen is the source cohort plus its own fine-tuning set.
        "effective_n": (n or 0) + (inherited or 0),
        "lam_causal": ckpt.get("lam_causal"),
        "w_outcome": (ckpt.get("config_snapshot") or {})
                     .get("reward", {}).get("w_outcome"),
    }


def _greedy_values(policy: HybridIQL, states_norm: np.ndarray) -> np.ndarray:
    """V(s) = max_a Q(s, a) for each state under the policy's trained Q-network."""
    return np.array([policy.q_values(s).max() for s in states_norm], dtype=float)


# Below this many held-out episodes the bootstrap CI is not worth reading, whatever
# it says: the resample is drawing from a handful of patients, not a population.
MIN_POWERED_EPISODES = 10
# Ratio of train sizes beyond which the two arms are no longer comparable as a
# clean test of "do the 6 waveform dims add information".
MAX_TRAIN_SIZE_RATIO = 2.0


def _train_size_confounded(p18: dict, p12: dict) -> bool:
    """Do the two arms' Q-functions rest on comparable amounts of data?

    Compares EFFECTIVE training size, so a warm-started 18-dim arm (which inherits
    the 12-dim arm's full-cohort fit and then fine-tunes) counts as comparable —
    that is precisely what makes the comparison nested rather than confounded.
    """
    n18, n12 = p18.get("effective_n"), p12.get("effective_n")
    if not n18 or not n12:
        return True          # unknown train sizes → assume the worst
    lo, hi = sorted((n18, n12))
    return (hi / lo) > MAX_TRAIN_SIZE_RATIO


def _reward_mismatched(p18: dict, p12: dict) -> bool:
    """Were the arms trained under the same reward?

    ΔV̂ compares each arm's own ``max_a Q(s,a)``, which is denominated in its
    training reward. Different ``lam_causal``/``w_outcome`` therefore makes the
    difference meaningless regardless of sample size — Track B shipped 0.4/0.0
    against Track A's 2.0/0.25 until this was synced.
    """
    for key in ("lam_causal", "w_outcome"):
        a, b = p18.get(key), p12.get(key)
        if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
            return True
    return False


def _limitation(n_episodes: int, n_transitions: int,
                p18: dict, p12: dict) -> str:
    """State the binding limitation of THIS run, in its own numbers.

    This string used to be a hardcoded "~15 stays / ~102 transitions" sentence. Once
    the Track B cohort was rebuilt (2026-10-02) that was simply false, and a stale
    caveat is worse than none — a reader who checks it against n_test_episodes stops
    trusting the rest of the artifact. It is generated now.

    Note which limitation binds. Low power was the whole story at ~7 test
    transitions. With a real held-out set the dominant problem becomes the
    TRAIN-SIZE CONFOUND: the 12-dim arm is the deployed Track A policy fitted on the
    full ~694k-transition cohort, while the 18-dim arm is fitted only on the
    waveform-overlapping subset. The two arms therefore differ in training data
    volume as well as in state dimensionality, so the sign of ΔV̂ does NOT isolate
    the waveform features' contribution — a negative ΔV̂ is the expected result of
    fitting 125 actions on ~600 transitions, not evidence that waveforms are
    uninformative. Removing the confound needs both arms to share a Q-function
    trained on the same data (warm-start the 18-dim arm from Track A and let only
    the state adapter differ), which is a training change, not an evaluation one.
    """
    parts = [f"Held out {n_episodes} episodes / {n_transitions} transitions."]
    if n_episodes < MIN_POWERED_EPISODES:
        parts.append(
            f"UNDERPOWERED: fewer than {MIN_POWERED_EPISODES} held-out episodes, so "
            "the bootstrap CI understates the true uncertainty and NO significance "
            "is claimed.")
    if _reward_mismatched(p18, p12):
        parts.append(
            f"REWARD MISMATCH: 18-dim arm trained with lam_causal="
            f"{p18.get('lam_causal')}/w_outcome={p18.get('w_outcome')} vs the 12-dim "
            f"arm's {p12.get('lam_causal')}/{p12.get('w_outcome')}. ΔV̂ compares each "
            "arm's own max_a Q(s,a), which is denominated in its training reward, so "
            "this difference is not interpretable at all until the rewards match.")
    if _train_size_confounded(p18, p12):
        parts.append(
            f"CONFOUNDED BY TRAIN SIZE: the 18-dim arm's Q-function rests on "
            f"{p18.get('effective_n')} transitions and the 12-dim arm's on "
            f"{p12.get('effective_n')}. The arms differ in training data volume as "
            "well as state dimensionality, so the SIGN of ΔV̂ does not isolate the "
            "waveform features' contribution; a negative ΔV̂ is the expected "
            "consequence of the smaller training set. Warm-start the 18-dim arm from "
            "the 12-dim policy to make this a nested comparison.")
    elif p18.get("init_from"):
        parts.append(
            f"NESTED comparison: the 18-dim arm was warm-started from "
            f"{p18['init_from']} (inheriting a Q-function fitted on "
            f"{p18.get('init_from_n_transitions')} transitions) and fine-tuned on "
            f"{p18.get('n_transitions')}, with the waveform dims initialised to zero "
            "influence. Both arms therefore rest on the same base fit and the same "
            "reward, so ΔV̂ reflects the 6 extra state dimensions. It remains an "
            "on-policy greedy-value point estimate, not an FQE.")
    parts.append("See SYSTEM_SUMMARY §4, §14.")
    return " ".join(parts)


def _boot_ci(vals: np.ndarray, B: int = 2000, seed: int = 0) -> list[float]:
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                     for _ in range(B)])
    return [round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4)]


def run(track_b: str = "b", track_a: str = "a") -> dict:
    d = D.load_mdp(track_b)                      # raw 18-dim states (12 tab + 6 wave)
    feats = d["feature_order"]
    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(d["actions"]))
    n_test_episodes = int(np.unique(d["stay_id"][test]).size)

    # 18-dim arm: Track B policy, Track B normaliser, full 18 dims.
    stats_b = N.load(config.MODEL_PATH / "normaliser_stats_track_b.json")
    S18 = N.transform(d["states"], stats_b, feats).astype(np.float32)[test]
    pol_b, prov18 = _load_policy(track_b)
    v18 = _greedy_values(pol_b, S18)

    # 12-dim arm: Track A policy, Track A normaliser, the same states' tabular dims.
    stats_a = N.load(config.MODEL_PATH / "normaliser_stats.json")
    raw_tab = d["states"][test][:, :len(TABULAR)]
    S12 = N.transform(raw_tab, stats_a, TABULAR).astype(np.float32)
    pol_a, prov12 = _load_policy(track_a)
    v12 = _greedy_values(pol_a, S12)

    delta_v = float(v18.mean() - v12.mean())
    result = {
        "comparison": "V_hat(pi_18dim) - V_hat(pi_12dim) on Track B test episodes",
        "v_hat_18dim": round(float(v18.mean()), 4),
        "v_hat_18dim_CI95": _boot_ci(v18),
        "v_hat_12dim": round(float(v12.mean()), 4),
        "v_hat_12dim_CI95": _boot_ci(v12),
        "delta_v": round(delta_v, 4),
        "delta_v_CI95": _boot_ci(v18 - v12) if len(v18) == len(v12) else [float("nan")] * 2,
        "n_test_transitions": int(len(test)),
        "n_test_episodes": n_test_episodes,
        "arm_18dim": prov18,
        "arm_12dim": prov12,
        "is_underpowered": n_test_episodes < MIN_POWERED_EPISODES,
        "is_confounded_by_train_size": _train_size_confounded(prov18, prov12),
        "is_reward_mismatched": _reward_mismatched(prov18, prov12),
        "is_nested": bool(prov18.get("init_from"))
                     and not _train_size_confounded(prov18, prov12)
                     and not _reward_mismatched(prov18, prov12),
        "power_note": _limitation(n_test_episodes, int(len(test)),
                                  prov18, prov12),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = config.LOGS_PATH / "ablation_track_b.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("Ablation (n_test_episodes=%d, underpowered=%s, train-size-confounded=%s): ΔV̂=%.4f "
             "[18d=%.4f vs 12d=%.4f] → %s",
             n_test_episodes, result["is_underpowered"],
             result["is_confounded_by_train_size"], delta_v,
             result["v_hat_18dim"], result["v_hat_12dim"], out)
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
