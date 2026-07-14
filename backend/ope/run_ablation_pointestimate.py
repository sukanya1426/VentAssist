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


def _load_policy(track: str) -> HybridIQL:
    ckpt = torch.load(config.MODEL_PATH / f"policy_track_{track}.pt", map_location="cpu")
    m = HybridIQL(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"],
                  hidden_dim=ckpt["hidden_dim"])
    m.load_state_dict(ckpt["state_dict"])
    return m


def _greedy_values(policy: HybridIQL, states_norm: np.ndarray) -> np.ndarray:
    """V(s) = max_a Q(s, a) for each state under the policy's trained Q-network."""
    return np.array([policy.q_values(s).max() for s in states_norm], dtype=float)


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
    pol_b = _load_policy(track_b)
    v18 = _greedy_values(pol_b, S18)

    # 12-dim arm: Track A policy, Track A normaliser, the same states' tabular dims.
    stats_a = N.load(config.MODEL_PATH / "normaliser_stats.json")
    raw_tab = d["states"][test][:, :len(TABULAR)]
    S12 = N.transform(raw_tab, stats_a, TABULAR).astype(np.float32)
    pol_a = _load_policy(track_a)
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
        "is_underpowered": True,
        "power_note": ("Track B PoC has ~15 stays / ~102 transitions — far too few to "
                       "exclude zero. ΔV̂ is a point estimate only; NO significance is "
                       "claimed (see SYSTEM_SUMMARY §4, §14)."),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = config.LOGS_PATH / "ablation_track_b.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("Ablation (UNDERPOWERED, n_test_episodes=%d): ΔV̂=%.4f "
             "[18d=%.4f vs 12d=%.4f] → %s",
             n_test_episodes, delta_v, result["v_hat_18dim"],
             result["v_hat_12dim"], out)
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
