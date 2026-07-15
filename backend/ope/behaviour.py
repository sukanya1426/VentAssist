"""Behaviour (clinician) policy value — OPE reference baseline.

Mirrors the physician/average-estimator baseline used by Lee et al. (2025) and
the IntelliLung ``estimators/physician.py``: the empirical discounted return of
the LOGGED clinician policy, computed directly from the observed rewards in the
offline dataset. No model is fit — for the behaviour policy the on-data Monte
Carlo return is itself an unbiased value estimate.

This anchors the FQE / DFQE / NWE numbers on the SAME reward scale, so the
learned policy's V̂ can be read as "does it beat the clinician?": a HybridIQL V̂
above this baseline is the headline claim; below it is a red flag.

Per episode (one stay_id, ordered by hour) the discounted return is
    G = Σ_t γ^t · r_t
and V̂_clin = mean(G) over episodes, reported overall and on the test split, with
a bootstrap 95% CI over episodes.

Run:
    python -m backend.ope.behaviour --track a
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from backend.mdp import dataset as D
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("ope_behaviour")


def _episode_returns(stay_id: np.ndarray, hour: np.ndarray, reward: np.ndarray,
                     gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Discounted return per episode. Returns (stay_ids, returns) aligned arrays."""
    order = np.lexsort((hour, stay_id))            # sort by stay_id, then hour
    sid, r = stay_id[order], reward[order]
    returns: list[float] = []
    ids: list[int] = []
    start = 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            seg = r[start:i]
            disc = gamma ** np.arange(len(seg), dtype=np.float64)
            returns.append(float(np.sum(disc * seg)))
            ids.append(int(sid[start]))
            start = i
    return np.asarray(ids, dtype=np.int64), np.asarray(returns, dtype=np.float64)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def behaviour_value(track: str = "a", gamma: float = 0.99) -> dict:
    d = D.load_mdp(track)
    ids, G = _episode_returns(d["stay_id"], d["hour"], d["rewards"], gamma)

    # test-split episodes (any transition in the test split → episode is test)
    test_ids = set(np.unique(d["stay_id"][d["split"] == "test"]).tolist())
    test_mask = np.array([i in test_ids for i in ids], dtype=bool)
    G_test = G[test_mask] if test_mask.any() else G

    lo, hi = _bootstrap_ci(G_test)
    result = {
        "track": track,
        "estimator": "behaviour_mc",
        "V_hat": round(float(G_test.mean()), 4),
        "v_hat": round(float(G_test.mean()), 4),
        "V_std": round(float(G_test.std()), 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "n_episodes": int(len(G)),
        "n_test_episodes": int(len(G_test)),
        "gamma": gamma,
        "V_hat_all_splits": round(float(G.mean()), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = config.LOGS_PATH / f"behaviour_track_{track}.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("Behaviour (clinician) Track %s: V̂=%.4f [%.4f, %.4f] (n_test=%d episodes)",
             track.upper(), result["V_hat"], lo, hi, len(G_test))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Behaviour-policy OPE baseline.")
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()
    behaviour_value(args.track, gamma=args.gamma)


if __name__ == "__main__":
    main()
