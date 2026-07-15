"""Clinician action-conditional density π_β(a|s) — the IntelliLung Rule-5 OOD gate.

BENCHMARK_PLAN §0 trap 5: the reference system's headline metric is a **pair**
— `(FQE value, action log-likelihood under a clinician density model)`. An FQE
gain accompanied by a drop in action likelihood is *extrapolation*, not an
improvement. VentAssist could not previously produce that second number: we have
a **state**-density OOD detector (`backend/rl/ood_autoencoder.py`) but no
**action-conditional** density.

This module fits π_β(a|s) — the probability the *clinician* would have taken
action `a` in state `s` — and scores a policy by the mean log-likelihood of the
actions it recommends on held-out states. Higher = more on-support.

Ours is structurally simpler than theirs: IntelliLung needs a Gaussian over 5
continuous settings **plus** a categorical over the mode (a hybrid density). Our
action space is fully discrete, so π_β(a|s) is a single 125-way categorical — a
softmax MLP trained with cross-entropy, which gives properly normalised log-probs.

Reported (mirrors their `ae_losses.csv` + `individual_action_losses.json`):
  * ``loglik_policy``     — mean log π_β(π(s) | s) over held-out states
  * ``loglik_clinician``  — mean log π_β(a_clin | s) on the same states (reference)
  * ``delta``             — policy − clinician. **≤ 0 means the policy is acting
                            off the clinician's support.**
  * per-action-factor breakdown (ΔPEEP / ΔTV / ΔFiO₂), obtained by marginalising
    the 125-way categorical — shows *which knob* a policy is being reckless about.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.action_density --track a
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.mdp import action_space
from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("bench_action_density")

RESULTS = Path(__file__).resolve().parent / "results"

# Index of each action's (ΔPEEP, ΔTV, ΔFiO₂) bin, for the per-factor marginals.
_PEEP_BIN = np.array([action_space.DELTA_PEEP_BINS.index(action_space.ACTION_MAP[i][0])
                      for i in range(action_space.N_ACTIONS)])
_TV_BIN = np.array([action_space.DELTA_TV_BINS.index(action_space.ACTION_MAP[i][1])
                    for i in range(action_space.N_ACTIONS)])
_FIO2_BIN = np.array([action_space.DELTA_FIO2_BINS.index(action_space.ACTION_MAP[i][2])
                      for i in range(action_space.N_ACTIONS)])
_FACTORS = {"delta_PEEP": _PEEP_BIN, "delta_TV": _TV_BIN, "delta_FiO2": _FIO2_BIN}


class BehaviourDensity(nn.Module):
    """Softmax MLP over the 125 discrete actions → properly normalised log π_β(a|s)."""

    def __init__(self, state_dim: int, n_actions: int = action_space.N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)                                   # logits

    @torch.no_grad()
    def log_probs(self, S: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """(N, 125) log π_β(a|s)."""
        self.eval()
        out = np.empty((len(S), action_space.N_ACTIONS), dtype=np.float32)
        for i in range(0, len(S), chunk):
            s = torch.as_tensor(S[i:i + chunk], dtype=torch.float32)
            out[i:i + chunk] = F.log_softmax(self(s), dim=-1).numpy()
        return out


def fit(S_train: np.ndarray, A_train: np.ndarray, epochs: int = 15,
        batch_size: int = 4096, lr: float = 1e-3, seed: int = 0) -> BehaviourDensity:
    torch.manual_seed(seed)
    model = BehaviourDensity(S_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    St = torch.as_tensor(S_train, dtype=torch.float32)
    At = torch.as_tensor(A_train, dtype=torch.long)
    rng = np.random.default_rng(seed)
    n = len(S_train)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = torch.as_tensor(perm[i:i + batch_size])
            loss = F.cross_entropy(model(St[b]), At[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        log.info("π_β epoch %d/%d | CE=%.4f", ep + 1, epochs, tot / n)
    return model


def _factor_loglik(logp: np.ndarray, actions: np.ndarray) -> dict:
    """Marginalise the 125-way categorical onto each action factor and score it.

    Mirrors IntelliLung's per-action-dimension NLL: shows which knob the policy is
    being reckless about, not just that it is off-support somewhere."""
    p = np.exp(logp)                                          # (N, 125)
    out = {}
    for name, bins in _FACTORS.items():
        n_bins = int(bins.max()) + 1
        # marginal prob of each bin = sum of the joint over actions in that bin
        marg = np.stack([p[:, bins == b].sum(1) for b in range(n_bins)], axis=1)  # (N, n_bins)
        taken = bins[actions]                                 # bin actually taken
        lp = np.log(np.clip(marg[np.arange(len(actions)), taken], 1e-12, None))
        out[name] = round(float(lp.mean()), 4)
    return out


def evaluate(track: str = "a", epochs: int = 15, seed: int = 0) -> dict:
    """Fit π_β on train, score the deployed policy vs the clinician on the test split."""
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / (
        "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"))
    feats = d["feature_order"]
    # Density model is fit in the SAME (training) encoding the behaviour data lives in.
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    A = d["actions"].astype(np.int64)

    train = np.where(d["split"] == "train")[0]
    test = np.where(d["split"] == "test")[0]
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("MDP has no train/test split.")

    log.info("Fitting clinician action density π_β(a|s) on %d train transitions…", len(train))
    model = fit(S[train], A[train], epochs=epochs, seed=seed)

    # --- score on held-out states ---
    from backend.ope.fqe import _load_policy
    policy, _ = _load_policy(track)
    S_test, A_clin = S[test], A[test]
    pi_actions = policy.act_batch(S_test)                     # the policy's chosen actions

    logp = model.log_probs(S_test)                            # (n_test, 125)
    idx = np.arange(len(S_test))
    ll_policy = float(logp[idx, pi_actions].mean())
    ll_clin = float(logp[idx, A_clin].mean())

    result = {
        "track": track,
        "n_test": int(len(test)),
        "n_transitions": int(len(A)),
        "loglik_policy": round(ll_policy, 4),
        "loglik_clinician": round(ll_clin, 4),
        # ≤ 0 ⇒ the policy acts off the clinician's support (extrapolation risk)
        "delta_policy_minus_clinician": round(ll_policy - ll_clin, 4),
        "perplexity_policy": round(float(np.exp(-ll_policy)), 2),
        "perplexity_clinician": round(float(np.exp(-ll_clin)), 2),
        "per_factor_loglik_policy": _factor_loglik(logp, pi_actions),
        "per_factor_loglik_clinician": _factor_loglik(logp, A_clin),
        "agreement_with_clinician": round(float(np.mean(pi_actions == A_clin)), 4),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"action_density_track_{track}.json").write_text(json.dumps(result, indent=2))
    log.info("π_β log-lik | policy=%.4f  clinician=%.4f  Δ=%.4f  (Δ≤0 ⇒ off-support)",
             ll_policy, ll_clin, ll_policy - ll_clin)
    log.info("per-factor (policy): %s", result["per_factor_loglik_policy"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evaluate(args.track, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
