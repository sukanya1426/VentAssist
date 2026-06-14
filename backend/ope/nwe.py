"""Nadaraya–Watson model-based evaluation (Section 7.3).

Non-parametric next-state estimator with an Epanechnikov kernel over (s, a, z),
used to roll out the policy and estimate its discounted return. Bandwidths are
simple median-heuristic defaults (full 5-fold CV omitted for the PoC).
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from backend.mdp import action_space, dataset as D
from backend.mdp import normaliser as N
from backend.mdp.reward import tier1_reward
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("ope_nwe")
TABULAR = config.TABULAR_FEATURES


def _epanechnikov(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)


class NWEModel:
    def __init__(self, S, A, NS, Z, hs, ha, hz, lam=1e-3):
        self.S, self.A, self.NS, self.Z = S, A, NS, Z
        self.hs, self.ha, self.hz, self.lam = hs, ha, hz, lam

    def next_state(self, s, a, z):
        ks = _epanechnikov(np.linalg.norm(self.S - s, axis=1) / self.hs)
        ka = _epanechnikov(np.abs(self.A - a) / self.ha)
        kz = _epanechnikov(np.abs(self.Z - z) / self.hz)
        w = ks * ka * kz
        denom = w.sum() + self.lam
        if denom <= self.lam:                      # no neighbours → nearest by state
            return self.NS[np.argmin(np.linalg.norm(self.S - s, axis=1))]
        return (w[:, None] * self.NS).sum(0) / denom

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


def _norm_file(track: str) -> str:
    return "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"


def fit(track: str = "a") -> NWEModel:
    """Fit the NW transition model over the 12 tabular state dims."""
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / _norm_file(track))
    feats = d["feature_order"]
    S = N.transform(d["states"], stats, feats)[:, :12]        # first 12 = tabular
    NS = N.transform(d["next_states"], stats, feats)[:, :12]
    A = d["actions"].astype(float)
    Z = d["propensity_z"].astype(float)
    hs = float(np.median(np.linalg.norm(S - S.mean(0), axis=1))) or 1.0
    model = NWEModel(S, A, NS, Z, hs=hs, ha=1.0, hz=0.2)
    model.save(config.MODEL_PATH / "nwe_model.pkl")
    log.info("Fitted NWE (n=%d, hs=%.3f) → nwe_model.pkl", len(S), hs)
    return model


def rollout_value(track: str = "a", T: int = 24, gamma: float = 0.99,
                  n_starts: int = 200) -> dict:
    d = D.load_mdp(track)
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    feats = d["feature_order"]
    model = fit(track)

    from backend.rl.hybrid_iql import HybridIQL
    import torch
    ck = torch.load(config.MODEL_PATH / f"policy_track_{track}.pt", map_location="cpu")
    pol = HybridIQL(state_dim=ck["state_dim"], action_dim=ck["action_dim"],
                    hidden_dim=ck["hidden_dim"]); pol.load_state_dict(ck["state_dict"])

    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(d["states"]))
    rng = np.random.default_rng(0)
    starts = rng.choice(test, size=min(n_starts, len(test)), replace=False)
    Sz = N.transform(d["states"], stats, feats)[:, :12]

    returns = []
    for i in starts:
        s = Sz[i].copy(); z = float(d["propensity_z"][i]); w = float(d["weight_kg"][i])
        G = 0.0
        for t in range(T):
            full = np.zeros(len(feats)); full[:12] = s
            a = pol.act(full)
            ns = model.next_state(s, float(a), z)
            s_raw = N.inverse_transform(s, stats, TABULAR)
            ns_raw = N.inverse_transform(ns, stats, TABULAR)
            sd = {f: float(s_raw[j]) for j, f in enumerate(TABULAR)}
            nd = {f: float(ns_raw[j]) for j, f in enumerate(TABULAR)}
            G += (gamma ** t) * tier1_reward(sd, nd, w)
            s = ns
        returns.append(G)
    returns = np.array(returns)
    result = {"track": track, "V_hat": round(float(returns.mean()), 4),
              "V_std": round(float(returns.std()), 4), "T": T,
              "n_starts": int(len(starts))}
    (config.LOGS_PATH / f"nwe_track_{track}.json").write_text(json.dumps(result, indent=2))
    log.info("NWE rollout Track %s: V̂=%.3f ± %.3f", track.upper(),
             result["V_hat"], result["V_std"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    args = ap.parse_args()
    rollout_value(args.track)


if __name__ == "__main__":
    main()
