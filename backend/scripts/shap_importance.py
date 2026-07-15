"""Global SHAP-style feature importance over the learned Q-function (§12.4, Task F).

The methodology asks for SHAP values over the deployed policy's value function to
rank global feature importance (waveform-vs-tabular in the full spec; for Track A
we produce the tabular ranking over the 12 clinical features).

The scalar being explained is the **greedy-action Q-value** (the state value)
    f(s) = max_a Q(s, a)
of the deployed Track A policy, evaluated on a sample of test states in the
model's normalised feature space (so the 12 z-scored features are on a comparable
scale and the training-mean baseline is the origin).

Estimator
---------
Default is an **in-house permutation-sampling Shapley estimator** (no ``shap``
dependency — consistent with the repo's light-dependency policy, see
IMPLEMENTATION_NOTES.md). For each sampled instance x and each of ``n_perms``
random feature orderings, features are revealed one at a time from the baseline
(the training mean → the origin in normalised space) to x; feature i's marginal
contribution is f(coalition ∪ {i}) − f(coalition), averaged over permutations to
give its Shapley value φ_i(x). Global importance = mean_x |φ_i(x)|, ranked.

If the optional ``shap`` package is installed, ``--backend shap`` uses
``KernelExplainer`` on the same f and background, and the two rankings can be
cross-checked (Spearman); otherwise the in-house estimator is authoritative. This
is an ADDITIVE global view — the per-recommendation perturbation attribution
(backend/explainability/attribution.py) is unchanged and still used at serve time.

Run:
    python -m backend.scripts.shap_importance --track a
    python -m backend.scripts.shap_importance --track a --backend shap   # if installed
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.ope.fqe import _load_policy
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("shap_importance")
TAB = config.TABULAR_FEATURES


def _value_fn(policy):
    """f(z) = max_a Q(z) for a (n, d) batch of normalised states → (n,)."""
    @torch.no_grad()
    def f(z: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(z, dtype=np.float32))
        if t.ndim == 1:
            t = t.reshape(1, -1)
        return policy.Q(policy._phi(t)).max(dim=-1).values.cpu().numpy()
    return f


def _sample_states(track: str, n_states: int, seed: int):
    d = D.load_mdp(track)
    feats = d["feature_order"]
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    S = N.transform(d["states"], stats, feats).astype(np.float32)[:, :len(TAB)]
    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(S))
    rng = np.random.default_rng(seed)
    idx = rng.choice(test, size=min(n_states, len(test)), replace=False)
    X = S[idx]
    # Baseline = training mean in normalised space (≈ origin). Use the actual
    # train-sample mean rather than exact 0 so it is robust to winsor asymmetry.
    train = np.where(d["split"] == "train")[0]
    baseline = S[train].mean(0) if len(train) else np.zeros(S.shape[1], np.float32)
    return X, baseline.astype(np.float32)


def shapley_permutation(f, X: np.ndarray, baseline: np.ndarray,
                        n_perms: int = 64, seed: int = 0) -> np.ndarray:
    """Per-instance Shapley values via permutation sampling → (n_states, n_feat).

    Deterministic for a fixed ``seed``. Vectorised over instances: for each
    permutation the 13 coalition states (all-baseline → full x) are evaluated in
    one batched forward pass per instance block."""
    n, m = X.shape
    rng = np.random.default_rng(seed)
    phi = np.zeros((n, m), dtype=np.float64)
    base = np.broadcast_to(baseline, (n, m)).copy()
    for _ in range(n_perms):
        order = rng.permutation(m)
        # coalitions[k] = features order[:k] taken from X, rest from baseline.
        coalitions = np.empty((m + 1, n, m), dtype=np.float32)
        cur = base.copy()
        coalitions[0] = cur
        for k, feat in enumerate(order):
            cur = cur.copy()
            cur[:, feat] = X[:, feat]
            coalitions[k + 1] = cur
        vals = f(coalitions.reshape((m + 1) * n, m)).reshape(m + 1, n)  # (m+1, n)
        marg = np.diff(vals, axis=0)                                    # (m, n)
        for k, feat in enumerate(order):
            phi[:, feat] += marg[k]
    return phi / n_perms


def shap_kernel(f, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Optional cross-check via shap.KernelExplainer (only if shap installed)."""
    import shap                                    # optional dependency
    explainer = shap.KernelExplainer(f, baseline.reshape(1, -1))
    sv = explainer.shap_values(X, nsamples=200, silent=True)
    return np.asarray(sv)


def compute(track: str = "a", backend: str = "inhouse", n_states: int = 200,
            n_perms: int = 64, seed: int = 0) -> dict:
    policy, _ = _load_policy(track)
    f = _value_fn(policy)
    X, baseline = _sample_states(track, n_states, seed)

    phi = shapley_permutation(f, X, baseline, n_perms=n_perms, seed=seed)
    importance = np.abs(phi).mean(0)               # global mean |φ|
    order = np.argsort(importance)[::-1]
    ranking = [{"feature": TAB[i], "importance": round(float(importance[i]), 5),
                "mean_signed": round(float(phi[:, i].mean()), 5)} for i in order]

    result = {
        "track": track, "backend": "inhouse_permutation_shapley",
        "value_fn": "max_a Q(s,a)", "n_states": int(len(X)), "n_perms": n_perms,
        "seed": seed, "feature_order": TAB,
        "importance": {TAB[i]: round(float(importance[i]), 5) for i in range(len(TAB))},
        "ranking": ranking,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if backend == "shap":
        try:
            sv = shap_kernel(f, X, baseline)
            imp_shap = np.abs(sv).mean(0)
            from scipy.stats import spearmanr
            rho = float(spearmanr(importance, imp_shap).correlation)
            result["shap_importance"] = {TAB[i]: round(float(imp_shap[i]), 5)
                                         for i in range(len(TAB))}
            result["spearman_inhouse_vs_shap"] = round(rho, 4)
            log.info("shap cross-check: Spearman(in-house, shap) = %.3f", rho)
        except ImportError:
            log.warning("shap not installed — reporting in-house importances only.")
            result["shap_note"] = "shap not installed; in-house estimator used"

    out = config.LOGS_PATH / f"shap_importance_track_{track}.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("SHAP importance Track %s (top-3): %s", track.upper(),
             ", ".join(f"{r['feature']}={r['importance']:.3f}" for r in ranking[:3]))
    return result


def main():
    ap = argparse.ArgumentParser(description="Global Q-function feature importance.")
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--backend", default="inhouse", choices=["inhouse", "shap"])
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--n-perms", type=int, default=64)
    args = ap.parse_args()
    compute(args.track, backend=args.backend, n_states=args.n_states,
            n_perms=args.n_perms)


if __name__ == "__main__":
    main()
