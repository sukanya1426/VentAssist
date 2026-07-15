"""Multi-output (LMC / intrinsic coregionalization) GP imputation — Methodology §3.

The default lab imputation (backend/pipeline/gp_imputation.py) fits an *independent*
Matérn-3/2 GP per blood-gas signal (PaO₂, PaCO₂, pH), ignoring the strong
inter-signal correlation — those three come from the *same* arterial blood-gas
draw and move together (e.g. pH ↔ PaCO₂ via the Henderson–Hasselbalch relation).

This module adds the LMC path as an **Intrinsic Coregionalization Model (ICM)**: a
single shared latent temporal kernel k(t,t') (Matérn-3/2) coupled across signals
by a coregionalization matrix W = B Bᵀ, giving the multi-task covariance

    Cov(f_d(t), f_{d'}(t')) = W[d,d'] · k(t, t').

Observing one signal then informs the others at unobserved times through W — the
cross-signal information the independent GP throws away.

In-house (numpy) implementation — no gpytorch dependency (kept optional per the
repo's light-dependency policy; the independent path still uses gpytorch's Matérn).
W is estimated from the empirical correlation of same-draw co-observations (a
principled, data-driven B = chol(W)); the shared lengthscale is chosen by 5-fold
CV over held-out time points, minimising held-out RMSE.
"""

from __future__ import annotations

import numpy as np

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("gp_lmc")

FEATURES = config.LAB_FEATURES                      # ["PaO2", "PaCO2", "pH"]
_LENGTHSCALE_GRID = [1.0, 2.0, 4.0, 8.0]
_JITTER = 1e-6


def _matern32(dt: np.ndarray, lengthscale: float) -> np.ndarray:
    """Matérn-3/2 correlation for a |t-t'| distance matrix."""
    r = np.sqrt(3.0) * np.abs(dt) / lengthscale
    return (1.0 + r) * np.exp(-r)


class ICMImputer:
    """Fixed-hyperparameter ICM posterior-mean imputer over the lab signals."""

    def __init__(self, lengthscale: float, noise: float, W: np.ndarray,
                 task_mean: np.ndarray, task_std: np.ndarray,
                 features: list[str] = FEATURES):
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.W = np.asarray(W, float)               # (D, D) task covariance
        self.task_mean = np.asarray(task_mean, float)
        self.task_std = np.asarray(task_std, float)
        self.features = list(features)
        self.idx = {f: i for i, f in enumerate(features)}

    def _standardise(self, d: int, y: np.ndarray) -> np.ndarray:
        return (y - self.task_mean[d]) / self.task_std[d]

    def impute_episode(self, obs: dict[str, tuple[np.ndarray, np.ndarray]],
                       hours: np.ndarray) -> dict[str, np.ndarray]:
        """Posterior mean per signal at ``hours`` given per-signal (t_obs, y_obs).

        Signals absent from ``obs`` (or with < 2 points) are left to the caller's
        fallback — this returns predictions only for the signals it can inform
        (including via cross-signal coupling when *any* signal is observed)."""
        t_list, d_list, y_list = [], [], []
        for f, (t_obs, y_obs) in obs.items():
            if len(t_obs) == 0:
                continue
            d = self.idx[f]
            t_list.append(np.asarray(t_obs, float))
            d_list.append(np.full(len(t_obs), d, dtype=int))
            y_list.append(self._standardise(d, np.asarray(y_obs, float)))
        if not t_list:
            return {}
        t_all = np.concatenate(t_list)
        d_all = np.concatenate(d_list)
        y_all = np.concatenate(y_list)

        # K[(d_i,t_i),(d_j,t_j)] = W[d_i,d_j] · k(t_i,t_j) (+ noise on the diagonal)
        Kt = _matern32(t_all[:, None] - t_all[None, :], self.lengthscale)
        K = self.W[np.ix_(d_all, d_all)] * Kt
        K[np.diag_indices_from(K)] += self.noise + _JITTER
        try:
            alpha = np.linalg.solve(K, y_all)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K, y_all, rcond=None)[0]

        out: dict[str, np.ndarray] = {}
        observed_tasks = set(int(x) for x in d_all)
        for f in self.features:
            d = self.idx[f]
            # Predict a task only if it — or a signal correlated with it — is seen.
            if d not in observed_tasks and np.allclose(self.W[d, list(observed_tasks)], 0):
                continue
            Kst = _matern32(hours[:, None] - t_all[None, :], self.lengthscale)
            Ks = self.W[d, d_all][None, :] * Kst
            mu = Ks @ alpha
            lo, hi = config.LAB_CLIP_RANGES[f]
            out[f] = np.clip(mu * self.task_std[d] + self.task_mean[d], lo, hi)
        return out


# --------------------------------------------------------------------------- #
# Global fit: task stats, coregionalization W, and CV-selected lengthscale
# --------------------------------------------------------------------------- #
def _task_stats(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pooled = {f: [] for f in FEATURES}
    for ep in episodes:
        for f, (_, y) in ep.items():
            pooled[f].extend(np.asarray(y, float).tolist())
    mean = np.array([np.mean(pooled[f]) if pooled[f] else 0.0 for f in FEATURES])
    std = np.array([np.std(pooled[f]) or 1.0 if pooled[f] else 1.0 for f in FEATURES])
    return mean, std


def _estimate_W(episodes: list[dict], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Coregionalization matrix from same-time co-observations (empirical corr)."""
    D = len(FEATURES)
    rows = []
    for ep in episodes:
        # bucket standardised obs by integer time, keep only complete/co-observed
        by_t: dict[float, dict[int, float]] = {}
        for f, (t, y) in ep.items():
            d = FEATURES.index(f)
            for ti, yi in zip(np.asarray(t, float), np.asarray(y, float)):
                by_t.setdefault(round(float(ti), 3), {})[d] = (yi - mean[d]) / std[d]
        for vals in by_t.values():
            if len(vals) >= 2:                     # at least a pair to correlate
                v = np.full(D, np.nan)
                for d, x in vals.items():
                    v[d] = x
                rows.append(v)
    if len(rows) < 5:
        return np.eye(D)                            # insufficient → independent
    M = np.array(rows)
    # pairwise correlation over available (non-nan) pairs
    W = np.eye(D)
    for a in range(D):
        for b in range(a + 1, D):
            m = ~np.isnan(M[:, a]) & ~np.isnan(M[:, b])
            if m.sum() >= 5 and M[m, a].std() > 0 and M[m, b].std() > 0:
                c = float(np.corrcoef(M[m, a], M[m, b])[0, 1])
                W[a, b] = W[b, a] = np.clip(c, -0.99, 0.99)
    # project to PSD (correlation matrices should be PSD; ridge for safety)
    ev, U = np.linalg.eigh(W)
    W = (U * np.clip(ev, 1e-3, None)) @ U.T
    dsc = np.sqrt(np.diag(W))
    return W / np.outer(dsc, dsc)                   # renormalise to unit diagonal


def _cv_rmse(episodes: list[dict], W: np.ndarray, mean: np.ndarray, std: np.ndarray,
             lengthscale: float, noise: float, folds: int = 5,
             seed: int = 0) -> dict[str, float]:
    """Per-signal held-out RMSE via k-fold CV over observation time points."""
    rng = np.random.default_rng(seed)
    imp = ICMImputer(lengthscale, noise, W, mean, std)
    sq = {f: [] for f in FEATURES}
    for ep in episodes:
        # assign each observation to a fold
        assign = {f: rng.integers(0, folds, len(t)) for f, (t, _) in ep.items()}
        for k in range(folds):
            train = {f: (t[assign[f] != k], y[assign[f] != k])
                     for f, (t, y) in ep.items()}
            if all(len(t) == 0 for t, _ in train.values()):
                continue
            for f, (t, y) in ep.items():
                mask = assign[f] == k
                if not mask.any():
                    continue
                pred = imp.impute_episode(train, t[mask])
                if f in pred:
                    sq[f].extend(((pred[f] - y[mask]) ** 2).tolist())
    return {f: float(np.sqrt(np.mean(sq[f]))) if sq[f] else float("nan")
            for f in FEATURES}


def crossval_compare(episodes: list[dict], noise: float = 0.1,
                     folds: int = 5, seed: int = 0) -> dict:
    """5-fold CV comparison of independent-Matérn (W=I) vs LMC (W=corr), selecting
    the shared lengthscale that minimises mean LMC held-out RMSE."""
    mean, std = _task_stats(episodes)
    W = _estimate_W(episodes, mean, std)
    D = len(FEATURES)

    best_ls, best_score, best = _LENGTHSCALE_GRID[0], np.inf, None
    for ls in _LENGTHSCALE_GRID:
        rmse_lmc = _cv_rmse(episodes, W, mean, std, ls, noise, folds, seed)
        score = np.nanmean([rmse_lmc[f] for f in FEATURES])
        if score < best_score:
            best_ls, best_score, best = ls, score, rmse_lmc
    rmse_matern = _cv_rmse(episodes, np.eye(D), mean, std, best_ls, noise, folds, seed)
    return {
        "lengthscale": best_ls, "noise": noise,
        "W": W.tolist(), "task_mean": mean.tolist(), "task_std": std.tolist(),
        "rmse_matern": rmse_matern, "rmse_lmc": best,
        "mean_rmse_matern": float(np.nanmean([rmse_matern[f] for f in FEATURES])),
        "mean_rmse_lmc": float(best_score),
        "n_episodes": len(episodes),
    }


def fit_global(episodes: list[dict], noise: float = 0.1) -> tuple[ICMImputer, dict]:
    """Select hyperparameters by CV and return a ready ICM imputer + the report."""
    rep = crossval_compare(episodes, noise=noise)
    imp = ICMImputer(rep["lengthscale"], noise, np.array(rep["W"]),
                     np.array(rep["task_mean"]), np.array(rep["task_std"]))
    return imp, rep
