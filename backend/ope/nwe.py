"""Nadaraya–Watson model-based evaluation (Section 7.3).

Non-parametric next-state estimator with an Epanechnikov kernel over (s, a, z),
used to roll out the policy and estimate its discounted return. Bandwidths are
simple median-heuristic defaults (full 5-fold CV omitted for the PoC).
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone

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


# --------------------------------------------------------------------------- #
# Clinical feature groups over the 12-dim tabular state (Methodology §10.3).
# Rationale: signals that co-move physiologically share a bandwidth, so the kernel
# is not dominated by whichever raw signal happens to have the largest spread.
#   * respiratory  — the gas-exchange / ventilator axis (PEEP, TV, FiO2, SpO2,
#     PaO2, PaCO2, pH, RR) all move together under a ventilation change.
#   * hemodynamic  — circulatory (HR, SBP, Temp).
#   * mental       — sedation / consciousness (RASS), on its own bounded scale.
# Each group gets its own state bandwidth; the state kernel is the product of the
# per-group Epanechnikov kernels (a separable product kernel).
FEATURE_GROUPS: dict[str, list[str]] = {
    "respiratory": ["PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2", "pH", "RR"],
    "hemodynamic": ["HR", "SBP", "Temp"],
    "mental":      ["RASS"],
}


def _group_indices() -> list[np.ndarray]:
    idx = {f: i for i, f in enumerate(TABULAR)}
    return [np.array([idx[f] for f in feats], dtype=int)
            for feats in FEATURE_GROUPS.values()]


class NWEModel:
    def __init__(self, S, A, NS, Z, hs, ha, hz, lam=1e-3, groups=None):
        """``hs`` is either a scalar (single global bandwidth over all state dims,
        back-compat) or a sequence aligned with ``groups`` (per-group bandwidths).
        ``groups`` is a list of feature-index arrays partitioning the state dims;
        ``None`` → a single group over all dims (the original behaviour)."""
        self.S, self.A, self.NS, self.Z = S, A, NS, Z
        self.ha, self.hz, self.lam = ha, hz, lam
        if groups is None:
            self.groups = [np.arange(S.shape[1], dtype=int)]
            self.hs = [float(hs)]
        else:
            self.groups = [np.asarray(g, dtype=int) for g in groups]
            hs_seq = hs if np.ndim(hs) else [hs] * len(self.groups)
            self.hs = [float(h) for h in hs_seq]

    def next_state(self, s, a, z):
        ks = np.ones(len(self.S))
        for g, h in zip(self.groups, self.hs):
            ks *= _epanechnikov(np.linalg.norm(self.S[:, g] - s[g], axis=1) / h)
        ka = _epanechnikov(np.abs(self.A - a) / self.ha)
        kz = _epanechnikov(np.abs(self.Z - z) / self.hz)
        w = ks * ka * kz
        denom = w.sum() + self.lam
        if denom <= self.lam:                      # no neighbours → nearest by state
            return self.NS[np.argmin(np.linalg.norm(self.S - s, axis=1))]
        return (w[:, None] * self.NS).sum(0) / denom

    def next_state_batch(self, s_q, a_q, z_q, chunk: int = 100_000) -> np.ndarray:
        """Vectorised ``next_state`` for a batch of m queries → (m, dim).

        Numerically equivalent to calling ``next_state`` per row, but computes the
        Epanechnikov kernel against the reference set in chunked matrix blocks
        (||s_q - S||² via the a·b expansion, no m×n×d tensor) so a whole rollout
        step over all trajectories is one pass instead of m per-row scans. The
        state kernel is the product of the per-group kernels; the overall squared
        distance (summed over groups) drives the no-neighbour nearest fallback."""
        s_q = np.asarray(s_q, dtype=float)
        a_q = np.asarray(a_q, dtype=float)
        z_q = np.asarray(z_q, dtype=float)
        m, dim = len(s_q), self.NS.shape[1]
        num = np.zeros((m, dim)); den = np.full(m, self.lam)
        best_d = np.full(m, np.inf); best_i = np.zeros(m, dtype=np.int64)
        sq_norm_g = [(s_q[:, g] ** 2).sum(1) for g in self.groups]
        for i in range(0, len(self.S), chunk):
            Sb, Ab, Zb, NSb = (self.S[i:i + chunk], self.A[i:i + chunk],
                               self.Z[i:i + chunk], self.NS[i:i + chunk])
            ks = np.ones((m, len(Sb)))
            d2_tot = np.zeros((m, len(Sb)))
            for g, h, sqn in zip(self.groups, self.hs, sq_norm_g):
                d2g = sqn[:, None] + (Sb[:, g] ** 2).sum(1)[None, :] \
                    - 2.0 * s_q[:, g] @ Sb[:, g].T
                d2g = np.maximum(d2g, 0.0)
                ks *= _epanechnikov(np.sqrt(d2g) / h)
                d2_tot += d2g
            w = (ks
                 * _epanechnikov(np.abs(a_q[:, None] - Ab[None, :]) / self.ha)
                 * _epanechnikov(np.abs(z_q[:, None] - Zb[None, :]) / self.hz))
            num += w @ NSb
            den += w.sum(1)
            dist = np.sqrt(d2_tot)
            cmin = dist.argmin(1); cmin_d = dist[np.arange(m), cmin]
            upd = cmin_d < best_d
            best_d[upd], best_i[upd] = cmin_d[upd], (i + cmin)[upd]
        out = num / den[:, None]
        no_nb = den <= self.lam + 1e-12                           # no neighbours → nearest
        if no_nb.any():
            out[no_nb] = self.NS[best_i[no_nb]]
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


def _norm_file(track: str) -> str:
    return "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"


def _global_hs(S: np.ndarray) -> float:
    """The original single median-heuristic state bandwidth over all dims."""
    return float(np.median(np.linalg.norm(S - S.mean(0), axis=1))) or 1.0


def _one_step_mse(ref: dict, val: dict, hs, groups, ha: float, hz: float) -> float:
    """Mean 1-step next-state MSE of an NW model built on ``ref``, scored on ``val``."""
    model = NWEModel(ref["S"], ref["A"], ref["NS"], ref["Z"],
                     hs=hs, ha=ha, hz=hz, groups=groups)
    pred = model.next_state_batch(val["S"], val["A"], val["Z"])
    return float(np.mean((pred - val["NS"]) ** 2))


def select_bandwidths(track: str = "a", ref_cap: int = 15000, val_cap: int = 1500,
                      seed: int = 0, z_override: np.ndarray | None = None) -> dict:
    """Choose per-group state bandwidths + action/propensity bandwidths by grid
    search minimising 1-step next-state MSE on a 20% held-out validation split
    carved from TRAIN (Methodology §10.3, à la Lee et al.).

    ``z_override`` supplies a per-transition propensity (e.g. dynamic z_t) in place
    of the static per-patient ``propensity_z``.

    Returns the selected bandwidths, the clinical groups, and the baseline (single
    median-heuristic) vs selected validation MSE for comparison/logging."""
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / _norm_file(track))
    feats = d["feature_order"]
    S = N.transform(d["states"], stats, feats)[:, :12]
    NS = N.transform(d["next_states"], stats, feats)[:, :12]
    A = d["actions"].astype(float)
    Z = (np.asarray(z_override, float) if z_override is not None
         else d["propensity_z"].astype(float))
    groups = _group_indices()

    rng = np.random.default_rng(seed)
    train = np.where(d["split"] == "train")[0]
    if len(train) < 100:                             # tiny (track B) → use all rows
        train = np.arange(len(S))
    rng.shuffle(train)
    cut = int(0.8 * len(train))
    ref_idx, val_idx = train[:cut], train[cut:]      # 80% reference / 20% held-out
    ref_idx = rng.choice(ref_idx, size=min(ref_cap, len(ref_idx)), replace=False)
    val_idx = rng.choice(val_idx, size=min(val_cap, len(val_idx)), replace=False)
    ref = {"S": S[ref_idx], "A": A[ref_idx], "NS": NS[ref_idx], "Z": Z[ref_idx]}
    val = {"S": S[val_idx], "A": A[val_idx], "NS": NS[val_idx], "Z": Z[val_idx]}

    # Per-group base scale (group-local median heuristic) → candidate multipliers.
    base = [float(np.median(np.linalg.norm(ref["S"][:, g] - ref["S"][:, g].mean(0),
                                           axis=1))) or 1.0 for g in groups]
    mult = [0.5, 1.0, 2.0]
    hs_cands = [[b * m for m in mult] for b in base]
    ha_cands = [0.5, 1.0, 2.0, 4.0]
    hz_cands = [0.1, 0.2, 0.4]

    # Baseline: the current single global median-heuristic (ha=1.0, hz=0.2).
    base_mse = _one_step_mse(ref, val, _global_hs(ref["S"]),
                             [np.arange(12)], ha=1.0, hz=0.2)

    # Coordinate-descent grid search (2 passes) over the per-axis candidate grids.
    hs = list(base); ha, hz = 1.0, 0.2
    for _ in range(2):
        for gi in range(len(groups)):
            best, best_v = hs[gi], np.inf
            for c in hs_cands[gi]:
                trial = list(hs); trial[gi] = c
                v = _one_step_mse(ref, val, trial, groups, ha, hz)
                if v < best_v:
                    best, best_v = c, v
            hs[gi] = best
        for c in ha_cands:
            if _one_step_mse(ref, val, hs, groups, c, hz) < \
               _one_step_mse(ref, val, hs, groups, ha, hz):
                ha = c
        for c in hz_cands:
            if _one_step_mse(ref, val, hs, groups, ha, c) < \
               _one_step_mse(ref, val, hs, groups, ha, hz):
                hz = c
    sel_mse = _one_step_mse(ref, val, hs, groups, ha, hz)
    return {
        "groups": {name: g.tolist() for name, g in zip(FEATURE_GROUPS, groups)},
        "hs": {name: round(h, 4) for name, h in zip(FEATURE_GROUPS, hs)},
        "ha": ha, "hz": hz,
        "val_mse_selected": round(sel_mse, 6),
        "val_mse_baseline_median": round(base_mse, 6),
        "n_ref": int(len(ref_idx)), "n_val": int(len(val_idx)),
    }


def _resolve_z(track: str, d: dict, propensity: str) -> np.ndarray:
    """Static per-patient propensity_z (default) or dynamic per-transition z_t."""
    if propensity == "dynamic":
        from backend.pipeline import propensity_dynamic
        return propensity_dynamic.compute_zt(track).astype(float)
    return d["propensity_z"].astype(float)


def fit(track: str = "a", cv: bool = True, propensity: str = "static",
        z_override: np.ndarray | None = None) -> NWEModel:
    """Fit the NW transition model over the 12 tabular state dims.

    ``cv=True`` (default) selects per-group state bandwidths + action/propensity
    bandwidths by held-out 1-step-MSE grid search (§10.3) and persists them on the
    model; ``cv=False`` reproduces the original single global median heuristic.

    ``propensity`` selects the conditioning variable: ``"static"`` (default →
    unchanged) uses per-patient ``propensity_z``; ``"dynamic"`` uses the §10.4
    per-transition z_t. Only the static model is written to the canonical
    ``nwe_model.pkl``; the dynamic model saves to ``nwe_model_dynamic.pkl`` so the
    deployed default OPE path is never clobbered."""
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / _norm_file(track))
    feats = d["feature_order"]
    S = N.transform(d["states"], stats, feats)[:, :12]        # first 12 = tabular
    NS = N.transform(d["next_states"], stats, feats)[:, :12]
    A = d["actions"].astype(float)
    Z = (np.asarray(z_override, float) if z_override is not None
         else _resolve_z(track, d, propensity))
    pkl = "nwe_model.pkl" if propensity == "static" else "nwe_model_dynamic.pkl"

    if cv:
        bw = select_bandwidths(track, z_override=Z)
        groups = _group_indices()
        hs = [bw["hs"][name] for name in FEATURE_GROUPS]
        model = NWEModel(S, A, NS, Z, hs=hs, ha=bw["ha"], hz=bw["hz"], groups=groups)
        model.bandwidths = bw                                  # persisted on the model
        model.save(config.MODEL_PATH / pkl)
        if propensity == "static":
            (config.LOGS_PATH / f"nwe_bandwidths_track_{track}.json").write_text(
                json.dumps(bw, indent=2))
        log.info("Fitted NWE (n=%d, %s) CV hs=%s ha=%.2f hz=%.2f "
                 "(val MSE %.4f vs median %.4f) → %s", len(S), propensity,
                 bw["hs"], bw["ha"], bw["hz"],
                 bw["val_mse_selected"], bw["val_mse_baseline_median"], pkl)
    else:
        hs = _global_hs(S)
        model = NWEModel(S, A, NS, Z, hs=hs, ha=1.0, hz=0.2)
        model.bandwidths = {"hs": {"global": round(hs, 4)}, "ha": 1.0, "hz": 0.2}
        model.save(config.MODEL_PATH / pkl)
        log.info("Fitted NWE (n=%d, hs=%.3f, no CV, %s) → %s",
                 len(S), hs, propensity, pkl)
    return model


# feature indices into the canonical 12-dim tabular state order
_PEEP_I, _FIO2_I, _SPO2_I = 0, 2, 3          # PEEP, FiO2, SpO2 in TABULAR
SPO2_TARGET = 95.0                            # "adequate oxygenation" threshold
AGGR_PEEP, AGGR_FIO2 = 15.0, 0.8              # aggressive-setting thresholds


def _safety_metrics(start_spo2: np.ndarray, terminal_spo2: np.ndarray,
                    peep_steps: np.ndarray, fio2_steps: np.ndarray) -> dict:
    """Clinical-safety summary (Methodology §12.3) from a set of trajectories.

    ``start_spo2`` / ``terminal_spo2``: (n,) raw SpO₂ at each episode's first /
    last visited state. ``peep_steps`` / ``fio2_steps``: (n_steps_total,) resulting
    PEEP / FiO₂ at every recommended step across all trajectories.

    ``mean_asynchrony_rate`` is a *waveform* quantity, so for Track A it is ``None``
    (the tabular arm has no airflow/asynchrony signal — see Task E; not imputed)."""
    lt95 = start_spo2 < SPO2_TARGET
    return {
        "pct_terminal_spo2_ge95": round(float(np.mean(terminal_spo2 >= SPO2_TARGET)), 4),
        "mean_delta_spo2_start_lt95": (
            round(float(np.mean(terminal_spo2[lt95] - start_spo2[lt95])), 4)
            if lt95.any() else None),
        "n_episodes_start_lt95": int(lt95.sum()),
        "pct_aggressive_steps": round(float(np.mean(
            (peep_steps > AGGR_PEEP) | (fio2_steps > AGGR_FIO2))), 4)
            if len(peep_steps) else 0.0,
        "mean_asynchrony_rate": None,   # waveform-only — Track A omits it
    }


def _clinician_safety(d: dict, starts: np.ndarray, T: int) -> dict:
    """Safety metrics for the LOGGED clinician policy (reuse behaviour.py logic:
    read the observed trajectory forward from each start, no rollout model)."""
    states = d["states"]                        # RAW physiological values
    next_states = d["next_states"]
    sid, hour = d["stay_id"], d["hour"]
    order = np.lexsort((hour, sid))
    stay_rows: dict[int, list[int]] = {}
    for pos in order:                            # hour-sorted rows per stay
        stay_rows.setdefault(int(sid[pos]), []).append(int(pos))

    start_spo2, term_spo2 = [], []
    peep_steps, fio2_steps = [], []
    for i in starts:
        rows = stay_rows[int(sid[i])]
        p = rows.index(int(i))
        seg = rows[p:p + T]
        start_spo2.append(float(states[seg[0], _SPO2_I]))
        term_spo2.append(float(next_states[seg[-1], _SPO2_I]))
        for r in seg:                            # resulting (next-state) settings
            peep_steps.append(float(next_states[r, _PEEP_I]))
            fio2_steps.append(float(next_states[r, _FIO2_I]))
    return _safety_metrics(np.array(start_spo2), np.array(term_spo2),
                           np.array(peep_steps), np.array(fio2_steps))


def rollout_value(track: str = "a", T: int = 24, gamma: float = 0.99,
                  n_starts: int = 200, propensity: str = "static") -> dict:
    d = D.load_mdp(track)
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    feats = d["feature_order"]
    Zsrc = _resolve_z(track, d, propensity)      # static propensity_z or dynamic z_t
    model = fit(track, propensity=propensity, z_override=Zsrc)

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

    # Roll all trajectories forward in lockstep: batched policy actions and a
    # single batched kernel per step (was m per-row scans over ~1M rows per step).
    # The conditioning z is held at each start's value across the rollout.
    n = len(starts)
    S_cur = Sz[starts].astype(float).copy()                       # (n, 12)
    z = Zsrc[starts].astype(float)
    w = d["weight_kg"][starts].astype(float)
    G = np.zeros(n)
    # Safety-metric accumulators over the HybridIQL rollout (denormalised).
    start_spo2 = N.inverse_transform(S_cur, stats, TABULAR)[:, _SPO2_I].copy()
    last_spo2 = start_spo2.copy()
    peep_steps, fio2_steps = [], []
    for t in range(T):
        s_raw = N.inverse_transform(S_cur, stats, TABULAR)        # (n, 12) raw
        full = np.zeros((n, len(feats)), dtype=np.float32); full[:, :12] = S_cur
        acts = pol.act_batch(full)                               # (n,)
        NSb = model.next_state_batch(S_cur, acts.astype(float), z)  # (n, 12)
        ns_raw = N.inverse_transform(NSb, stats, TABULAR)         # (n, 12) raw
        for j in range(n):
            sd = {f: float(s_raw[j, k]) for k, f in enumerate(TABULAR)}
            nd = {f: float(ns_raw[j, k]) for k, f in enumerate(TABULAR)}
            atuple = action_space.decode_action(int(acts[j]))
            G[j] += (gamma ** t) * tier1_reward(sd, nd, atuple, float(w[j]))
        peep_steps.append(ns_raw[:, _PEEP_I]); fio2_steps.append(ns_raw[:, _FIO2_I])
        last_spo2 = ns_raw[:, _SPO2_I]
        S_cur = NSb
    returns = G

    safety = {
        "hybrid_iql": _safety_metrics(start_spo2, last_spo2,
                                      np.concatenate(peep_steps),
                                      np.concatenate(fio2_steps)),
        "clinician": _clinician_safety(d, starts, T),
    }
    result = {"track": track, "propensity": propensity,
              "V_hat": round(float(returns.mean()), 4),
              "v_hat": round(float(returns.mean()), 4),
              "V_std": round(float(returns.std()), 4), "T": T,
              "n_starts": int(len(starts)), "n_transitions": int(len(d["states"])),
              "safety_metrics": safety,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    # Static (default) writes the canonical log; dynamic writes a sidecar so the
    # deployed OPE numbers are never clobbered by the ablation path.
    suffix = "" if propensity == "static" else "_dynamic"
    (config.LOGS_PATH / f"nwe_track_{track}{suffix}.json").write_text(
        json.dumps(result, indent=2))
    log.info("NWE rollout Track %s: V̂=%.3f ± %.3f | π term-SpO₂≥95=%.2f agg=%.2f | "
             "clin term-SpO₂≥95=%.2f agg=%.2f", track.upper(),
             result["V_hat"], result["V_std"],
             safety["hybrid_iql"]["pct_terminal_spo2_ge95"],
             safety["hybrid_iql"]["pct_aggressive_steps"],
             safety["clinician"]["pct_terminal_spo2_ge95"],
             safety["clinician"]["pct_aggressive_steps"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    args = ap.parse_args()
    rollout_value(args.track)


if __name__ == "__main__":
    main()
