"""Sweep cql.alpha at the chosen lam_causal to balance responsiveness vs support.

The §14.4 causal reward (lam_causal=2.5) makes the policy respond to abnormal
physiology but raises the CQL support-violation rate (§7 constraint). Raising
cql.alpha tightens support at the cost of more "hold". This finds the smallest
alpha that keeps all three §14.1 seed acceptance states responding AND drives the
empirical support-violation rate below 5%.

Run:  python -m backend.scripts.sweep_cql_alpha 0.5 1.0 2.0 3.0
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from backend.mdp import action_space, dataset as D, normaliser as N
from backend.pipeline import config
from backend.rl import trainer as T

CFG = config.REPO_ROOT / "backend" / "configs" / "track_a_config.yaml"
SEED = {"PEEP": 8, "TV": 480, "FiO2": 0.5, "SpO2": 91, "PaO2": 68, "PaCO2": 44,
        "pH": 7.37, "HR": 92, "SBP": 118, "RR": 22, "RASS": -2, "Temp": 37.2}
TAB = config.TABULAR_FEATURES
K_NN, MAX_VIOL = 20, 0.05


def _set_alpha(a: float) -> None:
    cfg = yaml.safe_load(CFG.read_text())
    cfg["cql"]["alpha"] = a
    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _decode(model, stats, **ov):
    s = {**SEED, **ov}
    z = N.transform(np.array([float(s[f]) for f in TAB]), stats, TAB)
    return action_space.decode_action(model.act(z))


def _seed_ok(model, stats) -> bool:
    dp, _, df = _decode(model, stats, SpO2=84, FiO2=0.5, PEEP=8)
    hypox = not (dp == 0 and df == 0)
    _, _, df2 = _decode(model, stats, SpO2=99, FiO2=0.6, PEEP=8)
    _, dt3, _ = _decode(model, stats, PaCO2=65, TV=300)
    return hypox and (df2 < 0) and (dt3 > 0)


def _support_rate(model, d) -> float:
    rng = np.random.default_rng(0)
    tr = np.where(d["split"] == "train")[0]
    te = np.where(d["split"] == "test")[0]
    tr = rng.choice(tr, size=min(100_000, len(tr)), replace=False)
    te = rng.choice(te, size=min(200, len(te)), replace=False)
    trS, trA = d["states"][tr], d["actions"][tr]
    v = 0
    for i in te:
        a = model.act(d["states"][i])
        dist = np.linalg.norm(trS - d["states"][i], axis=1)
        nn = np.argpartition(dist, K_NN)[:K_NN]
        if a not in set(int(x) for x in trA[nn]):
            v += 1
    return v / len(te)


def run(alphas, steps=30000):
    cfg_all = yaml.safe_load(CFG.read_text())
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    chosen = None
    for a in alphas:
        _set_alpha(a)
        cfg_all["cql"]["alpha"] = a
        d = T._normalise(D.load_mdp("a"), "a")
        tr = np.where(d["split"] == "train")[0]
        va = np.where(d["split"] == "val")[0]
        model, best, _ = T._run_training(d, tr, va, cfg_all, steps, "cpu", tag=f"[a{a}] ")
        ev = np.where(np.isin(d["split"], ["val", "test"]))[0]
        sub = np.random.default_rng(0).choice(ev, size=min(20000, len(ev)), replace=False)
        acts = np.array([model.act(d["states"][i]) for i in sub])
        hold = action_space.encode_action(0, 0, 0.0)
        bm = float(np.mean(acts == d["actions"][sub]))
        hs = float(np.mean(acts == hold))
        seed = _seed_ok(model, stats)
        viol = _support_rate(model, d)
        ok = seed and viol < MAX_VIOL
        print(f"alpha={a}: seed_ok={seed} support_viol={viol:.1%} "
              f"behaviour_match={bm:.4f} hold={hs:.4f}  -> {'FEASIBLE' if ok else 'no'}",
              flush=True)
        if ok and chosen is None:
            chosen = a
    print(f"\nCHOSEN cql.alpha = {chosen}")
    return chosen


if __name__ == "__main__":
    run([float(x) for x in sys.argv[1:]] or [0.5, 1.0, 2.0, 3.0])
