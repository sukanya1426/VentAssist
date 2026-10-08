"""Joint (lam_causal, cql.alpha) search for an operating point that satisfies BOTH
the §14.1 seed acceptance AND the §7 support constraint (<5% violation).

Responsiveness (causal reward) and support (CQL) are antagonistic: raising
lam_causal makes the policy take indicated-but-locally-rare actions (acceptance ↑,
support ↓), raising cql.alpha pushes back toward hold (support ↑, acceptance ↓).
This evaluates explicit (lam, alpha) pairs and reports the trade-off so a feasible
corner — if one exists — can be chosen.

Run:  python -m backend.scripts.sweep_joint  3:1.0  4:1.0  6:1.0  6:1.5
      (each arg is lam:alpha)
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


def _set(lam: float, alpha: float) -> None:
    cfg = yaml.safe_load(CFG.read_text())
    cfg["reward"]["lam_causal"] = lam
    cfg["cql"]["alpha"] = alpha
    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _decode(model, stats, **ov):
    s = {**SEED, **ov}
    z = N.transform(np.array([float(s[f]) for f in TAB]), stats, TAB)
    return action_space.decode_action(model.act(z))


def _seed_ok(model, stats) -> bool:
    dp, _, df = _decode(model, stats, SpO2=84, FiO2=0.5, PEEP=8)
    _, _, df2 = _decode(model, stats, SpO2=99, FiO2=0.6, PEEP=8)
    _, dt3, _ = _decode(model, stats, PaCO2=65, TV=300)
    return (not (dp == 0 and df == 0)) and (df2 < 0) and (dt3 > 0)


def _support_rate(model, d) -> float:
    rng = np.random.default_rng(0)
    tr = rng.choice(np.where(d["split"] == "train")[0],
                    size=min(100_000, int((d["split"] == "train").sum())), replace=False)
    te = rng.choice(np.where(d["split"] == "test")[0],
                    size=min(200, int((d["split"] == "test").sum())), replace=False)
    trS, trA = d["states"][tr], d["actions"][tr]
    v = 0
    for i in te:
        a = model.act(d["states"][i])
        nn = np.argpartition(np.linalg.norm(trS - d["states"][i], axis=1), K_NN)[:K_NN]
        if a not in set(int(x) for x in trA[nn]):
            v += 1
    return v / len(te)


def run(pairs, steps=30000):
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    chosen = None
    for lam, alpha in pairs:
        _set(lam, alpha)
        cfg_all = yaml.safe_load(CFG.read_text())
        d = T._normalise(D.load_mdp("a"), "a")
        tr = np.where(d["split"] == "train")[0]
        va = np.where(d["split"] == "val")[0]
        model, _, _ = T._run_training(d, tr, va, cfg_all, steps, "cpu",
                                   tag=f"[l{lam}a{alpha}] ")
        ev = np.where(np.isin(d["split"], ["val", "test"]))[0]
        sub = np.random.default_rng(0).choice(ev, size=min(20000, len(ev)), replace=False)
        acts = np.array([model.act(d["states"][i]) for i in sub])
        hold = action_space.encode_action(0, 0, 0.0)
        bm = float(np.mean(acts == d["actions"][sub]))
        hs = float(np.mean(acts == hold))
        seed = _seed_ok(model, stats)
        viol = _support_rate(model, d)
        ok = seed and viol < MAX_VIOL
        print(f"lam={lam} alpha={alpha}: seed_ok={seed} support={viol:.1%} "
              f"behaviour_match={bm:.4f} hold={hs:.4f}  -> {'FEASIBLE' if ok else 'no'}",
              flush=True)
        if ok and chosen is None:
            chosen = (lam, alpha)
    print(f"\nCHOSEN (lam, alpha) = {chosen}")
    return chosen


if __name__ == "__main__":
    pairs = [tuple(float(x) for x in a.split(":")) for a in sys.argv[1:]]
    run(pairs or [(3, 1.0), (4, 1.0), (6, 1.0), (6, 1.5)])
