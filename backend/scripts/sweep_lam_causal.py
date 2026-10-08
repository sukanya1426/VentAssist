"""Sweep the action-causal reward weight lam_causal (§14.4 tuning).

For each candidate lam_causal: set it in the Track A config, retrain the final
HybridIQL on the canonical split (no dataset rebuild — load_mdp recombines
reward_base + lam*causal_unit), and report (a) whether the three §14.1 seed
acceptance states are responded to, and (b) behaviour-match / hold-share on a
val+test subsample. Picks the smallest lam that passes all three seed tests.

Run:  python -m backend.scripts.sweep_lam_causal
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


def _set_lam(lam: float) -> None:
    cfg = yaml.safe_load(CFG.read_text())
    cfg["reward"]["lam_causal"] = lam
    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _act(model, stats, **ov):
    s = {**SEED, **ov}
    vec = np.array([float(s[f]) for f in TAB])
    z = N.transform(vec, stats, TAB)
    return action_space.decode_action(model.act(z))


def _seed_pass(model, stats) -> dict:
    dp, dt, df = _act(model, stats, SpO2=84, FiO2=0.5, PEEP=8)
    hypox = not (dp == 0 and df == 0)
    _, _, df2 = _act(model, stats, SpO2=99, FiO2=0.6, PEEP=8)
    hyper = df2 < 0
    _, dt3, _ = _act(model, stats, PaCO2=65, TV=300)
    hcap = dt3 > 0
    return {"hypoxaemia": hypox, "hyperoxia": hyper, "hypercapnia": hcap,
            "all": hypox and hyper and hcap}


def run(lams=(0.8, 1.5, 2.5, 4.0), steps=30000):
    cfg_all = yaml.safe_load(CFG.read_text())
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    chosen = None
    for lam in lams:
        _set_lam(lam)
        d = T._normalise(D.load_mdp("a"), "a")
        tr = np.where(d["split"] == "train")[0]
        va = np.where(d["split"] == "val")[0]
        model, best, _ = T._run_training(d, tr, va, cfg_all, steps, "cpu", tag=f"[lam{lam}] ")
        # behaviour-match / hold-share on a subsample of val+test
        ev = np.where(np.isin(d["split"], ["val", "test"]))[0]
        sub = np.random.default_rng(0).choice(ev, size=min(20000, len(ev)), replace=False)
        acts = np.array([model.act(d["states"][i]) for i in sub])
        hold = action_space.encode_action(0, 0, 0.0)
        bm = float(np.mean(acts == d["actions"][sub]))
        hs = float(np.mean(acts == hold))
        sp = _seed_pass(model, stats)
        print(f"lam={lam}: seed={sp} | behaviour_match={bm:.4f} hold={hs:.4f} val_q={best:.4f}",
              flush=True)
        if sp["all"] and chosen is None:
            chosen = lam
    print(f"\nCHOSEN lam_causal = {chosen}")
    return chosen


if __name__ == "__main__":
    lams = [float(x) for x in sys.argv[1:]] or [0.8, 1.5, 2.5, 4.0]
    run(lams)
