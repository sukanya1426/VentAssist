"""Train a final Track A model at (lam_causal, alpha) and probe HOLISTIC behaviour.

Unlike sweep_*.py (which only check the 3 seed acceptance states), this prints the
recommendation across a clinically diverse battery so we can pick an operating point
that responds sensibly across PEEP/TV/FiO2 AND safety — not one that games 3 tests
by spamming a single action.

Run:  python -m backend.scripts.probe_operating_point 1.0:0.5 1.5:0.5 2.5:0.5
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from backend.mdp import action_space, dataset as D, normaliser as N
from backend.pipeline import config
from backend.rl import trainer as T

CFG = config.REPO_ROOT / "backend" / "configs" / "track_a_config.yaml"
TAB = config.TABULAR_FEATURES
BASE = {"PEEP": 8, "TV": 480, "FiO2": 0.5, "SpO2": 91, "PaO2": 68, "PaCO2": 44,
        "pH": 7.37, "HR": 92, "SBP": 118, "RR": 22, "RASS": -2, "Temp": 37.2}
CASES = {
    "stable in-band (SpO2 94,FiO2 .4)": {"SpO2": 94, "FiO2": 0.4, "PaCO2": 40},
    "ACCEPT hypoxaemia (SpO2 84,FiO2 .5)": {"SpO2": 84, "FiO2": 0.5, "PEEP": 8},
    "ACCEPT hyperoxia (SpO2 99,FiO2 .6)": {"SpO2": 99, "FiO2": 0.6, "PEEP": 8},
    "ACCEPT hypercapnia (PaCO2 65,TV 300)": {"PaCO2": 65, "TV": 300},
    "hypocapnia (PaCO2 30,TV 620)": {"PaCO2": 30, "TV": 620},
    "high PEEP (18)": {"PEEP": 18},
    "low PEEP (3)": {"PEEP": 3},
    "volutrauma (TV 760)": {"TV": 760},
}


def _set(lam, alpha):
    cfg = yaml.safe_load(CFG.read_text())
    cfg["reward"]["lam_causal"] = lam
    cfg["cql"]["alpha"] = alpha
    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _decode(model, stats, ov):
    s = {**BASE, **ov}
    z = N.transform(np.array([float(s[f]) for f in TAB]), stats, TAB)
    return action_space.decode_action(model.act(z))


def run(pairs, steps=30000):
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    for lam, alpha in pairs:
        _set(lam, alpha)
        cfg_all = yaml.safe_load(CFG.read_text())
        d = T._normalise(D.load_mdp("a"), "a")
        tr = np.where(d["split"] == "train")[0]
        va = np.where(d["split"] == "val")[0]
        model, _ = T._run_training(d, tr, va, cfg_all, steps, "cpu", tag=f"[l{lam}a{alpha}] ")
        ev = np.where(np.isin(d["split"], ["val", "test"]))[0]
        sub = np.random.default_rng(0).choice(ev, size=min(20000, len(ev)), replace=False)
        acts = np.array([model.act(d["states"][i]) for i in sub])
        hold = action_space.encode_action(0, 0, 0.0)
        bm = float(np.mean(acts == d["actions"][sub]))
        hs = float(np.mean(acts == hold))
        n_distinct = len(set(int(a) for a in acts))
        print(f"\n===== lam={lam} alpha={alpha} | behaviour_match={bm:.3f} "
              f"hold={hs:.3f} distinct_actions={n_distinct} =====", flush=True)
        for name, ov in CASES.items():
            dp, dt, df = _decode(model, stats, ov)
            print(f"  {name:34s} -> dPEEP={dp:+d} dTV={dt:+4d} dFiO2={df:+.2f}", flush=True)


if __name__ == "__main__":
    pairs = [tuple(float(x) for x in a.split(":")) for a in sys.argv[1:]]
    run(pairs or [(1.0, 0.5), (1.5, 0.5), (2.5, 0.5)])
