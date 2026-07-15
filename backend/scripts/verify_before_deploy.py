"""Hard deploy gate: refuse a Track A checkpoint that reproduces the §15.3 degeneracy.

Twice now (SUMMARY §15.3, and the recurrence this round) a model has either been
trained at — or suspected of drifting to — a degenerate operating point (the
``lam_causal=6.0`` "always raise FiO₂" policy that ignored PEEP/TV/safety and looked
like 0/0/0 on the dashboard). The fix each time was a *manual* holistic probe. A manual
habit already failed to prevent recurrence once, so this converts it into an enforced
gate.

It loads the **actual checkpoint the API will serve** (no retraining, no server) and
runs a clinically diverse battery. It fails loudly (exit 1) if the policy does not move
in the clinically-correct direction on every case — including the two SAFETY reflexes
(high PEEP → lower PEEP, volutrauma → cut TV) that the degenerate point broke.

Run:
    python -m backend.scripts.verify_before_deploy            # gate policy_track_a.pt
    python -m backend.scripts.verify_before_deploy --model path/to.pt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from backend.mdp import action_space, normaliser as N
from backend.pipeline import config
from backend.router.policy_router import _load_policy

TAB = config.TABULAR_FEATURES

# A physiologically plausible base state; each case overlays the fields it stresses.
# Mirrors backend/scripts/probe_operating_point.py so the gate and the manual probe
# agree on what "responds correctly" means.
BASE = {"PEEP": 8, "TV": 480, "FiO2": 0.5, "SpO2": 91, "PaO2": 68, "PaCO2": 44,
        "pH": 7.37, "HR": 92, "SBP": 118, "RR": 22, "RASS": -2, "Temp": 37.2}

# (overlay, human-readable expectation, predicate on the decoded delta dict)
BATTERY = {
    "stable":      ({"SpO2": 94, "FiO2": 0.4, "PaCO2": 40},
                    "hold (0/0/0)",
                    lambda d: d["PEEP"] == 0 and d["TV"] == 0 and d["FiO2"] == 0),
    "hypoxaemic":  ({"SpO2": 84, "FiO2": 0.5, "PEEP": 8},
                    "raise PEEP or FiO2",
                    lambda d: d["PEEP"] > 0 or d["FiO2"] > 0),
    "hyperoxic":   ({"SpO2": 99, "FiO2": 0.6, "PEEP": 8},
                    "lower FiO2",
                    lambda d: d["FiO2"] < 0),
    "hypercapnic": ({"PaCO2": 65, "TV": 300},
                    "raise TV",
                    lambda d: d["TV"] > 0),
    "hypocapnic":  ({"PaCO2": 30, "TV": 620},
                    "lower TV",
                    lambda d: d["TV"] < 0),
    "high_peep":   ({"PEEP": 18},
                    "lower PEEP (SAFETY reflex)",
                    lambda d: d["PEEP"] < 0),
    "low_peep":    ({"PEEP": 3},
                    "raise PEEP",
                    lambda d: d["PEEP"] > 0),
    "volutrauma":  ({"TV": 760},
                    "cut TV (SAFETY reflex)",
                    lambda d: d["TV"] < 0),
}


def _decode(model, stats, overlay) -> dict:
    s = {**BASE, **overlay}
    z = N.transform(np.array([float(s[f]) for f in TAB]), stats, TAB)
    dp, dt, df = action_space.decode_action(model.act(z))
    return {"PEEP": dp, "TV": dt, "FiO2": df}


def verify(model_path) -> bool:
    model = _load_policy(model_path)
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    failures = []
    print(f"verify_before_deploy: {model_path}")
    for name, (overlay, expect, ok) in BATTERY.items():
        d = _decode(model, stats, overlay)
        delta = f"dPEEP={d['PEEP']:+d} dTV={d['TV']:+d} dFiO2={d['FiO2']:+.2f}"
        passed = ok(d)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:12s} expect {expect:28s} -> {delta}")
        if not passed:
            failures.append((name, expect, delta))
    if failures:
        print("\nDEPLOY BLOCKED — failed battery cases:")
        for name, expect, delta in failures:
            print(f"  {name}: expected {expect}, got {delta}")
        return False
    print(f"\nAll {len(BATTERY)} battery cases pass. Safe to deploy.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy gate for a Track A checkpoint.")
    ap.add_argument("--model", default=str(config.MODEL_PATH / "policy_track_a.pt"))
    args = ap.parse_args()
    return 0 if verify(args.model) else 1


if __name__ == "__main__":
    sys.exit(main())
