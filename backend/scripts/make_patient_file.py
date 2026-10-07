"""Write an upload-ready patient file from a REAL Track B (waveform) episode.

The dashboard's upload accepts a ``key: value`` text file, NOT raw WFDB signal —
the 6 waveform features must already be extracted. This script emits such a file
straight from the pipeline's own artifacts, so every clinical and waveform value
in it is real: the 12 clinical fields come from chartevents/labevents, and the
waveform features were computed from that record's actual ECG / Pleth / Resp
signal by ``backend/waveform/``.

It also prints the recommendation the file will produce, so a demo can be set up
without guessing, and it marks any blood gas that is the population median
``gp_imputation`` falls back to when an episode has no ABG in that hour.

    # list the real episodes available, with the action each one produces
    python -m backend.scripts.make_patient_file --list

    # write one (index from --list); lands in samples/1-tier-a-clinical-only/
    python -m backend.scripts.make_patient_file --index 36 --name "Patient W1"

    # only the rows where the policy actually changes a setting
    python -m backend.scripts.make_patient_file --list --acting-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.pipeline import config
from backend.router import safety_filter as SF
from backend.router.policy_router import PolicyRouter, WAVEFORM

TABULAR = list(config.TABULAR_FEATURES)
# Values equal to these are gp_imputation's population-median fallback (used when
# the episode has no arterial blood gas in the hour), not measurements.
MEDIAN_FALLBACK = {"PaO2": 100.0, "PaCO2": 41.0, "pH": 7.37}
INT_FIELDS = {"TV", "SpO2", "HR", "SBP", "RR", "RASS", "PaO2", "PaCO2"}


def _rows() -> pd.DataFrame:
    """Real (clinical state, extracted waveform, patient) rows, complete only."""
    need = [config.PROCESSED_PATH / n for n in
            ("waveform_features_track_b.parquet", "tabular_states_track_b.parquet",
             "cohort_track_b.csv")]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Track B artifacts missing: {missing}. Run "
                         "`python -m backend.scripts.run_track_b_pipeline` first.")
    wv = pd.read_parquet(need[0])
    st = pd.read_parquet(need[1])
    co = pd.read_csv(need[2])
    df = st.merge(wv, on=["stay_id", "hour"]).merge(
        co[["stay_id", "record_id", "subject_id", "age", "weight_kg"]], on="stay_id")
    return df.dropna(subset=WAVEFORM + TABULAR).reset_index(drop=True)


def _gender(subject_id: int) -> str:
    """Real recorded gender — the parser coerces anything not starting with F to M."""
    try:
        pat = pd.read_csv(config.PATIENTS, usecols=["subject_id", "gender"])
        g = pat.set_index("subject_id")["gender"].get(subject_id)
        return "F" if str(g).upper().startswith("F") else "M"
    except Exception:
        return "M"


def _fmt(field: str, v: float) -> str:
    if field in ("FiO2", "pH"):
        return f"{v:.2f}"
    if field == "Temp":
        return f"{v:.1f}"
    if field in INT_FIELDS:
        return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
    return f"{v:g}"


def _action_text(dp: int, dt: int, df: float) -> str:
    """Same wording the API route produces, so the file's note matches the UI."""
    def part(name, v, unit, prec=0):
        if v == 0:
            return f"Hold {name}"
        amt = f"{abs(v):.{prec}f}" if prec else f"{abs(v)}"
        return f"{'Increase' if v > 0 else 'Decrease'} {name} by {amt}" \
               f"{f' {unit}' if unit else ''}"
    return (f"{part('PEEP', dp, 'cmH2O')} · {part('tidal volume', dt, 'mL')} · "
            f"{part('FiO2', df, '', prec=2)}")


def _evaluate(router: PolicyRouter, row: pd.Series) -> tuple[dict, object]:
    state = {f: float(row[f]) for f in TABULAR}
    wave = {f: float(row[f]) for f in WAVEFORM}
    out = router.run_track_b(state, wave, ventilation_mode="volume_control")
    out["action_text"] = _action_text(out["delta_PEEP"], out["delta_TV"],
                                      out["delta_FiO2"])
    chk = SF.check(state, out["delta_PEEP"], out["delta_TV"], out["delta_FiO2"],
                   float(row["weight_kg"]), wave["Arrhythmia_rate"])
    return out, chk


def write_file(row: pd.Series, out: dict, chk, name: str, bed: str,
               summary: str, dest) -> None:
    state = {f: float(row[f]) for f in TABULAR}
    wave = {f: float(row[f]) for f in WAVEFORM}
    imputed = [f for f, m in MEDIAN_FALLBACK.items() if abs(state[f] - m) < 1e-9]

    L = [
        f"# {summary}",
        f"# Source: MIMIC-IV-WDB subject p{int(row.subject_id)}, record "
        f"{int(row.record_id)}, ICU stay {int(row.stay_id)}, hour {int(row.hour)}.",
        "# Every value below is REAL: the 12 clinical fields come from "
        "chartevents/labevents",
        "# via the VentAssist pipeline, and the 6 waveform features were extracted "
        "from that",
        "# record's actual ECG / Pleth / Resp signal by backend/waveform/ (1-hour window).",
        "#",
        f"# Expected recommendation: {out['action_text']}",
        f"# Expected confidence: {out['confidence']:.3f}",
    ]
    if chk.flags:
        L.append("# Expected safety flags:")
        L += [f"#   {d['level']}: {d['message']}" for d in chk.flags]
    else:
        L.append("# Expected safety flags: none")
    L += [
        "#",
        "# The waveform block selects Track B and feeds Arrhythmia_rate to the safety",
        "# filter, but it will NOT change the recommendation: the trained model assigns",
        "# the waveform dims zero weight, so Track B reports the same action and the same",
        "# confidence as Track A and labels itself 'waveform recorded, not yet influencing'.",
        "",
        f"name: {name}",
        f"age: {int(row.age)}",
        f"sex: {_gender(int(row.subject_id))}",
        f"bed: {bed}",
        f"summary: {summary}",
        "",
        f"weight: {float(row.weight_kg):g}",
    ]
    if imputed:
        L += [f"# {', '.join(imputed)}: population median that gp_imputation falls back",
              "# to when the episode has no arterial blood gas this hour — imputed, not",
              "# measured. The remaining values are charted measurements."]
    L.append("")
    L += [f"{f}: {_fmt(f, state[f])}" for f in TABULAR]
    L += ["", "ventilation_mode: volume_control", "track: track_b", "",
          "# --- waveform features, extracted from the real record ---"]
    L += [f"{f}: {wave[f]:.4g}" for f in WAVEFORM]
    dest.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list episodes and exit")
    ap.add_argument("--acting-only", action="store_true",
                    help="with --list, show only rows where the policy changes a setting")
    ap.add_argument("--index", type=int, help="row index from --list")
    ap.add_argument("--name", default="Uploaded Patient")
    ap.add_argument("--bed", default="ICU-W0")
    ap.add_argument("--summary", default="Real mimic4wdb waveform episode")
    ap.add_argument("--out", help="output path (default frontend/public/samples/<slug>.txt)")
    args = ap.parse_args()

    df = _rows()
    router = PolicyRouter()
    if router.track_b is None:
        raise SystemExit("Track B model not available — train it first.")

    if args.list or args.index is None:
        print(f"{len(df)} complete real (state, waveform) rows\n")
        print(f"{'idx':>4} {'stay':>10} {'record':>10} {'hr':>3} "
              f"{'PEEP':>5} {'TV':>5} {'FiO2':>5} {'SpO2':>5} {'arr':>5}  recommendation")
        for i, row in df.iterrows():
            out, _ = _evaluate(router, row)
            acting = out["delta_PEEP"] or out["delta_TV"] or out["delta_FiO2"]
            if args.acting_only and not acting:
                continue
            print(f"{i:>4} {int(row.stay_id):>10} {int(row.record_id):>10} "
                  f"{int(row.hour):>3} {row.PEEP:>5.1f} {row.TV:>5.0f} {row.FiO2:>5.2f} "
                  f"{row.SpO2:>5.1f} {row.Arrhythmia_rate:>5.2f}  "
                  f"dPEEP={out['delta_PEEP']:+d} dTV={out['delta_TV']:+d} "
                  f"dFiO2={out['delta_FiO2']:+.2f}")
        if args.index is None:
            print("\nPick one with --index N to write its patient file.")
        return

    if not 0 <= args.index < len(df):
        raise SystemExit(f"--index must be in [0, {len(df) - 1}]")
    row = df.iloc[args.index]
    out, chk = _evaluate(router, row)
    slug = args.name.lower().replace(" ", "-")
    if slug.startswith("patient-"):          # avoid patient-patient-w4.txt
        slug = slug[len("patient-"):]
    # Default into the Tier A input folder for the same reason as above.
    dest = (Path(args.out) if args.out
            else config.REPO_ROOT / "frontend" / "public" / "samples"
            / "1-tier-a-clinical-only" / f"patient-{slug}.txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_file(row, out, chk, args.name, args.bed, args.summary, dest)
    print(f"wrote {dest}")
    print(f"  source      : subject p{int(row.subject_id)} record {int(row.record_id)} "
          f"stay {int(row.stay_id)} hour {int(row.hour)}")
    print(f"  expect      : {out['action_text']}")
    print(f"  confidence  : {out['confidence']:.3f}  (informative="
          f"{out['waveform_informative']})")
    print(f"  safety      : all_clear={chk.all_clear}")
    for d in chk.flags:
        print(f"     {d['level']}: {d['message']}")


if __name__ == "__main__":
    main()
