"""Export a window of a real mimic4wdb record as an uploadable waveform.

A record in the dataset is a multi-segment WFDB set — the one used by the sample
patients is **104 files and 157 MB spanning days**. The Track B features need
minutes, so handing the whole directory to the browser is wasteful and fragile.
This trims a window into something a clinician can actually upload:

  * ``--format wfdb`` writes a **single-segment record folder** (one ``.hea`` +
    one ``.dat``, ~2 MB for 10 min). This is the dataset's own binary format, so
    nothing is transcribed and the header still declares the rate and channel
    names. Upload the folder to ``/api/waveform/extract-record``.
  * ``--format txt`` writes the documented text form (``fs:`` / ``channels:`` /
    ``signal:``), which is larger but human-readable — you can open it and see
    what it is. Upload to ``/api/waveform/extract``.

Both carry the same samples, so both extract to the same features; the script
prints them so a demo can be set up without guessing.

    # list the records that have a ventilation window to export from
    python -m backend.scripts.export_waveform --list

    # 10 minutes of the volutrauma patient, as a WFDB folder and as text
    python -m backend.scripts.export_waveform --subject 13364831 --minutes 10
    python -m backend.scripts.export_waveform --subject 13364831 --format txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from backend.pipeline import config
from backend.waveform import extract_live as X

# Offset into the ventilation window. Hour 0 is often the intubation itself,
# where the signal is full of handling artefact; a few hours in is representative.
DEFAULT_OFFSET_H = 7.0


def _cohort() -> pd.DataFrame:
    path = config.PROCESSED_PATH / "cohort_track_b.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python -m backend.pipeline.cohort_track_b` first.")
    return pd.read_csv(path, parse_dates=["vent_start", "vent_end"])


def _read_window(subject: int, record: str, start: pd.Timestamp, minutes: float):
    base = str(config.waveform_record_base(subject, record))
    hdr = wfdb.rdheader(base, rd_segments=False)
    fs = float(hdr.fs)
    rec_start = pd.Timestamp.combine(hdr.base_date, hdr.base_time)
    a = int(max(0.0, (start - rec_start).total_seconds()) * fs)
    b = min(a + int(minutes * 60 * fs), int(hdr.sig_len))
    if b <= a:
        raise SystemExit(
            f"the requested window starts past the end of record {record}")
    rec = wfdb.rdrecord(base, sampfrom=a, sampto=b)
    return rec, fs, rec_start, a, b


def _channels(rec) -> list[tuple[str, np.ndarray]]:
    """The three channels Track B uses, named as the extractor expects."""
    names = list(rec.sig_name)
    out: list[tuple[str, np.ndarray]] = []
    for lead in config.ECG_LEAD_PREFERENCE:
        if lead in names:
            # Keep the record's own lead name (II, I, …) rather than renaming it
            # to "ECG": the exported folder should be faithful to the dataset, and
            # both readers accept the lead names.
            out.append((lead, rec.p_signal[:, names.index(lead)]))
            break
    for want, label in ((config.PLETH_CHANNEL, "Pleth"), (config.RESP_CHANNEL, "Resp")):
        if want in names:
            out.append((label, rec.p_signal[:, names.index(want)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list exportable records")
    ap.add_argument("--subject", type=int, help="subject_id from --list")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--offset-hours", type=float, default=DEFAULT_OFFSET_H,
                    help="how far into the ventilation window to start")
    ap.add_argument("--format", choices=["wfdb", "txt", "both"], default="both")
    ap.add_argument("--out", help="output directory (default frontend/public/samples)")
    ap.add_argument("--name",
                    help="base name for the outputs. For the WFDB record this is "
                         "also the RECORD name, which the .hea stores internally "
                         "along with its .dat filename — so the files cannot be "
                         "renamed afterwards without rewriting the header, and "
                         "this is the only place to choose it.")
    args = ap.parse_args()

    co = _cohort()
    if args.list or not args.subject:
        print(f"{len(co)} exportable records\n")
        print(f"{'subject':>10} {'record':>10} {'stay':>10} {'overlap_h':>10}  channels")
        for r in co.itertuples():
            ch = [c for c, flag in (("ECG", r.has_ecg), ("Pleth", r.has_pleth),
                                    ("Resp", r.has_resp)) if flag]
            print(f"{int(r.subject_id):>10} {int(r.record_id):>10} "
                  f"{int(r.stay_id):>10} {r.overlap_hours:>10.1f}  {','.join(ch)}")
        if not args.subject:
            print("\nPick one with --subject N.")
        return

    ep = co[co.subject_id == args.subject]
    if ep.empty:
        raise SystemExit(f"subject {args.subject} is not in the Track B cohort")
    ep = ep.iloc[0]
    record = str(int(ep.record_id))
    start = ep.vent_start + pd.Timedelta(hours=args.offset_hours)
    if start >= ep.vent_end:
        start = ep.vent_start
        print(f"note: offset exceeds the {ep.overlap_hours:.1f} h window — "
              "starting at vent_start")

    rec, fs, rec_start, a, b = _read_window(args.subject, record, start,
                                            args.minutes)
    chans = _channels(rec)
    if not chans:
        raise SystemExit(f"record {record} has none of ECG/Pleth/Resp")
    n = min(len(v) for _, v in chans)
    # Default into the curated Tier B text-input folder: samples/ holds one
    # directory per input type and no loose files, so writing to its root would
    # recreate exactly the clutter those folders removed.
    out_dir = Path(args.out) if args.out else (
        config.REPO_ROOT / "frontend" / "public" / "samples"
        / "2-tier-b-waveform-txt")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"waveform-p{args.subject}-{record}-{int(args.minutes)}min"
    print(f"subject p{args.subject} record {record} stay {int(ep.stay_id)}")
    print(f"  window {start} + {args.minutes:g} min  ({n} samples @ {fs:.4f} Hz)")
    print(f"  channels: {', '.join(c for c, _ in chans)}")

    # --- WFDB: one single-segment record, the dataset's own format ---------- #
    if args.format in ("wfdb", "both"):
        # With an explicit --out the record is written straight there, so a
        # curated input folder holds the .hea/.dat directly rather than nesting
        # another directory inside it.
        rec_dir = out_dir if args.out else out_dir / stem
        if rec_dir == out_dir:
            rec_dir.mkdir(parents=True, exist_ok=True)
        else:
            if rec_dir.exists():
                shutil.rmtree(rec_dir)
            rec_dir.mkdir(parents=True)
        sig = np.column_stack([v[:n] for _, v in chans]).astype(np.float64)
        # Gaps would be written as a sentinel integer and read back as a real
        # value, inventing signal; carry the channel's median instead and say so.
        if not np.isfinite(sig).all():
            for j in range(sig.shape[1]):
                col = sig[:, j]
                bad = ~np.isfinite(col)
                if bad.any():
                    col[bad] = np.nanmedian(col) if np.isfinite(col).any() else 0.0
            print("  note: non-finite samples replaced with the channel median "
                  "so the binary stays readable")
        wfdb.wrsamp(record_name=stem, fs=fs,
                    units=["NU" if c in ("Pleth", "Resp") else "mV"
                           for c, _ in chans],
                    sig_name=[c for c, _ in chans], p_signal=sig,
                    fmt=["16"] * len(chans), write_dir=str(rec_dir))
        size = sum(f.stat().st_size for f in rec_dir.iterdir())
        print(f"  wrote {rec_dir}/  ({len(list(rec_dir.iterdir()))} files, "
              f"{size / 1e6:.2f} MB)")

    # --- text: human-readable ---------------------------------------------- #
    if args.format in ("txt", "both"):
        txt = out_dir / f"{stem}.txt"
        L = [
            "# VentAssist waveform file — REAL signal from MIMIC-IV-WDB.",
            f"# Source: subject p{args.subject}, record {record}, "
            f"ICU stay {int(ep.stay_id)}.",
            f"# Window: {start} + {args.minutes:g} min.",
            f"# Channels exported straight from the record: "
            f"{', '.join(c for c, _ in chans)}.",
            "#",
            "# Upload on the roster's upload card together with a patient file to",
            "# create a Track B patient: the backend runs the same extractors the",
            "# training features came from (backend/waveform/).",
            "",
            f"fs: {fs:.4f}",
            "channels: " + ",".join(c for c, _ in chans),
            "signal:",
        ]
        rows = np.column_stack([v[:n] for _, v in chans])
        L += [",".join("" if not np.isfinite(x) else f"{x:.4f}" for x in r)
              for r in rows]
        txt.write_text("\n".join(L) + "\n")
        print(f"  wrote {txt}  ({txt.stat().st_size / 1e6:.2f} MB)")

    # --- what it will extract to ------------------------------------------- #
    got = {c: v[:n] for c, v in chans}
    ecg_key = next((c for c, _ in chans if c not in ("Pleth", "Resp")), None)
    res = X.extract_features(ecg=got.get(ecg_key), pleth=got.get("Pleth"),
                             resp=got.get("Resp"), fs=fs)
    print(f"\n  extracts to {int(res.coverage * 6)}/6 features "
          f"({res.coverage:.0%} coverage):")
    for k, v in res.features.items():
        print(f"    {k:22s} {'—' if v is None else f'{v:.4f}'}")
    for w in res.warnings:
        print(f"    warning: {w}")


if __name__ == "__main__":
    main()
