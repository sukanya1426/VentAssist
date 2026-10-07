"""Extract the 6 Track B features from a raw waveform at SERVING time.

The Track B goal is "upload a waveform, get a recommendation". Everything needed
to compute the features already existed in this package — but only reachable from
the offline pipeline, where [aggregator.py](aggregator.py) walks a cohort parquet
and reads WFDB records off disk. Nothing could turn an uploaded signal into the
6 numbers the 18-dim state needs, so a clinician had to compute HRV, perfusion
index and the rest by hand and type them into the patient file. This module is
the missing step.

It deliberately reuses ``preprocessor`` / ``ecg_features`` / ``pleth_features`` /
``resp_features`` through the same call shape ``aggregator._hour_features`` uses:
filter the channel, gate on its quality flag and finite fraction, then extract.
A served feature must be computed the same way as the features the policy was
trained on, or the 18-dim state is in different units than the model expects.

Two differences from the training path, both reported rather than hidden:

  * **Window length.** Training used exactly 1 hour per row. An upload is
    whatever the clinician recorded, so ``duration_s`` comes back in the result
    and short windows are flagged. HRV over 60 s is not HRV over an hour; it is
    still the same estimator, on less data.
  * **Missing channels.** The training aggregator emitted NaN for a channel that
    was absent or failed its quality gate, and the row was then dropped. Here a
    NaN is passed through to the router, which imputes it and reports reduced
    waveform coverage — the clinician gets a recommendation plus an honest
    statement of what was usable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from backend.waveform import ecg_features as E
from backend.waveform import pleth_features as PL
from backend.waveform import preprocessor as P
from backend.waveform import resp_features as R

FEATURES = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
            "RRV", "Breathing_Regularity", "Asynchrony_Score"]

# Same gate as the training aggregator: a channel must be ≥80% finite before its
# features are trusted, so an upload with long dropouts fails the same way the
# cohort rows did rather than quietly producing a confident number.
MIN_VALID_FRACTION = 0.80

# Below this the extractors are running on very few cycles — ECG needs ≥5 R-peaks,
# Pleth ≥10 pulses, Resp ≥10 breaths. 30 s at 62.5 Hz clears all three for a
# normal adult, so it is the point at which a warning rather than a refusal is right.
SHORT_WINDOW_S = 30.0

# A sampling rate far from the training data's ~62.5 Hz changes what the filters
# and peak detectors see; the band-pass is capped just below Nyquist, so a very
# low rate silently narrows it.
TRAINING_FS = 62.5


@dataclass
class ChannelReport:
    present: bool
    valid_fraction: Optional[float] = None
    quality_ok: Optional[bool] = None
    note: Optional[str] = None


@dataclass
class ExtractionResult:
    features: dict                      # the 6, any of which may be None
    fs: float
    n_samples: int
    duration_s: float
    channels: dict = field(default_factory=dict)   # name → ChannelReport
    warnings: list = field(default_factory=list)

    @property
    def coverage(self) -> float:
        got = sum(1 for f in FEATURES if self.features.get(f) is not None)
        return got / len(FEATURES)


def _clean(x) -> Optional[np.ndarray]:
    """Coerce one channel to a float array, or None if it carries no signal."""
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    return arr


def _extract(channel: str, x: Optional[np.ndarray], fs: float, extractor,
             keys: list[str]) -> tuple[dict, ChannelReport]:
    """Preprocess + gate + extract one channel, mirroring aggregator._hour_features."""
    nan = {k: None for k in keys}
    if x is None:
        return nan, ChannelReport(present=False, note="channel not supplied")
    if len(x) <= fs:
        return nan, ChannelReport(present=True, note="under one second of signal")

    pc = P.preprocess_hour(channel, x, fs)
    feats = extractor(pc.signal, fs)
    report = ChannelReport(present=True, valid_fraction=round(pc.valid_fraction, 4),
                           quality_ok=bool(pc.quality_ok))
    if not pc.quality_ok:
        report.note = f"failed the {channel} quality check (flatline/motion/disconnect)"
        return nan, report
    if pc.valid_fraction < MIN_VALID_FRACTION:
        report.note = (f"only {pc.valid_fraction:.0%} of the window is finite "
                       f"(need {MIN_VALID_FRACTION:.0%})")
        return nan, report
    # An extractor returns NaN when it could not find enough cycles; carry that
    # through as None rather than letting NaN reach JSON.
    out = {k: (None if not np.isfinite(feats.get(k, np.nan)) else float(feats[k]))
           for k in keys}
    if all(v is None for v in out.values()):
        report.note = "too few detectable cycles to compute a feature"
    return out, report


def extract_features(ecg=None, pleth=None, resp=None,
                     fs: float = TRAINING_FS) -> ExtractionResult:
    """Raw signal → the 6 Track B features, with a per-channel quality report.

    Each channel is independent: supplying only Pleth yields Perfusion_Index and
    leaves the other five None, which the router treats as partial coverage.
    """
    if not fs or fs <= 0:
        raise ValueError("fs must be a positive sampling rate in Hz")

    ecg_a, pleth_a, resp_a = _clean(ecg), _clean(pleth), _clean(resp)
    lengths = [len(a) for a in (ecg_a, pleth_a, resp_a) if a is not None]
    if not lengths:
        raise ValueError(
            "no usable signal — supply at least one of ECG, Pleth or Resp")
    n = max(lengths)
    duration = n / float(fs)

    features: dict = {}
    channels: dict = {}
    for name, arr, extractor, keys in (
        ("ECG", ecg_a, E.ecg_features, ["HRV_SDNN", "Arrhythmia_rate"]),
        ("Pleth", pleth_a, PL.pleth_features, ["Perfusion_Index"]),
        ("Resp", resp_a, R.resp_features,
         ["RRV", "Breathing_Regularity", "Asynchrony_Score"]),
    ):
        got, report = _extract(name, arr, fs, extractor, keys)
        features.update(got)
        channels[name] = report

    warnings: list[str] = []
    if duration < SHORT_WINDOW_S:
        warnings.append(
            f"window is {duration:.0f}s; the features were trained on 1-hour "
            f"windows and the extractors need ~{SHORT_WINDOW_S:.0f}s of signal to "
            "find enough cardiac and respiratory cycles")
    if abs(fs - TRAINING_FS) / TRAINING_FS > 0.5:
        warnings.append(
            f"sampling rate {fs:g} Hz is far from the ~{TRAINING_FS:g} Hz the "
            "features were trained on, so filter bands and peak detection differ")
    for name, rep in channels.items():
        if rep.note:
            warnings.append(f"{name}: {rep.note}")

    return ExtractionResult(features=features, fs=float(fs), n_samples=int(n),
                            duration_s=round(duration, 2), channels=channels,
                            warnings=warnings)


# --------------------------------------------------------------------------- #
# Text/CSV waveform files
# --------------------------------------------------------------------------- #
def parse_waveform_text(text: str) -> dict:
    """Parse an uploaded waveform file into channel arrays + the sampling rate.

    The format is deliberately the same shape as the Tier A patient file — ``key:
    value`` headers, ``#`` comments — followed by sample rows, so a clinician can
    open it, read the header and see what it is:

        fs: 62.5
        channels: ECG,Pleth,Resp
        signal:
        -0.125,0.512,0.031
        ...

    A bare CSV with a header row naming the channels also works, with ``fs``
    defaulting to the training rate. Column names are matched case-insensitively
    and the ECG lead may be named by its lead (``II``, ``I``, ``III``, ``V``,
    ``aVR``) as WFDB does, so a channel exported straight from a record is
    recognised without renaming.
    """
    from backend.pipeline import config

    fs: Optional[float] = None
    names: Optional[list[str]] = None
    rows: list[list[float]] = []
    in_signal = False

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not in_signal and ":" in line and not _looks_numeric(line):
            key, _, val = line.partition(":")
            k, v = key.strip().lower(), val.strip()
            if k == "fs" and v:
                fs = float(v)
            elif k in ("channels", "signals", "columns") and v:
                names = [c.strip() for c in v.replace("\t", ",").split(",") if c.strip()]
            elif k == "signal" and not v:
                in_signal = True
            continue
        parts = [p for p in line.replace("\t", ",").split(",") if p.strip() != ""]
        if names is None and not _looks_numeric(line):
            names = [p.strip() for p in parts]        # bare-CSV header row
            continue
        in_signal = True
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue                                   # skip an unparseable row

    if not rows:
        raise ValueError("no sample rows found in the waveform file")
    if names is None:
        raise ValueError(
            "the waveform file does not say which channels the columns are — add a "
            "'channels: ECG,Pleth,Resp' header")

    width = max(len(r) for r in rows)
    if len(names) != width:
        raise ValueError(
            f"{len(names)} channel name(s) but {width} column(s) of samples")
    arr = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    ecg_aliases = {"ecg", *(l.lower() for l in config.ECG_LEAD_PREFERENCE)}
    out: dict = {"fs": fs or TRAINING_FS, "ecg": None, "pleth": None, "resp": None}
    for i, name in enumerate(names):
        low = name.strip().lower()
        col = arr[:, i]
        if low in ecg_aliases and out["ecg"] is None:
            out["ecg"] = col
        elif low.startswith("pleth") and out["pleth"] is None:
            out["pleth"] = col
        elif low.startswith("resp") and out["resp"] is None:
            out["resp"] = col
    if out["ecg"] is None and out["pleth"] is None and out["resp"] is None:
        raise ValueError(
            f"none of the columns {names} is an ECG, Pleth or Resp channel")
    return out


# --------------------------------------------------------------------------- #
# WFDB records (.hea + .dat) — the dataset's own format
# --------------------------------------------------------------------------- #
def extract_from_wfdb_dir(directory, record_name: Optional[str] = None,
                          max_seconds: float = 900.0) -> ExtractionResult:
    """Read a WFDB record from a directory of ``.hea``/``.dat`` files and extract.

    This is the dataset's native format, so a clinician can hand over a trimmed
    record folder instead of transcribing samples into text. The channel naming
    and lead preference are the same ones ``loader.py`` uses on the full cohort
    records, so a folder exported from ``files/mimic4wdb`` behaves identically
    here and in the training pipeline.

    ``record_name`` defaults to the only ``.hea`` that is not a segment header —
    a multi-segment record has a master header plus one per segment, and reading
    a segment header alone would give a few seconds instead of the whole window.

    ``max_seconds`` caps how much signal is read. A full mimic4wdb record is days
    long and 157 MB across ~100 files; the features only need minutes, so a long
    record is truncated from its start rather than refused.
    """
    import wfdb
    from backend.pipeline import config

    d = Path(directory)
    heas = sorted(d.glob("*.hea"))
    if not heas:
        raise ValueError("no .hea header found — a WFDB record needs its header file")

    if record_name is None:
        # Segment headers are "<record>_NNNN.hea"; the master header is not.
        masters = [h for h in heas if not re.match(r".*_\d{4}[a-z]?$", h.stem)]
        if not masters:
            masters = heas                        # single-segment upload
        if len(masters) > 1:
            raise ValueError(
                f"{len(masters)} records in this folder ({', '.join(h.stem for h in masters)})"
                " — upload one record at a time")
        record_name = masters[0].stem

    base = str(d / record_name)
    try:
        hdr = wfdb.rdheader(base, rd_segments=False)
    except Exception as e:
        raise ValueError(f"could not read the WFDB header {record_name}.hea: {e}")
    fs = float(getattr(hdr, "fs", 0) or 0)
    if fs <= 0:
        raise ValueError(f"{record_name}.hea does not declare a usable sampling rate")

    n_total = int(getattr(hdr, "sig_len", 0) or 0)
    sampto = min(n_total, int(max_seconds * fs)) if n_total else int(max_seconds * fs)
    try:
        rec = wfdb.rdrecord(base, sampfrom=0, sampto=sampto or None)
    except Exception as e:
        raise ValueError(
            f"could not read the signal for {record_name}: {e}. A multi-segment "
            "record needs every .dat file its header lists to be in the folder.")

    names = list(rec.sig_name or [])
    sig = rec.p_signal
    if sig is None or not names:
        raise ValueError(f"{record_name} carries no readable signal")

    def pick(wanted) -> Optional[np.ndarray]:
        for w in wanted:
            if w in names:
                return sig[:, names.index(w)].astype(np.float64)
        return None

    # Lead names first (a dataset record names the channel "II"/"I"/…), then the
    # generic "ECG" that an exported or hand-built record uses. Same alias set the
    # text parser accepts, so both upload formats behave identically.
    ecg = pick([*config.ECG_LEAD_PREFERENCE, "ECG"])
    pleth = pick([config.PLETH_CHANNEL])
    resp = pick([config.RESP_CHANNEL])
    if ecg is None and pleth is None and resp is None:
        raise ValueError(
            f"{record_name} has channels {names} — none is an ECG lead "
            f"({'/'.join(config.ECG_LEAD_PREFERENCE)} or ECG), "
            f"{config.PLETH_CHANNEL} or {config.RESP_CHANNEL}")

    res = extract_features(ecg=ecg, pleth=pleth, resp=resp, fs=fs)
    if n_total and sampto < n_total:
        res.warnings.append(
            f"record is {n_total / fs / 3600:.1f} h long; only the first "
            f"{sampto / fs / 60:.0f} min were read")
    return res


def _looks_numeric(line: str) -> bool:
    first = line.replace("\t", ",").split(",")[0].strip()
    try:
        float(first)
        return True
    except ValueError:
        return False
