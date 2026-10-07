"""Tests for the Track B serving-time waveform extraction.

This is the step that made "upload a waveform, get a recommendation" possible.
Before it, the 6 features existed only in the offline pipeline, so Track B could
only be used by hand-computing HRV / perfusion index / RRV and typing them into
the patient file.

The property that matters most is that a SERVED feature equals the feature the
training pipeline would have produced from the same samples — otherwise the
18-dim state is in units the policy was never fitted on. That is asserted
directly against ``aggregator._hour_features``, the training call path.

Run:  pytest backend/tests/test_waveform_extract.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.pipeline import config
from backend.waveform import extract_live as X

FS = 62.5
# The curated upload folders a clinician is told to use — one per input type.
# Tests assert against these directly rather than globbing, so moving or
# renaming a shipped input fails loudly instead of silently skipping.
SAMPLES = config.REPO_ROOT / "frontend" / "public" / "samples"


class Skip(Exception):
    pass


def _synth(seconds: float = 90.0, fs: float = FS, hr_bpm: float = 72.0,
           rr_bpm: float = 15.0, seed: int = 0) -> dict:
    """A crude but detectable ECG / Pleth / Resp triple.

    Not physiological — just periodic enough that the R-peak, pulse and breath
    detectors find the number of cycles they require, so the tests exercise the
    real extractors rather than their NaN fallbacks.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    beat = hr_bpm / 60.0
    # Narrow spikes at the beat rate read as R-peaks after band-pass filtering.
    phase = (t * beat) % 1.0
    ecg = np.where(phase < 0.04, 1.0, 0.0) + 0.02 * rng.standard_normal(n)
    # Smooth pulsatile wave with a DC offset, so AC/DC is a finite ratio.
    pleth = 1.0 + 0.25 * np.sin(2 * np.pi * beat * t) + 0.01 * rng.standard_normal(n)
    resp = np.sin(2 * np.pi * (rr_bpm / 60.0) * t) + 0.01 * rng.standard_normal(n)
    return {"ecg": ecg, "pleth": pleth, "resp": resp, "fs": fs}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_all_six_features_from_three_channels():
    s = _synth()
    res = X.extract_features(s["ecg"], s["pleth"], s["resp"], fs=s["fs"])
    assert set(res.features) == set(X.FEATURES)
    got = [f for f in X.FEATURES if res.features[f] is not None]
    assert len(got) == 6, f"only {got} extracted from a clean 3-channel window"
    assert res.coverage == pytest.approx(1.0)
    assert res.duration_s == pytest.approx(90.0, abs=0.1)
    assert not res.warnings, res.warnings


def test_features_land_in_the_api_accepted_ranges():
    """A served feature must satisfy the same bounds the API enforces.

    If an extractor can emit a value the request model rejects, the upload path
    produces a 422 the clinician cannot act on.
    """
    bounds = {"HRV_SDNN": (0, 500), "Arrhythmia_rate": (0, 1),
              "Perfusion_Index": (0, 30), "RRV": (0, 10),
              "Breathing_Regularity": (0, 1), "Asynchrony_Score": (0, 1)}
    s = _synth()
    res = X.extract_features(s["ecg"], s["pleth"], s["resp"], fs=s["fs"])
    for f, (lo, hi) in bounds.items():
        v = res.features[f]
        if v is not None:
            assert lo <= v <= hi, f"{f}={v} outside the API's [{lo}, {hi}]"


def test_channels_are_independent():
    """Pleth alone yields its feature and leaves the rest absent."""
    s = _synth()
    res = X.extract_features(pleth=s["pleth"], fs=s["fs"])
    assert res.features["Perfusion_Index"] is not None
    for f in ("HRV_SDNN", "Arrhythmia_rate", "RRV", "Breathing_Regularity"):
        assert res.features[f] is None
    assert res.coverage == pytest.approx(1 / 6)
    assert res.channels["ECG"].present is False
    assert res.channels["Resp"].present is False


def test_no_signal_at_all_is_an_error():
    with pytest.raises(ValueError):
        X.extract_features(fs=FS)
    with pytest.raises(ValueError):
        X.extract_features(ecg=[], fs=FS)
    # An all-NaN channel carries no signal either.
    with pytest.raises(ValueError):
        X.extract_features(ecg=[float("nan")] * 100, fs=FS)


def test_bad_sampling_rate_is_rejected():
    s = _synth()
    for fs in (0, -1):
        with pytest.raises(ValueError):
            X.extract_features(s["ecg"], fs=fs)


def test_short_window_warns_but_still_extracts():
    s = _synth(seconds=12.0)
    res = X.extract_features(s["ecg"], s["pleth"], s["resp"], fs=s["fs"])
    assert any("window is" in w for w in res.warnings), res.warnings


def test_flatline_channel_is_gated_not_trusted():
    """A dead channel must yield no feature, with the reason reported.

    Quality gating is the same one the training aggregator applied; without it a
    flat ECG would produce a confident, meaningless HRV.
    """
    s = _synth()
    res = X.extract_features(np.zeros(len(s["ecg"])), s["pleth"], s["resp"], fs=s["fs"])
    assert res.features["HRV_SDNN"] is None
    assert res.channels["ECG"].quality_ok is False or res.channels["ECG"].note
    assert any("ECG" in w for w in res.warnings)


def test_never_returns_nan():
    """NaN does not survive JSON — absent features must be None, not NaN."""
    res = X.extract_features(np.zeros(2000), fs=FS)
    for f, v in res.features.items():
        assert v is None or np.isfinite(v), f"{f} is NaN"


# --------------------------------------------------------------------------- #
# Serving must agree with training
# --------------------------------------------------------------------------- #
def test_serving_matches_the_training_extraction_path():
    """The core guarantee: same samples in, same numbers out as the pipeline.

    ``aggregator._hour_features`` is what produced every feature the policy was
    trained on. If this drifts, the served 18-dim state is in different units
    than the model's, and no test of the router would catch it.
    """
    from backend.waveform import aggregator as A
    from backend.waveform import ecg_features as E
    from backend.waveform import pleth_features as PL
    from backend.waveform import resp_features as R

    s = _synth()
    served = X.extract_features(s["ecg"], s["pleth"], s["resp"], fs=s["fs"]).features
    trained = {}
    trained.update(A._hour_features("ECG", s["ecg"], s["fs"], E.ecg_features))
    trained.update(A._hour_features("Pleth", s["pleth"], s["fs"], PL.pleth_features))
    trained.update(A._hour_features("Resp", s["resp"], s["fs"], R.resp_features))

    for f in X.FEATURES:
        t = trained[f]
        v = served[f]
        if not np.isfinite(t):
            assert v is None, f"{f}: training said NaN, serving said {v}"
        else:
            assert v == pytest.approx(float(t)), \
                f"{f}: serving {v} != training {t}"


# --------------------------------------------------------------------------- #
# The uploaded file format
# --------------------------------------------------------------------------- #
def test_parses_the_documented_header_format():
    text = ("# a comment\n"
            "fs: 62.5\n"
            "channels: ECG,Pleth,Resp\n"
            "signal:\n"
            "0.1,0.5,0.2\n"
            "0.2,0.6,0.3\n")
    p = X.parse_waveform_text(text)
    assert p["fs"] == pytest.approx(62.5)
    assert len(p["ecg"]) == 2 and len(p["pleth"]) == 2 and len(p["resp"]) == 2
    assert p["ecg"][1] == pytest.approx(0.2)


def test_parses_a_bare_csv_with_a_channel_header():
    p = X.parse_waveform_text("ECG,Pleth,Resp\n0.1,0.5,0.2\n0.2,0.6,0.3\n")
    assert p["fs"] == pytest.approx(X.TRAINING_FS)       # defaults to training rate
    assert len(p["ecg"]) == 2


def test_wfdb_lead_names_are_recognised_as_ecg():
    """A column exported straight from a record is named by its lead, not 'ECG'."""
    for lead in config.ECG_LEAD_PREFERENCE:
        p = X.parse_waveform_text(f"fs: 62.5\nchannels: {lead},Pleth\n0.1,0.5\n0.2,0.6\n")
        assert p["ecg"] is not None, f"lead {lead} not recognised as ECG"
        assert p["pleth"] is not None


def test_missing_channel_header_is_an_error():
    with pytest.raises(ValueError, match="channels"):
        X.parse_waveform_text("fs: 62.5\n0.1,0.5,0.2\n0.2,0.6,0.3\n")


def test_channel_count_mismatch_is_an_error():
    with pytest.raises(ValueError, match="column"):
        X.parse_waveform_text("fs: 62.5\nchannels: ECG,Pleth\n0.1,0.5,0.9\n0.2,0.6,0.8\n")


def test_no_samples_is_an_error():
    with pytest.raises(ValueError, match="no sample rows"):
        X.parse_waveform_text("fs: 62.5\nchannels: ECG,Pleth,Resp\nsignal:\n")


def test_unrecognised_columns_are_an_error():
    with pytest.raises(ValueError, match="ECG, Pleth or Resp"):
        X.parse_waveform_text("fs: 62.5\nchannels: ABP,CVP\n1,2\n3,4\n")


def test_blank_cells_become_gaps_not_zeros():
    """An empty cell is missing signal; reading it as 0.0 would invent a flatline."""
    p = X.parse_waveform_text(
        "fs: 62.5\nchannels: ECG,Pleth\n0.1,0.5\n0.2\n0.3,0.7\n")
    assert np.isnan(p["pleth"][1]), "a missing cell must be NaN, not 0.0"


# --------------------------------------------------------------------------- #
# The real shipped sample file
# --------------------------------------------------------------------------- #
def test_the_shipped_real_waveform_file_extracts_all_six():
    """The file a clinician is told to upload must actually work."""
    path = SAMPLES / "2-tier-b-waveform-txt" / "waveform-10min.txt"
    if not path.exists():
        pytest.skip(f"{path} not present")
    p = X.parse_waveform_text(path.read_text())
    res = X.extract_features(p["ecg"], p["pleth"], p["resp"], fs=p["fs"])
    assert res.coverage == pytest.approx(1.0), \
        f"shipped sample only yielded {res.features}"
    assert res.duration_s > 60
    assert not res.warnings, res.warnings


# --------------------------------------------------------------------------- #
# WFDB record folders — the dataset's own format
# --------------------------------------------------------------------------- #
def _shipped_record_dir():
    d = SAMPLES / "3-tier-b-wfdb-record"
    if not (d / "waveform-10min.hea").exists():
        pytest.skip(f"{d} does not hold an exported record")
    return d


def test_wfdb_record_folder_extracts_all_six():
    d = _shipped_record_dir()
    res = X.extract_from_wfdb_dir(d)
    assert res.coverage == pytest.approx(1.0), f"only got {res.features}"
    assert res.duration_s > 60
    assert res.fs > 0


def test_wfdb_and_text_formats_agree():
    """The two upload formats must read the same samples the same way.

    Built as a matched pair here rather than read from the sample folders: those
    now hold a DIFFERENT patient per folder (so that uploading each gives a
    visibly different recommendation), which makes them unsuitable for an
    equality check. The property under test is the readers, not the samples.

    Not bit-identical: the .dat is 16-bit quantised while the text is written to
    4 decimals, so a sub-percent difference in a variance-based feature like HRV
    is expected. A large divergence would mean one path reads the signal wrong —
    scaling, channel order or sampling rate.
    """
    import tempfile
    import wfdb

    sig = _synth(seconds=120.0)
    fs = sig["fs"]
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        arr = np.column_stack([sig["ecg"], sig["pleth"], sig["resp"]])
        wfdb.wrsamp(record_name="pair", fs=fs, units=["mV", "NU", "NU"],
                    sig_name=["II", "Pleth", "Resp"], p_signal=arr,
                    fmt=["16", "16", "16"], write_dir=str(d))
        lines = [f"fs: {fs}", "channels: II,Pleth,Resp", "signal:"]
        lines += [",".join(f"{v:.4f}" for v in row) for row in arr]
        (d / "pair.txt").write_text("\n".join(lines) + "\n")

        a = X.extract_from_wfdb_dir(d, record_name="pair").features
        p_ = X.parse_waveform_text((d / "pair.txt").read_text())
        b = X.extract_features(p_["ecg"], p_["pleth"], p_["resp"],
                               fs=p_["fs"]).features

    for f in X.FEATURES:
        va, vb = a[f], b[f]
        assert (va is None) == (vb is None), f"{f}: wfdb={va} text={vb}"
        if va is not None:
            assert abs(va - vb) <= 0.02 * max(abs(vb), 1e-9) + 1e-9, \
                f"{f}: wfdb {va} vs text {vb} differ by more than 2%"


def test_wfdb_dir_without_a_header_is_an_error():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "orphan.dat").write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="hea"):
            X.extract_from_wfdb_dir(t)


def test_wfdb_lead_named_channel_is_found():
    """An exported record names ECG by its lead (II); a hand-built one says ECG.

    Both must be recognised, or one of the two upload routes silently loses the
    ECG-derived features while still reporting success.
    """
    import tempfile
    import wfdb
    s = _synth(seconds=90.0)
    for label in ("II", "ECG"):
        with tempfile.TemporaryDirectory() as t:
            sig = np.column_stack([s["ecg"], s["pleth"]]).astype(np.float64)
            wfdb.wrsamp(record_name="r", fs=s["fs"], units=["mV", "NU"],
                        sig_name=[label, "Pleth"], p_signal=sig,
                        fmt=["16", "16"], write_dir=t)
            res = X.extract_from_wfdb_dir(t)
            assert res.features["HRV_SDNN"] is not None, \
                f"ECG channel named {label!r} was not recognised"


# --------------------------------------------------------------------------- #
# The three curated upload folders
# --------------------------------------------------------------------------- #
# Exactly what a clinician is told to drop on the upload card, per input type.
EXPECTED_INPUTS = {
    "1-tier-a-clinical-only": ["README.txt", "patient-hypercapnic.txt",
                               "patient-hypoxaemic.txt", "patient-volutrauma.txt"],
    "2-tier-b-waveform-txt": ["README.txt", "patient-hyperoxic.txt",
                              "waveform-10min.txt"],
    "3-tier-b-wfdb-record": ["README.txt", "patient-lowtv.txt",
                             "waveform-10min.dat", "waveform-10min.hea"],
}


@pytest.mark.parametrize("folder", sorted(EXPECTED_INPUTS))
def test_input_folder_holds_exactly_its_files(folder):
    """The folders exist to remove ambiguity, so a stray file is a real defect.

    An extra file here is what sends someone uploading the wrong thing; a missing
    one breaks the instructions in that folder's README.
    """
    d = SAMPLES / folder
    if not d.is_dir():
        pytest.skip(f"{folder} not present")
    assert sorted(f.name for f in d.iterdir()) == EXPECTED_INPUTS[folder]


def test_samples_root_holds_only_the_three_input_folders():
    """No loose files at the top level — that was the confusion being removed."""
    if not SAMPLES.is_dir():
        pytest.skip("samples/ not present")
    assert sorted(p.name for p in SAMPLES.iterdir()) == sorted(EXPECTED_INPUTS)


@pytest.mark.parametrize("folder", ["1-tier-a-clinical-only",
                                    "2-tier-b-waveform-txt",
                                    "3-tier-b-wfdb-record"])
def test_every_shipped_patient_file_is_complete(folder):
    """Each clinical file must carry all 12 values plus weight, or upload 422s."""
    d = SAMPLES / folder
    if not d.is_dir():
        pytest.skip(f"{folder} not present")
    for f in sorted(d.glob("patient-*.txt")):
        m = {}
        for line in f.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line and ":" in line:
                k, _, v = line.partition(":")
                m[k.strip()] = v.strip()
        missing = [k for k in config.TABULAR_FEATURES if k not in m]
        assert not missing, f"{folder}/{f.name} is missing {missing}"
        assert "weight" in m, f"{folder}/{f.name} has no weight (needed for mL/kg)"


def test_each_input_folder_uses_a_different_patient():
    """Uploading a different folder must produce a different recommendation.

    All three folders once shipped the SAME patient file, so a user who tried
    folder 2 and then folder 3 saw an identical answer and reasonably concluded
    the engine ignored its input. The files were identical; the engine was fine.
    Distinct patients per folder is what makes that visible.
    """
    seen = {}
    for folder in EXPECTED_INPUTS:
        d = SAMPLES / folder
        if not d.is_dir():
            pytest.skip(f"{folder} not present")
        for f in sorted(d.glob("patient-*.txt")):
            seen.setdefault(f.read_bytes(), []).append(f"{folder}/{f.name}")
    dupes = [v for v in seen.values() if len(v) > 1]
    assert not dupes, f"identical patient files across folders: {dupes}"


def test_both_waveform_inputs_cover_all_six_features():
    """Both Tier B folders must yield 6/6 — that is what their READMEs promise."""
    txt = SAMPLES / "2-tier-b-waveform-txt" / "waveform-10min.txt"
    rec = SAMPLES / "3-tier-b-wfdb-record"
    if not txt.exists() or not (rec / "waveform-10min.hea").exists():
        pytest.skip("Tier B input folders not present")
    p = X.parse_waveform_text(txt.read_text())
    a = X.extract_features(p["ecg"], p["pleth"], p["resp"], fs=p["fs"])
    b = X.extract_from_wfdb_dir(rec)
    assert a.coverage == pytest.approx(1.0), f"text input: {a.features}"
    assert b.coverage == pytest.approx(1.0), f"wfdb input: {b.features}"


WAVEFORM_KEYS = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
                 "RRV", "Breathing_Regularity", "Asynchrony_Score"]


def _patient_file_keys(path: Path) -> dict:
    """The `key: value` pairs a shipped patient file declares, comments stripped."""
    m = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and ":" in line:
            k, _, v = line.partition(":")
            m[k.strip()] = v.strip()
    return m


@pytest.mark.parametrize("folder", sorted(EXPECTED_INPUTS))
def test_no_shipped_patient_file_prefills_waveform_features(folder):
    """A patient file must never hardcode the 6 features. They come from a recording.

    Every file in all three folders used to carry a full waveform block and
    ``track: track_b``. The consequences differed per folder and all three were
    wrong:

    * Folder 1 is advertised as clinical-only, and its README documents Track A
      behaviour — but the files opened on Track B with six features pre-filled.
      Uploading "tabular only" showed populated waveform boxes, which reads as
      the engine inventing data.
    * Folders 2 and 3 exist to demonstrate extraction FROM a recording. The
      upload card overwrites ``patient.waveform`` with the extracted features,
      so the inline block was dead except when the clinical file was uploaded on
      its own — in which case the demo silently skipped the extraction it was
      supposed to be demonstrating and served stale numbers instead. Folder 2's
      block also disagreed with its own recording (Arrhythmia_rate 0.30 typed in
      vs 0.00 extracted), which is where its README's phantom arrhythmia warning
      came from.

    The features belong to the signal, so the signal is the only thing allowed to
    supply them.
    """
    d = SAMPLES / folder
    if not d.is_dir():
        pytest.skip(f"{folder} not present")
    for f in sorted(d.glob("patient-*.txt")):
        m = _patient_file_keys(f)
        present = [k for k in WAVEFORM_KEYS if k in m]
        assert not present, (
            f"{folder}/{f.name} hardcodes {present} — remove the block and let "
            "the uploaded recording supply the features")


@pytest.mark.parametrize("folder", sorted(EXPECTED_INPUTS))
def test_no_shipped_patient_file_preselects_track_b(folder):
    """``track: track_b`` without features is a track the file cannot support.

    Track B is enabled by the presence of a waveform, which now only ever comes
    from an uploaded recording. A file that asks for it up front either gets
    overridden (folders 2 and 3, once the recording is extracted) or lands the
    user on a Track B with nothing behind it (folder 1).
    """
    d = SAMPLES / folder
    if not d.is_dir():
        pytest.skip(f"{folder} not present")
    for f in sorted(d.glob("patient-*.txt")):
        track = _patient_file_keys(f).get("track")
        assert track != "track_b", (
            f"{folder}/{f.name} sets track_b; the upload card sets it when a "
            "recording is supplied")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except (Skip, pytest.skip.Exception) as e:
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
