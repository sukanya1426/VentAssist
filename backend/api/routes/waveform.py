"""/api/waveform — turn an uploaded raw waveform into the 6 Track B features.

This is the step Track B was missing. The 18-dim state needs HRV, arrhythmia
rate, perfusion index, RRV, breathing regularity and an asynchrony score; the
extractors for all six have always been in ``backend/waveform/``, but only the
offline pipeline could reach them, so the only way to use Track B was to compute
those numbers by hand and type them into the patient file. A clinician with a
recording had no route in.

Now the recording itself is the input: POST the signal here, get the features
back with a per-channel quality report, and hand them to ``/api/recommend`` as
the waveform block — or let the upload card do both in one step.

Extraction is deliberately the same code path the training features came from
(see ``extract_live``), because a served feature computed differently from the
trained one puts the 18-dim state in the wrong units.

Nothing is stored here. The endpoint is a pure transform, so a waveform can be
inspected before any patient is created from it.
"""

from __future__ import annotations

import logging

import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api import models as M
from backend.waveform import extract_live as X

router = APIRouter()
log = logging.getLogger(__name__)

# A 1-hour, 3-channel window at ~62.5 Hz is ~675k numbers ≈ 8 MB of text. 32 MB
# leaves generous headroom while still refusing a file that was never a waveform.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

# A WFDB record folder is binary and denser: a trimmed 10-minute single-segment
# record is ~2 MB, while a full multi-segment mimic4wdb record is ~157 MB across
# ~100 files. 256 MB accepts a generously-sized record without letting an
# accidental whole-dataset drop through.
MAX_RECORD_BYTES = 256 * 1024 * 1024
MAX_RECORD_FILES = 256
# Only the two extensions a record is made of. Anything else in a dropped folder
# (.csv exports, notes, .DS_Store) is ignored rather than failing the upload.
WFDB_SUFFIXES = {".hea", ".dat"}


def _result(res: X.ExtractionResult) -> M.WaveformExtraction:
    return M.WaveformExtraction(
        features=M.WaveformFeatures(**{k: v for k, v in res.features.items()
                                       if v is not None}),
        fs=res.fs,
        n_samples=res.n_samples,
        duration_s=res.duration_s,
        coverage=round(res.coverage, 4),
        channels={
            name: M.WaveformChannelReport(
                present=rep.present, valid_fraction=rep.valid_fraction,
                quality_ok=rep.quality_ok, note=rep.note)
            for name, rep in res.channels.items()
        },
        warnings=res.warnings,
    )


@router.post("/waveform/extract", response_model=M.WaveformExtraction)
async def extract_from_file(file: UploadFile = File(...)) -> M.WaveformExtraction:
    """Upload a waveform file → the 6 features.

    Accepts the documented ``fs``/``channels``/``signal`` text format or a bare
    CSV whose header names the channels. A partly-unusable recording is NOT an
    error: the features it could not support come back absent, with the reason in
    ``channels`` and ``warnings``, so the caller can decide whether to proceed on
    reduced coverage.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Waveform file is {len(raw) / 1e6:.1f} MB; the limit is "
                   f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=415,
            detail="This looks like a binary file. A raw WFDB .dat cannot be read "
                   "directly — export its samples as CSV (one column per channel) "
                   "with an 'fs:' and 'channels:' header.")
    try:
        parsed = X.parse_waveform_text(text)
        res = X.extract_features(ecg=parsed["ecg"], pleth=parsed["pleth"],
                                 resp=parsed["resp"], fs=parsed["fs"])
    except ValueError as e:
        # A malformed waveform is the caller's input problem, not a server fault.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:                       # pragma: no cover - defensive
        log.exception("waveform extraction failed for %s", file.filename)
        raise HTTPException(status_code=500,
                            detail=f"Could not extract features: {e}")
    log.info("Extracted waveform features from %s: %.0fs @ %.4g Hz, coverage %.0f%%",
             file.filename, res.duration_s, res.fs, 100 * res.coverage)
    return _result(res)


@router.post("/waveform/extract-record", response_model=M.WaveformExtraction)
async def extract_from_record(files: List[UploadFile] = File(...)) -> M.WaveformExtraction:
    """Upload a WFDB record folder (.hea + .dat) → the 6 features.

    This is the dataset's own format, so a recording can be handed over as-is
    instead of being transcribed into text. Drop the record's directory: the
    header and every ``.dat`` it references must arrive together, because the
    header is what says how the binary is laid out and which channel is which.

    Files are written to a private temporary directory under their original
    names — WFDB resolves segments by filename, so the names carry meaning — and
    the directory is removed as soon as extraction finishes. Nothing is retained.
    """
    picked = [f for f in files
              if Path(f.filename or "").suffix.lower() in WFDB_SUFFIXES]
    if not picked:
        raise HTTPException(
            status_code=422,
            detail="No .hea or .dat files in the upload. A WFDB record is a .hea "
                   "header plus the .dat signal files it references.")
    if not any(Path(f.filename or "").suffix.lower() == ".hea" for f in picked):
        raise HTTPException(
            status_code=422,
            detail="No .hea header in the upload. The .dat files alone cannot be "
                   "read — the header defines the layout, channels and rate.")
    if len(picked) > MAX_RECORD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"{len(picked)} files; the limit is {MAX_RECORD_FILES}. Upload a "
                   "single trimmed record rather than a whole dataset directory.")

    total = 0
    with tempfile.TemporaryDirectory(prefix="ventassist-wfdb-") as tmp:
        for f in picked:
            # Flatten to a bare basename: a dropped folder sends paths like
            # "82697127/82697127.hea", and anything with .. or a leading / must
            # not be allowed to choose where it lands.
            name = os.path.basename((f.filename or "").replace("\\", "/"))
            if not name or name.startswith("."):
                continue
            raw = await f.read()
            total += len(raw)
            if total > MAX_RECORD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Record exceeds {MAX_RECORD_BYTES / 1e6:.0f} MB. Export a "
                           "shorter window — the features need minutes, not days.")
            (Path(tmp) / name).write_bytes(raw)
        try:
            res = X.extract_from_wfdb_dir(tmp)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:                   # pragma: no cover - defensive
            log.exception("WFDB extraction failed")
            raise HTTPException(status_code=500,
                                detail=f"Could not read the record: {e}")
    log.info("Extracted from WFDB record (%d files, %.1f MB): %.0fs @ %.4g Hz, "
             "coverage %.0f%%", len(picked), total / 1e6, res.duration_s, res.fs,
             100 * res.coverage)
    return _result(res)


@router.post("/waveform/extract-samples", response_model=M.WaveformExtraction)
async def extract_from_samples(body: M.WaveformSamples) -> M.WaveformExtraction:
    """Same transform, for a caller that already holds the samples as arrays.

    Exists so the frontend (and any script) can pass parsed signal without
    re-serialising it into a file.
    """
    try:
        res = X.extract_features(ecg=body.ECG, pleth=body.Pleth, resp=body.Resp,
                                 fs=body.fs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _result(res)
