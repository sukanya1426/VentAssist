"""/api/patients — the PostgreSQL-backed roster and its recommendation history.

The roster the dashboard shows is whatever is in the ``patient`` table: the six
seeded presets plus everything uploaded since. Uploading POSTs here, deleting a
card DELETEs here (taking that patient's saved recommendations with it), and the
patient page reads both the patient and their history back out.

Reads go through the two views in ``schema.sql`` rather than the base tables, so
the attributes the ER diagram marks as DERIVED are computed by the database:
``patient_with_counts`` supplies RecommendationCount, and ``recommendation_full``
resolves ΔPEEP/ΔTV/ΔFiO₂ by joining the ACTION space that ``action_idx`` points
at. Neither is stored, so neither can disagree with the rows behind it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Query

from backend.api import db
from backend.api import models as M
from backend.mdp import action_masking

router = APIRouter()

# API field name → column name. The composite State/Waveform attributes are flat
# columns in the table, so this mapping is the single place the two spellings meet.
STATE_FIELDS = {"PEEP": "peep", "TV": "tv", "FiO2": "fio2", "SpO2": "spo2",
                "PaO2": "pao2", "PaCO2": "paco2", "pH": "ph", "HR": "hr",
                "SBP": "sbp", "RR": "rr", "RASS": "rass", "Temp": "temp"}
WAVEFORM_FIELDS = {"HRV_SDNN": "hrv_sdnn", "Arrhythmia_rate": "arrhythmia_rate",
                   "Perfusion_Index": "perfusion_index", "RRV": "rrv",
                   "Breathing_Regularity": "breathing_regularity",
                   "Asynchrony_Score": "asynchrony_score"}


def _guard(e: Exception) -> HTTPException:
    """Any database problem is a 503 — the API itself is fine, its store is not."""
    return HTTPException(
        status_code=503,
        detail=f"Patient database unavailable: {e}",
    )


def _iso(v: Any) -> Optional[str]:
    if isinstance(v, datetime):
        # Columns are TIMESTAMPTZ, so this is already aware; the fallback only
        # matters if the column type is ever loosened.
        return v.replace(tzinfo=v.tzinfo or timezone.utc).isoformat()
    return v if isinstance(v, str) else None


def _state(row: asyncpg.Record) -> dict:
    return {api: float(row[col]) for api, col in STATE_FIELDS.items()}


def _waveform(row: asyncpg.Record) -> Optional[dict]:
    """The 6 waveform fields, or None when the patient has no waveform data.

    All six are written together, so a single NULL means "no waveform" rather
    than a partially-filled block.
    """
    vals = {api: row[col] for api, col in WAVEFORM_FIELDS.items()}
    if all(v is None for v in vals.values()):
        return None
    return {k: float(v) for k, v in vals.items() if v is not None}


def _to_patient(row: asyncpg.Record) -> M.Patient:
    return M.Patient(
        id=row["patient_id"],
        name=row["name"],
        weight=row["weight_kg"] if row["weight_kg"] is not None else 75,
        age=row["age"] or 0,
        sex=row["sex"] or "M",
        bed=row["bed"] or "Uploaded",
        summary=row["summary"] or "",
        state=_state(row),
        hint=row["hint"],
        waveform=_waveform(row),
        ventilation_mode=row["ventilation_mode"],
        track=row["track"],
        source=row["source"],
        created_at=_iso(row["created_at"]),
        updated_at=_iso(row["updated_at"]),
        recommendation_count=row["recommendation_count"],
    )


@router.get("/patients", response_model=M.PatientList)
async def list_patients() -> M.PatientList:
    """The whole roster, presets first (by seeded order) then uploads oldest-first.

    The ordering is done in SQL by the index that backs it, and the per-patient
    recommendation counts come from the view in one pass rather than N queries.
    """
    try:
        rows = await db.fetch(
            """
            SELECT * FROM patient_with_counts
            ORDER BY (source <> 'preset'), display_order NULLS LAST, created_at
            LIMIT 500
            """
        )
    except db.DB_ERRORS as e:
        raise _guard(e)
    return M.PatientList(patients=[_to_patient(r) for r in rows])


@router.get("/patients/{patient_id}", response_model=M.Patient)
async def get_patient(patient_id: str) -> M.Patient:
    try:
        row = await db.fetchrow(
            "SELECT * FROM patient_with_counts WHERE patient_id = $1", patient_id)
    except db.DB_ERRORS as e:
        raise _guard(e)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id!r}")
    return _to_patient(row)


@router.post("/patients", response_model=M.Patient, status_code=201)
async def create_patient(body: M.PatientCreate) -> M.Patient:
    """Save an uploaded patient. Re-posting the same id updates it in place."""
    pid = body.id or f"upload-{uuid.uuid4().hex[:12]}"
    state = body.state.model_dump()
    wave = body.waveform.model_dump() if body.waveform else {}

    # Built as one upsert so re-uploading the same id cannot briefly leave the
    # roster without that patient. created_at is preserved on update; updated_at
    # moves — the row records when it first appeared and when it last changed.
    cols = (["patient_id", "name", "bed", "summary", "hint", "age", "sex",
             "weight_kg"]
            + list(STATE_FIELDS.values()) + list(WAVEFORM_FIELDS.values())
            + ["ventilation_mode", "track", "source"])
    values = (
        [pid, body.name, body.bed, body.summary, body.hint, body.age, body.sex,
         float(body.weight)]
        + [float(state[k]) for k in STATE_FIELDS]
        + [(float(wave[k]) if wave.get(k) is not None else None) for k in WAVEFORM_FIELDS]
        # Stored as the canonical category, not the raw string: the API accepts any
        # ventilator-mode text (e.g. "PCV+") and the classifier is what turns it
        # into the two values masking can act on. Storing the raw text would mean
        # the column's enum rejects perfectly valid uploads.
        + [action_masking.classify_ventilator_mode(body.ventilation_mode),
           body.track, body.source]
    )
    # Enum columns need an explicit cast: asyncpg sends text, and Postgres will
    # not coerce text into an enum domain on its own inside a parameterised insert.
    casts = {"ventilation_mode": "::vent_mode_t", "track": "::track_t",
             "source": "::patient_source_t"}
    placeholders = ", ".join(f"${i + 1}{casts.get(c, '')}"
                             for i, c in enumerate(cols))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "patient_id")

    try:
        await db.execute(
            f"""
            INSERT INTO patient ({', '.join(cols)}) VALUES ({placeholders})
            ON CONFLICT (patient_id) DO UPDATE
               SET {updates}, updated_at = now()
            """,
            *values,
        )
        saved = await db.fetchrow(
            "SELECT * FROM patient_with_counts WHERE patient_id = $1", pid)
    except asyncpg.CheckViolationError as e:
        # A range check fired: the value is outside what the policy can be asked about.
        raise HTTPException(status_code=422,
                            detail=f"Patient values out of accepted range: {e}")
    except db.DB_ERRORS as e:
        raise _guard(e)
    return _to_patient(saved)


@router.delete("/patients/{patient_id}", response_model=M.DeleteResult)
async def delete_patient(patient_id: str) -> M.DeleteResult:
    """Remove a patient from the roster along with every recommendation saved for them.

    One statement: ``recommendation`` cascades from ``patient`` and ``safety_flag``
    cascades from ``recommendation``, so the history cannot outlive the patient.
    The count is taken first because the cascade leaves nothing to count after.
    Nothing is retained: seeding only refills an empty roster, so the delete stands
    on its own without any record that it happened.
    """
    try:
        pool = await db.get_pool()
        async with pool.acquire() as con:
            async with con.transaction():
                n = await con.fetchval(
                    "SELECT COUNT(*) FROM recommendation WHERE patient_id = $1",
                    patient_id)
                removed = await con.fetchval(
                    "DELETE FROM patient WHERE patient_id = $1 RETURNING patient_id",
                    patient_id)
    except db.DB_ERRORS as e:
        raise _guard(e)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id!r}")
    return M.DeleteResult(id=patient_id, deleted=True,
                          recommendations_deleted=int(n))


def to_record(row: asyncpg.Record, flags: list[dict]) -> M.RecommendationRecord:
    return M.RecommendationRecord(
        id=str(row["recommendation_id"]),
        patient_id=row["patient_id"],
        patient_name=row["patient_name"],
        clinician=row["clinician_username"],
        created_at=_iso(row["created_at"]) or "",
        track=row["track"],
        patient_weight=row["patient_weight"],
        responsiveness=row["responsiveness"],
        ventilation_mode=row["ventilation_mode"],
        state=_state(row),
        waveform=_waveform(row),
        # Derived from action_idx by the recommendation_full view.
        delta_PEEP=row["delta_peep"],
        delta_TV=row["delta_tv"],
        delta_FiO2=row["delta_fio2"],
        action_text=row["action_text"] or "",
        confidence=row["confidence"],
        safety_all_clear=row["safety_all_clear"],
        safety_flags=flags,
        latency_ms=row["latency_ms"],
    )


@router.get("/patients/{patient_id}/recommendations", response_model=M.RecommendationHistory)
async def patient_recommendations(
    patient_id: str,
    limit: int = Query(25, ge=1, le=200),
) -> M.RecommendationHistory:
    """Newest first — every set of settings this patient has been asked about."""
    try:
        rows = await db.fetch(
            """
            SELECT * FROM recommendation_full
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            patient_id, limit,
        )
        # One query for every flag in the page rather than one per record.
        ids = [r["recommendation_id"] for r in rows]
        flag_rows = await db.fetch(
            """
            SELECT recommendation_id, level, message
            FROM safety_flag
            WHERE recommendation_id = ANY($1::BIGINT[])
            ORDER BY recommendation_id, position
            """,
            ids,
        ) if ids else []
    except db.DB_ERRORS as e:
        raise _guard(e)

    by_rec: dict[int, list[dict]] = {}
    for f in flag_rows:
        by_rec.setdefault(f["recommendation_id"], []).append(
            {"level": f["level"], "message": f["message"]})
    return M.RecommendationHistory(
        patient_id=patient_id,
        records=[to_record(r, by_rec.get(r["recommendation_id"], [])) for r in rows])
