"""/api/patients — the MongoDB-backed roster and its recommendation history.

The roster the dashboard shows is whatever is in the ``patients`` collection: the
six seeded presets plus everything uploaded since. Uploading POSTs here, deleting a
card DELETEs here (taking that patient's saved recommendations with it), and the
patient page reads both the patient and their history back out.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from backend.api import db
from backend.api import models as M

router = APIRouter()


def _guard(e: Exception) -> HTTPException:
    """Any database problem is a 503 — the API itself is fine, its store is not."""
    return HTTPException(
        status_code=503,
        detail=f"Patient database unavailable: {e}",
    )


def _iso(v: Any) -> Optional[str]:
    if isinstance(v, datetime):
        # Mongo stores naive UTC; label it so the browser renders local time right.
        return v.replace(tzinfo=v.tzinfo or timezone.utc).isoformat()
    return v if isinstance(v, str) else None


def _to_patient(doc: dict, rec_count: int = 0) -> M.Patient:
    return M.Patient(
        id=str(doc["_id"]),
        name=doc.get("name", "Unnamed"),
        weight=doc.get("weight", 75),
        age=doc.get("age", 0),
        sex=doc.get("sex", "M"),
        bed=doc.get("bed", "Uploaded"),
        summary=doc.get("summary", ""),
        state=doc["state"],
        hint=doc.get("hint"),
        waveform=doc.get("waveform"),
        ventilation_mode=doc.get("ventilation_mode"),
        track=doc.get("track"),
        source=doc.get("source", "upload"),
        created_at=_iso(doc.get("created_at")),
        updated_at=_iso(doc.get("updated_at")),
        recommendation_count=rec_count,
    )


@router.get("/patients", response_model=M.PatientList)
async def list_patients() -> M.PatientList:
    """The whole roster, presets first (by seeded order) then uploads oldest-first."""
    try:
        docs = await db.patients().find().to_list(length=500)
        # One aggregate beats N count queries when the roster grows.
        counts = {
            g["_id"]: g["n"]
            async for g in db.recommendations().aggregate(
                [{"$group": {"_id": "$patient_id", "n": {"$sum": 1}}}]
            )
        }
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)

    docs.sort(key=lambda d: (
        0 if d.get("source") == "preset" else 1,
        d.get("order", 0),
        d.get("created_at") or datetime.min.replace(tzinfo=timezone.utc),
    ))
    return M.PatientList(patients=[_to_patient(d, counts.get(str(d["_id"]), 0)) for d in docs])


@router.get("/patients/{patient_id}", response_model=M.Patient)
async def get_patient(patient_id: str) -> M.Patient:
    try:
        doc = await db.patients().find_one({"_id": patient_id})
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No patient {patient_id!r}")
        n = await db.recommendations().count_documents({"patient_id": patient_id})
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)
    return _to_patient(doc, n)


@router.post("/patients", response_model=M.Patient, status_code=201)
async def create_patient(body: M.PatientCreate) -> M.Patient:
    """Save an uploaded patient. Re-posting the same id updates it in place."""
    pid = body.id or f"upload-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    doc = body.model_dump(exclude={"id"})
    doc["state"] = body.state.model_dump()
    doc["waveform"] = body.waveform.model_dump(exclude_none=True) if body.waveform else None
    doc["updated_at"] = now

    try:
        await db.patients().update_one(
            {"_id": pid},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        saved = await db.patients().find_one({"_id": pid})
        n = await db.recommendations().count_documents({"patient_id": pid})
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)
    return _to_patient(saved, n)


@router.delete("/patients/{patient_id}", response_model=M.DeleteResult)
async def delete_patient(patient_id: str) -> M.DeleteResult:
    """Remove a patient from the roster along with every recommendation saved for them.

    Nothing is retained: seeding only refills an empty roster, so the delete stands
    on its own without any record that it happened.
    """
    try:
        removed = await db.patients().delete_one({"_id": patient_id})
        if removed.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"No patient {patient_id!r}")
        history = await db.recommendations().delete_many({"patient_id": patient_id})
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)
    return M.DeleteResult(id=patient_id, deleted=True,
                          recommendations_deleted=history.deleted_count)


def to_record(doc: dict) -> M.RecommendationRecord:
    return M.RecommendationRecord(
        id=str(doc["_id"]),
        patient_id=doc["patient_id"],
        patient_name=doc.get("patient_name"),
        clinician=doc.get("clinician"),
        created_at=_iso(doc.get("created_at")) or "",
        track=doc.get("track", "track_a"),
        patient_weight=doc.get("patient_weight", 0),
        responsiveness=doc.get("responsiveness", 0.0),
        ventilation_mode=doc.get("ventilation_mode"),
        state=doc["state"],
        waveform=doc.get("waveform"),
        delta_PEEP=doc["delta_PEEP"],
        delta_TV=doc["delta_TV"],
        delta_FiO2=doc["delta_FiO2"],
        action_text=doc.get("action_text", ""),
        confidence=doc.get("confidence"),
        safety_all_clear=doc.get("safety_all_clear", True),
        safety_flags=doc.get("safety_flags", []),
        latency_ms=doc.get("latency_ms"),
    )


@router.get("/patients/{patient_id}/recommendations", response_model=M.RecommendationHistory)
async def patient_recommendations(
    patient_id: str,
    limit: int = Query(25, ge=1, le=200),
) -> M.RecommendationHistory:
    """Newest first — every set of settings this patient has been asked about."""
    try:
        docs = await (db.recommendations()
                      .find({"patient_id": patient_id})
                      .sort("created_at", DESCENDING)
                      .limit(limit)
                      .to_list(length=limit))
    except (db.DatabaseUnavailable, PyMongoError) as e:
        raise _guard(e)
    return M.RecommendationHistory(patient_id=patient_id,
                                   records=[to_record(d) for d in docs])
