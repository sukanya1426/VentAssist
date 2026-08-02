"""POST /api/recommend — Dual-Track inference endpoint (redesign Section 6)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from backend.api import auth as A
from backend.api import db
from backend.api import models as M
from backend.api.state import get_services

router = APIRouter()
log = logging.getLogger(__name__)


def _action_text(dp: int, dt: int, df: float) -> str:
    def part(name, v, unit, prec=0):
        if v == 0:
            return f"Hold {name}"
        amt = f"{abs(v):.{prec}f}" if prec else f"{abs(v)}"
        unit_txt = f" {unit}" if unit else ""
        return f"{'Increase' if v > 0 else 'Decrease'} {name} by {amt}{unit_txt}"
    return (f"{part('PEEP', dp, 'cmH₂O')} · "
            f"{part('tidal volume', dt, 'mL')} · "
            f"{part('FiO₂', df, '', prec=2)}")


async def _save(req: M.RecommendationRequest, resp: M.RecommendationResponse,
                waveform: Optional[dict], clinician: Optional[str] = None) -> Optional[str]:
    """File this ask under its patient. One document per request, so re-asking with
    changed settings appends a second record rather than replacing the first.

    The signed-in clinician is stored alongside it: the history is a record of who
    asked what and when, which is the only reason it is worth keeping.

    A storage failure must not cost the clinician the recommendation they just
    computed, so this swallows its errors and reports the result as unsaved.
    """
    if not db.is_configured():
        return None
    rec = resp.recommendation
    doc = {
        "patient_id": req.patient_id,
        "patient_name": req.patient_name,
        "clinician": clinician,
        "created_at": datetime.now(timezone.utc),
        "track": rec.track_info.track,
        "patient_weight": req.patient_weight,
        "responsiveness": req.responsiveness,
        "ventilation_mode": req.ventilation_mode,
        "state": req.tabular_state.model_dump(),
        "waveform": {k: v for k, v in (waveform or {}).items() if v is not None} or None,
        "delta_PEEP": rec.delta_PEEP,
        "delta_TV": rec.delta_TV,
        "delta_FiO2": rec.delta_FiO2,
        "action_text": rec.action_text,
        "confidence": rec.track_info.confidence,
        "safety_all_clear": resp.safety.all_clear,
        "safety_flags": [f.model_dump() for f in resp.safety.flags],
        "latency_ms": resp.metadata.latency_ms,
    }
    try:
        res = await db.recommendations().insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        log.warning("Could not save recommendation for %s: %s", req.patient_id, e)
        return None


@router.post("/recommend", response_model=M.RecommendationResponse)
async def recommend(
    req: M.RecommendationRequest,
    # Declared again even though the router is already guarded: the guard makes the
    # route unreachable without a session, this makes *who* it was reachable by
    # available to the record we file.
    user: A.CurrentUser = Depends(A.current_user),
) -> M.RecommendationResponse:
    t0 = time.time()
    svc = get_services()
    ts = req.tabular_state.model_dump()

    wv: Optional[dict] = None

    try:
        if req.track == "track_a":
            routing = svc.router.run_track_a(ts, responsiveness=req.responsiveness,
                                             ventilation_mode=req.ventilation_mode)
        else:
            wv = {}
            if req.ecg_features:
                wv.update(req.ecg_features.model_dump())
            if req.pleth_features:
                wv.update(req.pleth_features.model_dump())
            if req.resp_features:
                wv.update(req.resp_features.model_dump())
            wv = {k: wv.get(k) for k in
                  ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
                   "RRV", "Breathing_Regularity", "Asynchrony_Score"]}
            routing = svc.router.run_track_b(ts, wv, responsiveness=req.responsiveness,
                                             ventilation_mode=req.ventilation_mode)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    dp, dt, df = routing["delta_PEEP"], routing["delta_TV"], routing["delta_FiO2"]

    arr = req.ecg_features.Arrhythmia_rate if req.ecg_features else None
    safety = svc.safety.check(ts, dp, dt, df, req.patient_weight, arr)

    from backend.explainability.attribution import top_features
    feats = top_features(routing["model"], routing["state_norm"],
                         routing["action"], routing["feature_order"])
    rule = (svc.tree_a if routing.get("decision_tree", "a") == "a" else svc.tree_b)
    decision_rule = rule.decision_rule(routing["state_norm"]) if rule else "n/a"

    info = M.TrackInfo(track=routing["track"], track_label=routing["track_label"],
                       confidence=routing["confidence"],
                       decision_margin=routing.get("decision_margin"),
                       waveform_used=routing["waveform_used"],
                       waveform_coverage=routing.get("waveform_coverage"),
                       imputation_used=routing.get("imputation_used"),
                       ventilation_mode=routing.get("ventilation_mode"),
                       tv_masked=routing.get("tv_masked"),
                       in_support=routing.get("in_support"),
                       support_ratio=routing.get("support_ratio"))
    alternatives = [
        M.AlternativeAction(
            delta_PEEP=a["delta_PEEP"], delta_TV=a["delta_TV"],
            delta_FiO2=a["delta_FiO2"],
            action_text=_action_text(a["delta_PEEP"], a["delta_TV"], a["delta_FiO2"]),
            margin_from_best=round(a["margin_from_best"], 4))
        for a in routing.get("alternatives", [])
    ]
    resp = M.RecommendationResponse(
        recommendation=M.Recommendation(
            delta_PEEP=dp, delta_TV=dt, delta_FiO2=df,
            action_text=_action_text(dp, dt, df), track_info=info,
            alternatives=alternatives),
        safety=M.Safety(all_clear=safety.all_clear,
                        flags=[M.SafetyFlag(**f) for f in safety.flags]),
        explanation=M.Explanation(
            top_features=[M.SHAPEntry(**f) for f in feats],
            decision_rule=decision_rule),
        metadata=M.Metadata(latency_ms=int((time.time() - t0) * 1000),
                            timestamp=datetime.now(timezone.utc).isoformat()),
    )
    resp.record_id = await _save(req, resp, wv, clinician=user.username)
    return resp
