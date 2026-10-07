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
from backend.mdp import action_masking

# Column order shared by the two inserts below. The composite State/Waveform
# attributes are flat columns, so the mapping from API field name to column lives
# in one place.
_STATE_ORDER = ["PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2",
                "pH", "HR", "SBP", "RR", "RASS", "Temp"]
_STATE_COLS = ["peep", "tv", "fio2", "spo2", "pao2", "paco2",
               "ph", "hr", "sbp", "rr", "rass", "temp"]
_WAVE_ORDER = ["HRV_SDNN", "Arrhythmia_rate", "Perfusion_Index",
               "RRV", "Breathing_Regularity", "Asynchrony_Score"]
_WAVE_COLS = ["hrv_sdnn", "arrhythmia_rate", "perfusion_index",
              "rrv", "breathing_regularity", "asynchrony_score"]

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
                waveform: Optional[dict], action_idx: int,
                clinician_id: Optional[str] = None,
                clinician: Optional[str] = None) -> Optional[str]:
    """File this ask under its patient. One row per request, so re-asking with
    changed settings appends a second record rather than replacing the first.

    The signed-in clinician is stored alongside it: the history is a record of who
    asked what and when, which is the only reason it is worth keeping.

    Only ``action_idx`` is stored, never the three deltas — they are derived by
    joining the ACTION table, so a stored recommendation cannot drift from the
    action space it chose from. The flags go to their own child table because
    SafetyFlags is a multivalued attribute; both inserts share one transaction so
    a recommendation can never be left holding a partial set of flags.

    A storage failure must not cost the clinician the recommendation they just
    computed, so this swallows its errors and reports the result as unsaved. It is
    also why the patient row is created on demand: asking about a patient that is
    not on the roster (a preset the clinician never uploaded, say) should still
    produce a recommendation rather than a foreign-key error.
    """
    if not db.is_configured():
        return None
    rec = resp.recommendation
    state = req.tabular_state.model_dump()
    wave = waveform or {}
    try:
        pool = await db.get_pool()
        async with pool.acquire() as con:
            async with con.transaction():
                # The history references the roster, so an ask about an unknown
                # patient id would violate the FK. Record a minimal row instead of
                # losing the history for it.
                await con.execute(
                    """
                    INSERT INTO patient (patient_id, name, weight_kg, peep, tv, fio2,
                        spo2, pao2, paco2, ph, hr, sbp, rr, rass, temp, source)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                            'upload'::patient_source_t)
                    ON CONFLICT (patient_id) DO NOTHING
                    """,
                    req.patient_id, req.patient_name or req.patient_id,
                    req.patient_weight,
                    *[float(state[k]) for k in _STATE_ORDER],
                )
                # The FK would reject a user_id with no clinician row, losing the
                # whole record. A token outlives its account (it is signature-
                # checked, not looked up), so this is reachable in normal use —
                # and the clinical record must survive a deleted account, which is
                # the same reason the column is ON DELETE SET NULL. Resolve it to
                # NULL instead of failing; clinician_username keeps who asked.
                known_user = clinician_id and await con.fetchval(
                    "SELECT user_id FROM clinician WHERE user_id = $1", clinician_id)
                rec_id = await con.fetchval(
                    f"""
                    INSERT INTO recommendation (
                        patient_id, user_id, patient_name, clinician_username,
                        action_idx,
                        {', '.join(_STATE_COLS)},
                        {', '.join(_WAVE_COLS)},
                        track, patient_weight, responsiveness, ventilation_mode,
                        action_text, confidence, safety_all_clear, latency_ms,
                        created_at)
                    VALUES ($1,$2,$3,$4,$5,
                            $6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
                            $18,$19,$20,$21,$22,$23,
                            $24::track_t,$25,$26,$27::vent_mode_t,
                            $28,$29,$30,$31,$32)
                    RETURNING recommendation_id
                    """,
                    req.patient_id, known_user, req.patient_name, clinician,
                    int(action_idx),
                    *[float(state[k]) for k in _STATE_ORDER],
                    *[(float(wave[k]) if wave.get(k) is not None else None)
                      for k in _WAVE_ORDER],
                    rec.track_info.track, req.patient_weight, req.responsiveness,
                    action_masking.classify_ventilator_mode(req.ventilation_mode),
                    rec.action_text, rec.track_info.confidence,
                    resp.safety.all_clear, resp.metadata.latency_ms,
                    datetime.now(timezone.utc),
                )
                if resp.safety.flags:
                    await con.executemany(
                        """
                        INSERT INTO safety_flag (recommendation_id, level, message,
                                                 position)
                        VALUES ($1, $2::safety_level_t, $3, $4)
                        """,
                        [(rec_id, f.level, f.message, i)
                         for i, f in enumerate(resp.safety.flags)],
                    )
        return str(rec_id)
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
                       waveform_influence=routing.get("waveform_influence"),
                       waveform_informative=routing.get("waveform_informative"),
                       delegated_to_track_a=routing.get("delegated_to_track_a"),
                       ventilation_mode=routing.get("ventilation_mode"),
                       tv_masked=routing.get("tv_masked"),
                       in_support=routing.get("in_support"),
                       support_ratio=routing.get("support_ratio"))
    alternatives = [
        M.AlternativeAction(
            delta_PEEP=a["delta_PEEP"], delta_TV=a["delta_TV"],
            delta_FiO2=a["delta_FiO2"],
            action_text=_action_text(a["delta_PEEP"], a["delta_TV"], a["delta_FiO2"]),
            margin_from_best=round(a["margin_from_best"], 4),
            preferred_by_policy=a["margin_from_best"] < 0)
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
    resp.record_id = await _save(req, resp, wv, routing["action"],
                                 clinician_id=user.id, clinician=user.username)
    return resp
