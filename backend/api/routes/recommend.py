"""POST /api/recommend — Dual-Track inference endpoint (redesign Section 6)."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.api import models as M
from backend.api.state import get_services

router = APIRouter()


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


@router.post("/recommend", response_model=M.RecommendationResponse)
async def recommend(req: M.RecommendationRequest) -> M.RecommendationResponse:
    t0 = time.time()
    svc = get_services()
    ts = req.tabular_state.model_dump()

    try:
        if req.track == "track_a":
            routing = svc.router.run_track_a(ts)
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
            routing = svc.router.run_track_b(ts, wv)
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
                       waveform_used=routing["waveform_used"],
                       waveform_coverage=routing.get("waveform_coverage"),
                       imputation_used=routing.get("imputation_used"))
    return M.RecommendationResponse(
        recommendation=M.Recommendation(
            delta_PEEP=dp, delta_TV=dt, delta_FiO2=df,
            action_text=_action_text(dp, dt, df), track_info=info),
        safety=M.Safety(all_clear=safety.all_clear,
                        flags=[M.SafetyFlag(**f) for f in safety.flags]),
        explanation=M.Explanation(
            top_features=[M.SHAPEntry(**f) for f in feats],
            decision_rule=decision_rule),
        metadata=M.Metadata(latency_ms=int((time.time() - t0) * 1000),
                            timestamp=datetime.now(timezone.utc).isoformat()),
    )
